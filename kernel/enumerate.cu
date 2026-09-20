/*
 * One-colour queen placements on the GPU.
 *
 * This kernel is the device version of `enumerate` in src/main.rs. For each
 * row set of a batch (an "item") it visits every placement of one queen per
 * row on squares of one colour, mutually non-attacking, and tallies each
 * complete placement into the item's table:
 *
 *   parity 0 (even squares, rows R):  +1 at the index of the placement's
 *                                     column set C, building E[R, C] for all C;
 *   parity 1 (odd squares, rows R̅):   +weight at the index of the complement
 *                                     of the placement's column set.
 *
 * The complement mapping aligns the two tables entry for entry, so the item's
 * result, sum over C of E[R, C] * O[R̅, C̅], is their dot product (join.cu).
 * Tallying is a fire-and-forget atomic, which the GPU handles far faster than
 * the random table reads a direct evaluation of the sum would need. Table
 * indexing is SolutionTable's (src/solution_table.rs): split the column mask
 * into dense even/odd masks, colex-rank each with the two-level rank tables,
 * index = rank(even) * odd_stride + rank(odd).
 *
 * Work division. The work is cut into tickets. A ticket fixes the columns of
 * the first NQ_PREFIX rows of its item as a mixed-radix number, digit t
 * choosing among the colour-compatible columns of row t. A ticket whose
 * prefix has a conflict is finished at once; the others run a depth-first
 * search over the remaining rows, identical to the CPU search. Item i owns
 * the tickets [ticket_base[i], ticket_base[i + 1]).
 *
 * Threads are persistent and the kernel body is one loop, each iteration of
 * which performs a single search step of the thread's ticket, or picks up a
 * new ticket. Tickets are handed out in order from a global counter, so the
 * tickets in progress at any moment are a narrow window of consecutive
 * tickets, i.e. of few items, and the tables being updated stay within the
 * L2 cache. (Letting each thread claim a run of tickets widens the window
 * and was measured to be several times slower.)
 *
 * Ticket setup (finding the item, decoding the prefix, discarding dead
 * tickets) is a long path, and most tickets are dead at the depths that
 * keep the window narrow, so it must not run one lane at a time while the
 * rest of the warp waits. Instead each warp keeps a pool of decoded
 * prefixes: when the pool is empty and a lane needs work, the warp claims
 * WARP consecutive tickets and every lane decodes one, together, keeping
 * the live ones (`refill_pool`); a lane then takes a live prefix with a few
 * shared-memory loads. The bookkeeping (which lanes need work, which pool
 * slots are live) is done with warp votes at the top of every iteration, so
 * it is uniform across the warp. One loop for every kind of work keeps the
 * lanes executing together even though tickets differ enormously in cost.
 *
 * The search state lives in registers: the columns taken, the two diagonal
 * families, the row being placed and the columns of it still to try. The
 * diagonals are kept in board coordinates (see `diag1_bit`), which makes
 * removing a queen the same XOR as placing it, so backing up recomputes the
 * parent's state from the child's instead of reloading it; all a stack
 * frame has to remember is the column it placed, one byte per frame per
 * thread in shared memory. Shared memory per thread would otherwise decide
 * how many warps a multiprocessor can hold; with byte frames the register
 * count does.
 *
 * Mirror symmetry (NQ_MIRROR, set by the host for odd n): as in the CPU
 * `backtrack_odd_solutions`, the odd-square search places its first row's
 * queen only in columns up to the centre; a placement left of centre stands
 * for itself and its mirror image (weight 2), one in the centre column for
 * itself (weight 1). The first prefix digit ranges over just those columns.
 *
 * Baked in by the host at compile time: NQ_N (board size), NQ_PREFIX (prefix
 * depth), NQ_BLOCK (threads per block), NQ_FRAMES (stack frames per thread,
 * at least the largest row count minus NQ_PREFIX minus 1: the last row of a
 * row set never pushes) and NQ_MIRROR (0 or 1).
 * kernel/emulate.cpp compiles this file as host C++ for the emulation test.
 */
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;

/* ---- board geometry ---- */
#define FULL      ((1u << NQ_N) - 1u)
#define EVEN_COLS (FULL & 0x55555555u)
#define ODD_COLS  (FULL & 0xAAAAAAAAu)
#define M_EVEN    ((NQ_N + 1u) / 2u)          /* number of even columns */
#define M_ODD     (NQ_N / 2u)                 /* number of odd columns */
#define EVEN_FULL ((1u << M_EVEN) - 1u)       /* dense mask of all even columns */
#define ODD_FULL  ((1u << M_ODD) - 1u)        /* dense mask of all odd columns */
#define MID       ((NQ_N - 1u) / 2u)          /* the centre column (odd n) */

/* Layout of the `counters` argument: the ticket counter the threads claim
 * from, then the statistics they add up at exit. Must match COUNTER_* in
 * src/gpu/plan.rs. */
enum { NEXT_TICKET = 0, TICKETS, DEAD, STEPS, LEAVES, N_COUNTERS };

/* ---- bit helpers ---- */
__device__ __forceinline__ u32 lowest_bit(u32 x) { return x & (0u - x); }
__device__ __forceinline__ u32 lowest_set(u32 x) { return __ffs(x) - 1u; }    /* x != 0 */
__device__ __forceinline__ u32 highest_set(u32 x) { return 31u - __clz(x); }  /* x != 0 */

/* Colour-compatible columns of row r: even columns when r has the colour's
 * parity, odd columns otherwise (`pmask` in the CPU enumerate). */
__device__ __forceinline__ u32 row_columns(u32 r, u32 parity) {
    return ((r & 1u) == parity) ? EVEN_COLS : ODD_COLS;
}

/* ---- diagonals in board coordinates ----
 *
 * A queen at (r, c) lies on the / diagonal c - r and the \ diagonal c + r.
 * The diagonals occupied by the queens placed so far are two 64-bit masks
 * with bit c - r + (NQ_N - 1) set in the first and bit c + r in the second
 * (2n - 1 diagonals each, so n <= 32 fits). The columns a family attacks
 * in row r are then a shift of its mask, and a queen is placed or removed
 * by the same XOR whatever the current row. */
__device__ __forceinline__ u64 diag1_bit(u32 bit, u32 r) { return (u64)bit << (NQ_N - 1u - r); }
__device__ __forceinline__ u64 diag2_bit(u32 bit, u32 r) { return (u64)bit << r; }

/* Columns of row r attacked by the queens in `cols`, `d1`, `d2` (or taken). */
__device__ __forceinline__ u32 attacked(u32 cols, u64 d1, u64 d2, u32 r) {
    return cols | (u32)(d1 >> (NQ_N - 1u - r)) | (u32)(d2 >> r);
}

/* ---- table indexing, as in solution_table.rs ---- */

/* Gathers the even-position bits of x into the low 16 bits (bit 2i -> bit i). */
__device__ __forceinline__ u32 gather_even_bits(u32 x) {
    x &= 0x55555555u;
    x = (x | (x >> 1)) & 0x33333333u;
    x = (x | (x >> 2)) & 0x0F0F0F0Fu;
    x = (x | (x >> 4)) & 0x00FF00FFu;
    x = (x | (x >> 8)) & 0x0000FFFFu;
    return x;
}

/* Colex rank of a dense (<= 16-bit) mask; rank_hi is RANK_HI flattened,
 * indexed [hi * 9 + popcount(lo)]. */
__device__ __forceinline__ u32 rank16(u32 dense, const u16* rank_lo, const u16* rank_hi) {
    const u32 lo = dense & 0xFFu;
    const u32 hi = dense >> 8;
    return (u32)rank_lo[lo] + (u32)rank_hi[hi * 9u + __popc(lo)];
}

/* Index of the column set `cols` (parity 0) or of its complement (parity 1)
 * in a table with the given odd stride. */
__device__ __forceinline__ u32 table_index(u32 cols, u32 parity, u32 odd_stride,
                                           const u16* rank_lo, const u16* rank_hi) {
    u32 even = gather_even_bits(cols);
    u32 odd = gather_even_bits(cols >> 1);
    if (parity == 1u) {
        even = ~even & EVEN_FULL;
        odd = ~odd & ODD_FULL;
    }
    return rank16(even, rank_lo, rank_hi) * odd_stride + rank16(odd, rank_lo, rank_hi);
}

/* ---- tickets ---- */

#define WARP 32u
#define FULL_WARP 0xffffffffu
#define WARPS_PER_BLOCK ((NQ_BLOCK + WARP - 1u) / WARP)
#define POOL ((NQ_BLOCK < WARP) ? NQ_BLOCK : WARP)   /* tickets claimed per refill: one per lane */

/* The item owning ticket t, searched for forward from item `from`: tickets
 * are claimed in increasing order, so a thread's item never moves backwards
 * and is usually `from` itself or the next one. Gallops to bracket the
 * ticket, then bisects; a couple of cached reads of ticket_base. */
__device__ __forceinline__ u32 find_item(const u64* ticket_base, u32 n_items, u32 from, u64 t) {
    u32 lo = from, hi = from + 1u;             /* ticket_base[lo] <= t; want t < ticket_base[hi] */
    for (u32 step = 1u; ticket_base[hi] <= t; step *= 2u) {
        lo = hi;
        hi = (hi + step < n_items) ? hi + step : n_items;
    }
    while (hi - lo > 1u) {
        const u32 mid = (lo + hi) / 2u;
        if (ticket_base[mid] <= t) lo = mid; else hi = mid;
    }
    return lo;
}

/* The number of columns a ticket digit ranges over: the columns of the
 * parity the row's colour requires, or, with the mirror symmetry, for the
 * first row of an odd-square search only those up to the centre. Must match
 * `ticket_count` in src/gpu/plan.rs. */
__device__ __forceinline__ u32 digit_radix(u32 col_parity, u32 parity, bool first) {
    u32 radix = (col_parity == 0u) ? M_EVEN : M_ODD;
    if (NQ_MIRROR && first && parity == 1u) radix = (MID - col_parity) / 2u + 1u;
    return radix;
}

/* What a ticket's prefix fixes. */
struct Prefix {
    u32 cols;           /* columns taken by the prefix queens */
    u64 d1, d2;         /* their diagonals */
    u32 rest;           /* rows still to place; 0: the prefix is the whole placement */
    u32 weight;         /* placements each leaf below stands for (mirror symmetry) */
    bool dead;          /* two prefix queens attack each other: nothing to count */
};

/* Decodes ticket `local` of an item with row set `rows`: the columns of its
 * first min(NQ_PREFIX, |rows|) rows, placed as the CPU search would. */
__device__ __forceinline__ Prefix decode_prefix(u32 local, u32 rows, u32 parity) {
    Prefix p = {0u, 0ull, 0ull, rows, 1u, false};
    const u32 l = __popc(rows);
    const u32 depth = (NQ_PREFIX < l) ? NQ_PREFIX : l;
    for (u32 d = 0; d < depth && !p.dead; ++d) {
        const u32 r = lowest_set(p.rest);
        p.rest &= p.rest - 1u;
        const u32 col_parity = (r & 1u) ^ parity;              /* this row's columns have c % 2 == col_parity */
        const u32 radix = digit_radix(col_parity, parity, d == 0u);
        const u32 c = 2u * (local % radix) + col_parity;
        local /= radix;
        const u32 bit = 1u << c;
        if (NQ_MIRROR && d == 0u && parity == 1u) p.weight = (c == MID) ? 1u : 2u;
        p.dead = (attacked(p.cols, p.d1, p.d2, r) & bit) != 0u;
        p.cols |= bit;
        p.d1 |= diag1_bit(bit, r);
        p.d2 |= diag2_bit(bit, r);
    }
    return p;
}

/* ---- the pool of decoded prefixes ---- */

/* Slot k of a warp's pool is filled by lane k at a refill. */
__shared__ u32 pool_item[WARPS_PER_BLOCK][WARP];
__shared__ u32 pool_cols[WARPS_PER_BLOCK][WARP];
__shared__ u64 pool_d1[WARPS_PER_BLOCK][WARP];
__shared__ u64 pool_d2[WARPS_PER_BLOCK][WARP];
__shared__ u32 pool_rest[WARPS_PER_BLOCK][WARP];
__shared__ u32 pool_weight[WARPS_PER_BLOCK][WARP];

/* Per-thread statistics, added to `counters` when the block is done. */
struct Stats {
    u32 tickets, dead;
    u64 steps, leaves;

    __device__ __forceinline__ void flush(u64* counters) const {
        atomicAdd(counters + TICKETS, (u64)tickets);
        atomicAdd(counters + DEAD, (u64)dead);
        atomicAdd(counters + STEPS, steps);
        atomicAdd(counters + LEAVES, leaves);
    }
};

/* Refills the calling warp's pool, all lanes together: lane 0 claims POOL
 * consecutive tickets, lane k decodes ticket k of them, and the prefixes
 * that have a search below them go into the pool. Prefixes that are the
 * whole placement are tallied here. Returns the mask of live pool slots,
 * or sets `exhausted` when the claimed tickets are past the end of the
 * work. `item` is the lane's last item, the starting point of the search
 * for the ticket's item. */
__device__ __forceinline__ u32 refill_pool(
    u64* counter, u64 total, u32 lane, u32 warp, u32 parity, u32& item, bool& exhausted, Stats& stats,
    const u32* item_rows, const u64* item_table, const u32* item_odd_stride, const u64* ticket_base, u32 n_items,
    u32* tables, const u16* rank_lo, const u16* rank_hi)
{
    u64 base = 0;
    if (lane == 0u) base = atomicAdd(counter, (u64)POOL);
    base = __shfl_sync(FULL_WARP, base, 0);
    exhausted = base >= total;
    const u64 t = base + lane;
    bool live = false;
    if (t < total && lane < POOL) {
        ++stats.tickets;
        item = find_item(ticket_base, n_items, item, t);
        const Prefix p = decode_prefix((u32)(t - ticket_base[item]), item_rows[item], parity);
        if (p.dead) {
            ++stats.dead;
        } else if (p.rest == 0u) {             /* the prefix is the whole placement (also |R| = 0) */
            atomicAdd(tables + item_table[item] + table_index(p.cols, parity, item_odd_stride[item], rank_lo, rank_hi), p.weight);
            ++stats.leaves;
        } else {
            pool_item[warp][lane] = item;
            pool_cols[warp][lane] = p.cols;
            pool_d1[warp][lane] = p.d1;
            pool_d2[warp][lane] = p.d2;
            pool_rest[warp][lane] = p.rest;
            pool_weight[warp][lane] = p.weight;
            live = true;
        }
    }
    return __ballot_sync(FULL_WARP, live);
}

/* ---- the search ---- */

/* The DFS stack: the column placed by frame f of the calling thread, at
 * byte [f][threadIdx.x]. Only frames below the top have placed a column. */
__shared__ unsigned char st_col[NQ_FRAMES][NQ_BLOCK];

/* The search below a ticket's prefix. The queens placed so far, by the
 * prefix and by the frames below the top, are in `cols`, `d1` and `d2`;
 * the top frame is trying the columns of `row` still in `free`, lowest
 * first. Backing up to a frame removes its queen by XOR and recomputes the
 * columns it has still to try: the free ones above the one it placed. */
struct Search {
    u32 cols;
    u64 d1, d2;
    u32 row;            /* the row the top frame places */
    u32 free;           /* columns of `row` still to try */

    /* Moves to row r: every colour-compatible column not attacked is free. */
    __device__ __forceinline__ void enter(u32 r, u32 parity) {
        row = r;
        free = row_columns(r, parity) & ~attacked(cols, d1, d2, r);
    }

    /* Places a queen in column `bit` of `row` and moves on to `next_row`. */
    __device__ __forceinline__ void push(u32 bit, u32 next_row, u32 parity) {
        cols |= bit;
        d1 |= diag1_bit(bit, row);
        d2 |= diag2_bit(bit, row);
        enter(next_row, parity);
    }

    /* Backs up to `prev_row`, whose queen was in column `bit`. */
    __device__ __forceinline__ void pop(u32 bit, u32 prev_row, u32 parity) {
        row = prev_row;
        cols ^= bit;
        d1 ^= diag1_bit(bit, row);
        d2 ^= diag2_bit(bit, row);
        free = row_columns(row, parity) & ~attacked(cols, d1, d2, row) & ~((bit << 1u) - 1u);
    }
};

/* The item and weight of the ticket a thread is working on. */
struct Ticket {
    u32 rows;           /* the item's row set: R (parity 0) or R̅ (parity 1) */
    u32 odd_stride;     /* geometry of the item's table */
    u32* table;         /* the item's table */
    u32 weight;         /* placements each leaf stands for */

    /* Counts a complete placement with column set `cols`. */
    __device__ __forceinline__ void tally(u32 cols, u32 parity, const u16* rank_lo, const u16* rank_hi) const {
        atomicAdd(table + table_index(cols, parity, odd_stride, rank_lo, rank_hi), weight);
    }
};

extern "C" __global__ void enumerate_kernel(
    u32 parity,                                /* colour: 0 for r + c even, 1 for r + c odd */
    u32 n_items,
    const u32* __restrict__ item_rows,         /* [n_items] row set of each item (R for parity 0, R̅ for parity 1) */
    const u64* __restrict__ item_table,        /* [n_items] offset of the item's table in `tables` */
    const u32* __restrict__ item_odd_stride,   /* [n_items] odd stride of that table */
    const u64* __restrict__ ticket_base,       /* [n_items + 1] first ticket of each item; last entry = total */
    u32* __restrict__ tables,                  /* this colour's tables for the batch, concatenated; zero at launch */
    const u16* __restrict__ rank_lo,
    const u16* __restrict__ rank_hi,
    u64* __restrict__ counters)                /* [N_COUNTERS] all zero at launch */
{
    const u32 tid = threadIdx.x;
    const u32 lane = tid % WARP, warp = tid / WARP;
    const u32 lanes_below = (1u << lane) - 1u;
    const u32 all_lanes = __ballot_sync(FULL_WARP, true);   /* the lanes that exist (the emulator has one) */
    const u64 total = ticket_base[n_items];

    u32 item = 0;                              /* the lane's last item; only ever increases */
    Ticket tk = {0u, 0u, tables, 1u};
    int f = -1;                                /* top frame of the search in progress; -1: none */
    Search s;
    Stats stats = {0u, 0u, 0ull, 0ull};
    u32 live = 0;                              /* pool slots holding a prefix nobody has taken (warp-uniform) */
    bool exhausted = false;                    /* the counter is past the end of the work (warp-uniform) */

    for (;;) {
        /* ---- hand out work: warp-uniform bookkeeping, then the lanes that need it pick up ---- */
        const u32 need = __ballot_sync(FULL_WARP, f < 0);
        if (need != 0u) {
            if (live == 0u && !exhausted) {
                live = refill_pool(counters + NEXT_TICKET, total, lane, warp, parity, item, exhausted, stats,
                                   item_rows, item_table, item_odd_stride, ticket_base, n_items, tables, rank_lo, rank_hi);
            }
            if (live == 0u && exhausted && need == all_lanes) break;   /* nothing left for anyone */
            const u32 n_live = (u32)__popc(live), n_need = (u32)__popc(need);
            const u32 rank = (u32)__popc(need & lanes_below);       /* my place among the lanes that need work */
            if (f < 0 && rank < n_live) {
                const u32 slot = __fns(live, 0u, (int)rank + 1);    /* the rank-th live slot */
                const u32 it = pool_item[warp][slot];
                tk.rows = item_rows[it];
                tk.odd_stride = item_odd_stride[it];
                tk.table = tables + item_table[it];
                tk.weight = pool_weight[warp][slot];
                s.cols = pool_cols[warp][slot];
                s.d1 = pool_d1[warp][slot];
                s.d2 = pool_d2[warp][slot];
                s.enter(lowest_set(pool_rest[warp][slot]), parity);
                f = 0;
            }
            for (u32 taken = (n_need < n_live) ? n_need : n_live; taken > 0u; --taken) {
                live &= live - 1u;                                  /* the lowest slots were taken */
            }
            if (f < 0) continue;                                    /* still nothing to do: wait for the next refill */
        }

        /* ---- one search step, as in the CPU enumerate ---- */
        ++stats.steps;
        if (s.free == 0u) {                    /* the row is exhausted: back up */
            if (--f >= 0) {
                const u32 bit = 1u << st_col[f][tid];
                s.pop(bit, highest_set(tk.rows & ((1u << s.row) - 1u)), parity);   /* to the previous row of R */
            }
            continue;
        }
        const u32 bit = lowest_bit(s.free);    /* try the lowest remaining column */
        s.free ^= bit;
        const u32 later = tk.rows >> (s.row + 1u);   /* rows of R after this one */
        if (later == 0u) {                     /* the last row: a complete placement */
            tk.tally(s.cols | bit, parity, rank_lo, rank_hi);
            ++stats.leaves;
            continue;
        }
        st_col[f][tid] = (unsigned char)lowest_set(bit);
        ++f;
        s.push(bit, s.row + __ffs(later), parity);   /* __ffs(later): distance to the next row of R */
    }
    stats.flush(counters);
}

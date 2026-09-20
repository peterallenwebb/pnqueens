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
 * which either sets up a freshly claimed ticket or performs a single search
 * step of the ticket in progress. Tickets are claimed one at a time from a
 * shared counter, in order, so the tickets in progress at any moment are a
 * narrow window of consecutive tickets, i.e. of few items, and the tables
 * being updated stay within the L2 cache. (Claiming tickets in chunks widens
 * the window and was measured to be several times slower.) One loop for
 * every kind of work keeps the lanes of a warp executing together even
 * though tickets differ enormously in cost.
 *
 * The DFS stack lives in shared memory: frame f of a thread is word
 * [f][threadIdx.x] of four arrays, so the threads of a warp touch
 * consecutive words. A frame holds the columns taken so far, the two
 * diagonal families shifted to the frame's row, and the columns of that row
 * still to try; the row itself is kept in a register and recomputed on
 * backing up. Shared memory per thread decides how many warps a
 * multiprocessor can hold, so frames are kept small.
 *
 * Mirror symmetry (NQ_MIRROR, set by the host for odd n): as in the CPU
 * `backtrack_odd_solutions`, the odd-square search places its first row's
 * queen only in columns up to the centre; a placement left of centre stands
 * for itself and its mirror image (weight 2), one in the centre column for
 * itself (weight 1). The first prefix digit ranges over just those columns.
 *
 * Baked in by the host at compile time: NQ_N (board size), NQ_PREFIX (prefix
 * depth), NQ_BLOCK (threads per block), NQ_FRAMES (stack frames per thread,
 * at least the largest row count minus NQ_PREFIX) and NQ_MIRROR (0 or 1).
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
    u32 cols, ld, rd;   /* columns taken, and the diagonals as seen from the first row still to place */
    u32 rest;           /* rows still to place; 0: the prefix is the whole placement */
    u32 weight;         /* placements each leaf below stands for (mirror symmetry) */
    bool dead;          /* two prefix queens attack each other: nothing to count */
};

/* Decodes ticket `local` of an item with row set `rows`: the columns of its
 * first min(NQ_PREFIX, |rows|) rows, placed as the CPU search would. */
__device__ __forceinline__ Prefix decode_prefix(u32 local, u32 rows, u32 parity) {
    Prefix p = {0u, 0u, 0u, rows, 1u, false};
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
        p.dead = ((p.cols | p.ld | p.rd) & bit) != 0u;
        const u32 gap = (p.rest != 0u) ? lowest_set(p.rest) - r : 0u;   /* to the next row to place */
        p.cols |= bit;
        p.ld = (p.ld | bit) << gap;
        p.rd = (p.rd | bit) >> gap;
    }
    return p;
}

/* ---- the search ---- */

/* The DFS stack: frame f of the calling thread is word [f][threadIdx.x]. */
__shared__ u32 st_cols[NQ_FRAMES][NQ_BLOCK];   /* columns taken by the rows above */
__shared__ u32 st_ld[NQ_FRAMES][NQ_BLOCK];     /* columns attacked along / diagonals, at this row */
__shared__ u32 st_rd[NQ_FRAMES][NQ_BLOCK];     /* columns attacked along \ diagonals, at this row */
__shared__ u32 st_free[NQ_FRAMES][NQ_BLOCK];   /* columns of this row still to try */

/* Writes frame f for `row`, given the columns taken and the diagonals as
 * seen from that row; every colour-compatible column not attacked is free. */
__device__ __forceinline__ void push_frame(int f, u32 row, u32 parity, u32 cols, u32 ld, u32 rd) {
    const u32 tid = threadIdx.x;
    st_cols[f][tid] = cols;
    st_ld[f][tid] = ld;
    st_rd[f][tid] = rd;
    st_free[f][tid] = row_columns(row, parity) & ~(cols | ld | rd);
}

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

/* Per-thread statistics, added to `counters` when the thread runs out of tickets. */
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
    const u64 total = ticket_base[n_items];

    u32 item = 0;                              /* item of the current ticket; only ever increases */
    Ticket tk = {0u, 0u, tables, 1u};
    int f = -1;                                /* top frame of the search in progress; -1: none */
    u32 row = 0;                               /* the row the top frame places */
    Stats stats = {0u, 0u, 0ull, 0ull};

    for (;;) {
        if (f < 0) {
            /* ---- claim the next ticket and set it up ---- */
            const u64 t = atomicAdd(counters + NEXT_TICKET, 1ull);
            if (t >= total) break;
            ++stats.tickets;
            item = find_item(ticket_base, n_items, item, t);
            tk.rows = item_rows[item];
            tk.odd_stride = item_odd_stride[item];
            tk.table = tables + item_table[item];
            const Prefix p = decode_prefix((u32)(t - ticket_base[item]), tk.rows, parity);
            tk.weight = p.weight;
            if (p.dead) {
                ++stats.dead;
            } else if (p.rest == 0u) {         /* the prefix is the whole placement (also |R| = 0) */
                tk.tally(p.cols, parity, rank_lo, rank_hi);
                ++stats.leaves;
            } else {
                row = lowest_set(p.rest);
                f = 0;
                push_frame(f, row, parity, p.cols, p.ld, p.rd);
            }
            continue;
        }

        /* ---- one search step, as in the CPU enumerate ---- */
        ++stats.steps;
        const u32 free = st_free[f][tid];
        if (free == 0u) {                      /* the row is exhausted: back up */
            if (--f >= 0) row = highest_set(tk.rows & ((1u << row) - 1u));   /* the previous row of R */
            continue;
        }
        const u32 bit = lowest_bit(free);      /* try the lowest remaining column */
        st_free[f][tid] = free ^ bit;
        const u32 cols = st_cols[f][tid] | bit;
        const u32 later = tk.rows >> (row + 1u);   /* rows of R after this one */
        if (later == 0u) {                     /* the last row: a complete placement */
            tk.tally(cols, parity, rank_lo, rank_hi);
            ++stats.leaves;
            continue;
        }
        const u32 gap = __ffs(later);          /* distance to the next row of R */
        const u32 ld = (st_ld[f][tid] | bit) << gap;
        const u32 rd = (st_rd[f][tid] | bit) >> gap;
        ++f;
        row += gap;
        push_frame(f, row, parity, cols, ld, rd);
    }
    stats.flush(counters);
}

/*
 * One-colour queen placements on the GPU.
 *
 * This kernel is the device version of `enumerate` in src/main.rs: for a row
 * set R it visits every placement of one queen per row of R on squares of one
 * colour, mutually non-attacking, and at each leaf either
 *
 *   mode 0 (tabulate): adds 1 to the item's table at the index of the leaf's
 *                      column set (this builds E[R, C] for all C), or
 *   mode 1 (read):     adds the table entry at the index of the *complement*
 *                      of the leaf's column set to a per-thread accumulator,
 *                      which is added to the item's sum when the ticket is
 *                      done (this computes sum over C of E[R, C] * O[R̅, C̅]).
 *
 * Table indexing is exactly SolutionTable's (src/solution_table.rs): split the
 * column mask into dense even/odd masks, colex-rank each with the two-level
 * tables, index = rank(even) * odd_stride + rank(odd).
 *
 * Work division: the work is cut into "tickets". A ticket fixes the columns
 * of the first NQ_PREFIX rows of R as a mixed-radix number (digit t chooses
 * among the colour-compatible columns of row t). Tickets whose prefix has a
 * conflict are finished at once; the others run a depth-first search over
 * the remaining rows, identical to the CPU search. Many items share one
 * launch: item i owns the tickets [ticket_base[i], ticket_base[i + 1]), and
 * a ticket finds its item by binary search.
 *
 * Thread organization: threads are persistent and the kernel body is one
 * loop. Each iteration performs a single search step for the thread's
 * current ticket; a thread whose search is exhausted takes the next
 * unclaimed ticket from a shared counter and sets up its prefix, inside the
 * same loop. Keeping every thread in the same loop keeps the lanes of a warp
 * executing together, which matters because tickets differ enormously in
 * cost.
 *
 * The DFS stack lives in shared memory: a thread's frame f is word
 * [f][threadIdx.x] of five arrays, so the threads of a warp touch consecutive
 * words (no bank conflicts). A frame holds the row being placed, the columns
 * taken so far, the two diagonal masks shifted to that row, and the columns
 * of that row still to try.
 *
 * Mirror symmetry (NQ_MIRROR, set by the host for odd n): as in the CPU
 * `backtrack_odd_solutions`, the read-side search places its first row's
 * queen only in columns up to the centre; a placement left of centre counts
 * for itself and its mirror image (weight 2), one in the centre column for
 * itself (weight 1). The first prefix digit ranges over just those columns.
 *
 * Baked in by the host at compile time: NQ_N (board size), NQ_PREFIX (prefix
 * depth), NQ_BLOCK (threads per block), NQ_FRAMES (stack frames per thread,
 * at least the largest row count minus NQ_PREFIX) and NQ_MIRROR (0 or 1).
 * All items in a launch share one mode and one colour.
 */
typedef unsigned int u32;
typedef unsigned long long u64;

#define FULL      ((1u << NQ_N) - 1u)
#define EVEN_COLS (FULL & 0x55555555u)
#define ODD_COLS  (FULL & 0xAAAAAAAAu)
#define M_EVEN    ((NQ_N + 1u) / 2u)          /* number of even columns */
#define M_ODD     (NQ_N / 2u)                 /* number of odd columns */
#define EVEN_FULL ((1u << M_EVEN) - 1u)       /* dense mask of all even columns */
#define ODD_FULL  ((1u << M_ODD) - 1u)        /* dense mask of all odd columns */
#define MID       ((NQ_N - 1u) / 2u)          /* the centre column (odd n) */

/* Gathers the even-position bits of x into the low 16 bits (bit 2i -> bit i).
 * Same as solution_table::gather_even_bits. */
__device__ __forceinline__ u32 gather_even_bits(u32 x) {
    x &= 0x55555555u;
    x = (x | (x >> 1)) & 0x33333333u;
    x = (x | (x >> 2)) & 0x0F0F0F0Fu;
    x = (x | (x >> 4)) & 0x00FF00FFu;
    x = (x | (x >> 8)) & 0x0000FFFFu;
    return x;
}

/* Colex rank of a dense (<= 16-bit) mask: solution_table::rank.
 * rank_hi is RANK_HI flattened, [hi * 9 + popcount(lo)]. */
__device__ __forceinline__ u32 rank16(u32 dense, const unsigned short* rank_lo, const unsigned short* rank_hi) {
    const u32 lo = dense & 0xFFu;
    const u32 hi = dense >> 8;
    return (u32)rank_lo[lo] + (u32)rank_hi[hi * 9u + __popc(lo)];
}

/* Index into the item's table of the column set `cols` (mode 0), or of its
 * complement (mode 1). Mirrors SolutionTable::index_dense / complement_dense. */
__device__ __forceinline__ u32 table_index(u32 cols, u32 mode, u32 odd_stride,
                                           const unsigned short* rank_lo, const unsigned short* rank_hi) {
    u32 even = gather_even_bits(cols);
    u32 odd = gather_even_bits(cols >> 1);
    if (mode == 1u) {
        even = ~even & EVEN_FULL;
        odd = ~odd & ODD_FULL;
    }
    return rank16(even, rank_lo, rank_hi) * odd_stride + rank16(odd, rank_lo, rank_hi);
}

/* Color-compatible columns of row r: even columns when r has the colour's
 * parity, odd columns otherwise (as `pmask` in the CPU enumerate). */
__device__ __forceinline__ u32 row_columns(u32 r, u32 parity) {
    return ((r & 1u) == parity) ? EVEN_COLS : ODD_COLS;
}

/* The DFS stack: frame f of the calling thread is word [f][threadIdx.x]. */
__shared__ u32 st_row[NQ_FRAMES][NQ_BLOCK];    /* row being placed at this level */
__shared__ u32 st_cols[NQ_FRAMES][NQ_BLOCK];   /* columns taken by the rows above */
__shared__ u32 st_ld[NQ_FRAMES][NQ_BLOCK];     /* columns attacked along / diagonals, at this row */
__shared__ u32 st_rd[NQ_FRAMES][NQ_BLOCK];     /* columns attacked along \ diagonals, at this row */
__shared__ u32 st_free[NQ_FRAMES][NQ_BLOCK];   /* columns of this row still to try */

extern "C" __global__ void enumerate_kernel(
    u32 mode,                                  /* 0: tabulate, 1: read (see above) */
    u32 parity,                                /* colour: 0 for r + c even, 1 for r + c odd */
    u32 n_items,
    const u32* __restrict__ item_rows,         /* [n_items] row set R of each item */
    const u64* __restrict__ item_table,        /* [n_items] offset of the item's table in `tables` */
    const u32* __restrict__ item_odd_stride,   /* [n_items] odd stride of that table */
    const u64* __restrict__ ticket_base,       /* [n_items + 1] first ticket of each item; last entry = total */
    u32* __restrict__ tables,                  /* all tables of the batch, concatenated */
    u64* __restrict__ sums,                    /* [n_items] mode 1 results */
    const unsigned short* __restrict__ rank_lo,
    const unsigned short* __restrict__ rank_hi,
    u64* __restrict__ next_ticket)             /* shared ticket counter, zero at launch */
{
    const u32 tid = threadIdx.x;
    const u64 total = ticket_base[n_items];

    /* The thread's current ticket. */
    u32 item = 0, rows = 0, odd_stride = 0;
    u32* table = tables;
    u32 weight = 1;                            /* mode 1: placements this ticket stands for (mirror symmetry) */
    u64 acc = 0;                               /* mode 1: weighted table entries read for this item so far */
    int f = -1;                                /* top stack frame; -1 = no ticket in progress */

    for (;;) {
        if (f < 0) {
            /* ---- take the next ticket ---- */
            if (mode == 1u && acc != 0ull) { atomicAdd(sums + item, acc); acc = 0; }
            const u64 ticket = atomicAdd(next_ticket, 1ull);
            if (ticket >= total) return;

            u32 lo = 0, hi = n_items;          /* invariant: ticket_base[lo] <= ticket < ticket_base[hi] */
            while (hi - lo > 1u) {
                const u32 mid = (lo + hi) / 2u;
                if (ticket_base[mid] <= ticket) lo = mid; else hi = mid;
            }
            item = lo;
            rows = item_rows[item];
            odd_stride = item_odd_stride[item];
            table = tables + item_table[item];
            u32 local = (u32)(ticket - ticket_base[item]);   /* the host keeps tickets per item below 2^32 */

            /* ---- decode the prefix: the first min(NQ_PREFIX, |R|) rows ---- */
            const u32 l = __popc(rows);
            const u32 depth = (NQ_PREFIX < l) ? NQ_PREFIX : l;
            u32 rest = rows;                   /* rows of R not yet placed */
            u32 cols = 0, ld = 0, rd = 0;
            bool dead = false;
            weight = 1;
            for (u32 t = 0; t < depth && !dead; ++t) {
                const u32 r = __ffs(rest) - 1u;
                rest &= rest - 1u;
                const u32 col_parity = (r & 1u) ^ parity;         /* columns c with c % 2 == col_parity */
                u32 radix = (col_parity == 0u) ? M_EVEN : M_ODD;  /* columns of that parity on the board */
#if NQ_MIRROR
                if (mode == 1u && t == 0u) radix = (MID - col_parity) / 2u + 1u;   /* ... up to the centre */
#endif
                const u32 q = local % radix;
                local /= radix;
                const u32 c = 2u * q + col_parity;
                const u32 bit = 1u << c;
#if NQ_MIRROR
                if (mode == 1u && t == 0u) weight = (c == MID) ? 1u : 2u;
#endif
                dead = ((cols | ld | rd) & bit) != 0u;
                cols |= bit;
                const u32 gap = (rest != 0u) ? (__ffs(rest) - 1u) - r : 0u;
                ld = (ld | bit) << gap;
                rd = (rd | bit) >> gap;
            }
            if (dead) continue;                /* a conflict within the prefix: nothing to count */

            if (rest == 0u) {
                /* The prefix is the whole placement (including |R| == 0: the
                 * single empty placement, cols == 0). */
                const u32 idx = table_index(cols, mode, odd_stride, rank_lo, rank_hi);
                if (mode == 0u) atomicAdd(table + idx, 1u); else acc += (u64)weight * table[idx];
                continue;
            }

            /* ---- start the search at the first row after the prefix ---- */
            const u32 r = __ffs(rest) - 1u;
            f = 0;
            st_row[0][tid] = r;
            st_cols[0][tid] = cols;
            st_ld[0][tid] = ld;
            st_rd[0][tid] = rd;
            st_free[0][tid] = row_columns(r, parity) & ~(cols | ld | rd);
            continue;
        }

        /* ---- one search step, as in the CPU enumerate ---- */
        const u32 free = st_free[f][tid];
        if (free == 0u) {                      /* this row is exhausted: back up */
            --f;
            continue;
        }
        const u32 bit = free & (0u - free);    /* try the lowest remaining column */
        st_free[f][tid] = free ^ bit;
        const u32 c = st_cols[f][tid] | bit;
        const u32 r = st_row[f][tid];
        const u32 later = rows >> (r + 1u);    /* rows of R after r */
        if (later == 0u) {                     /* r is the last row: a complete placement */
            const u32 idx = table_index(c, mode, odd_stride, rank_lo, rank_hi);
            if (mode == 0u) atomicAdd(table + idx, 1u); else acc += (u64)weight * table[idx];
            continue;
        }
        const u32 gap = __ffs(later);          /* distance to the next row of R */
        const u32 nr = r + gap;
        const u32 nld = (st_ld[f][tid] | bit) << gap;
        const u32 nrd = (st_rd[f][tid] | bit) >> gap;
        ++f;
        st_row[f][tid] = nr;
        st_cols[f][tid] = c;
        st_ld[f][tid] = nld;
        st_rd[f][tid] = nrd;
        st_free[f][tid] = row_columns(nr, parity) & ~(c | nld | nrd);
    }
}

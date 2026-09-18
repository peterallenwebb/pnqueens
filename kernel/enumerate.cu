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
 *                      which is added to the item's sum at the end (this
 *                      computes sum over C of E[R, C] * O[R̅, C̅] for the item).
 *
 * Table indexing is exactly SolutionTable's (src/solution_table.rs): split the
 * column mask into dense even/odd masks, colex-rank each with the two-level
 * tables, index = rank(even) * odd_stride + rank(odd).
 *
 * Parallelism: one thread per "ticket". A ticket fixes the columns of the
 * first NQ_PREFIX rows of R as a mixed-radix number (digit t chooses among the
 * colour-compatible columns of row t). Tickets whose prefix has a conflict
 * exit at once; the others run a depth-first search over the remaining rows,
 * identical to the CPU search. Many items share one launch: item i owns the
 * tickets [ticket_base[i], ticket_base[i + 1]), and a thread finds its item by
 * binary search.
 *
 * Baked in by the host at compile time: NQ_N (board size) and NQ_PREFIX
 * (prefix depth). All items in a launch share one mode and one colour.
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
    const unsigned short* __restrict__ rank_hi)
{
    /* ---- which item and which ticket within it ---- */
    const u64 ticket = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (ticket >= ticket_base[n_items]) return;
    u32 lo = 0, hi = n_items;                  /* invariant: ticket_base[lo] <= ticket < ticket_base[hi] */
    while (hi - lo > 1u) {
        const u32 mid = (lo + hi) / 2u;
        if (ticket_base[mid] <= ticket) lo = mid; else hi = mid;
    }
    const u32 item = lo;
    u64 local = ticket - ticket_base[item];
    const u32 rows = item_rows[item];
    const u32 odd_stride = item_odd_stride[item];
    u32* table = tables + item_table[item];
    u64 acc = 0;

    /* ---- per-level data, as in the CPU enumerate: rows of R in increasing
     * order, colour-compatible column mask, gap to the next row of R ---- */
    u32 row[32], pmask[32], gap[32];
    u32 l = 0;
    for (u32 m = rows; m != 0u; m &= m - 1u) row[l++] = __ffs(m) - 1u;
    for (u32 t = 0; t < l; ++t) {
        const u32 r = row[t];
        pmask[t] = ((r & 1u) == parity) ? EVEN_COLS : ODD_COLS;
        gap[t] = (t + 1u < l) ? row[t + 1u] - r : 0u;
    }

    /* ---- the prefix: decode the ticket into the first min(NQ_PREFIX, l) rows ---- */
    const u32 depth = (NQ_PREFIX < l) ? NQ_PREFIX : l;
    u32 cols = 0, ld = 0, rd = 0;
    for (u32 t = 0; t < depth; ++t) {
        const u32 col_parity = (row[t] & 1u) ^ parity;         /* columns c with c % 2 == col_parity */
        const u32 radix = (col_parity == 0u) ? M_EVEN : M_ODD;
        const u32 q = (u32)(local % radix);
        local /= radix;
        const u32 bit = 1u << (2u * q + col_parity);
        if ((cols | ld | rd) & bit) return;                    /* dead prefix: nothing to count */
        cols |= bit;
        ld = (ld | bit) << gap[t];
        rd = (rd | bit) >> gap[t];
    }

    if (depth == l) {
        /* The prefix is the whole placement (this includes l == 0: the single
         * empty placement, cols == 0). */
        const u32 idx = table_index(cols, mode, odd_stride, rank_lo, rank_hi);
        if (mode == 0u) atomicAdd(table + idx, 1u); else acc += table[idx];
    } else {
        /* ---- depth-first search over the remaining rows, as in the CPU
         * enumerate; level t of the stack is row[t] ---- */
        u32 st_free[32], st_cols[32], st_ld[32], st_rd[32];
        u32 t = depth;
        st_cols[t] = cols; st_ld[t] = ld; st_rd[t] = rd;
        st_free[t] = pmask[t] & ~(cols | ld | rd);
        const u32 last = l - 1u;
        for (;;) {
            const u32 free = st_free[t];
            if (free == 0u) {
                if (t == depth) break;
                --t;
                continue;
            }
            const u32 bit = free & (0u - free);
            st_free[t] = free ^ bit;
            const u32 c = st_cols[t] | bit;
            if (t == last) {
                const u32 idx = table_index(c, mode, odd_stride, rank_lo, rank_hi);
                if (mode == 0u) atomicAdd(table + idx, 1u); else acc += table[idx];
                continue;
            }
            const u32 sh = gap[t];
            const u32 nld = (st_ld[t] | bit) << sh;
            const u32 nrd = (st_rd[t] | bit) >> sh;
            ++t;
            st_cols[t] = c; st_ld[t] = nld; st_rd[t] = nrd;
            st_free[t] = pmask[t] & ~(c | nld | nrd);
        }
    }

    if (mode == 1u && acc != 0ull) atomicAdd(sums + item, acc);
}

/*
 * The join: for every item of a batch, the dot product of its two tables,
 * plus the checks on the tables themselves.
 *
 * After the two enumerate_kernel launches of a batch (see enumerate.cu), item
 * i has an E table at e_tables + item_table[i] and an O table at the same
 * offset in o_tables, both item_words[i] words long, two 16-bit entries per
 * word, aligned entry for entry. Its count of full-board solutions is the
 * sum over entries j of E[j] * O[j].
 *
 * Alongside, every block adds the entries it read to `entry_sums` (E and O
 * separately) and the largest entry it saw to `max_entry`. The host
 * compares the sums with the weights the enumerate launches tallied: an
 * entry that overflowed 16 bits would carry into its neighbour and change
 * the sum, so equality proves no entry overflowed.
 *
 * Grid: blockIdx.x is the item, blockIdx.y a chunk of JOIN_CHUNK consecutive
 * words of it; blocks past the end of an item's table exit at once. Each
 * thread sums a strided share of the chunk, warps reduce with shuffles, and
 * each warp adds its part to the item's sum.
 *
 * Baked in by the host at compile time: JOIN_THREADS (threads per block) and
 * JOIN_CHUNK (words per block).
 */
typedef unsigned int u32;
typedef unsigned long long u64;

__device__ __forceinline__ u64 warp_sum(u64 v) {
    for (u32 d = 16; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
    return v;
}
__device__ __forceinline__ u32 warp_max(u32 v) {
    for (u32 d = 16; d > 0; d >>= 1) { const u32 o = __shfl_down_sync(0xffffffffu, v, d); v = (o > v) ? o : v; }
    return v;
}

extern "C" __global__ void join_kernel(
    const u64* __restrict__ item_table,        /* [n_items] word offset of the item's tables */
    const u32* __restrict__ item_words,        /* [n_items] length of the item's tables, in words */
    const u32* __restrict__ e_tables,
    const u32* __restrict__ o_tables,
    u64* __restrict__ sums,                    /* [n_items] results, zero at launch */
    u64* __restrict__ entry_sums,              /* [2] sum of all E entries, of all O entries; zero at launch */
    u32* __restrict__ max_entry)               /* [1] largest entry of either table; zero at launch */
{
    const u32 item = blockIdx.x;
    const u32 words = item_words[item];
    const u32 start = blockIdx.y * JOIN_CHUNK;
    if (start >= words) return;
    const u32 end = (start + JOIN_CHUNK < words) ? start + JOIN_CHUNK : words;
    const u32* e = e_tables + item_table[item];
    const u32* o = o_tables + item_table[item];

    u64 part = 0, sum_e = 0, sum_o = 0;
    u32 largest = 0;
    for (u32 j = start + threadIdx.x; j < end; j += JOIN_THREADS) {
        const u32 ew = e[j], ow = o[j];
        const u32 e0 = ew & 0xFFFFu, e1 = ew >> 16, o0 = ow & 0xFFFFu, o1 = ow >> 16;
        part += (u64)e0 * o0 + (u64)e1 * o1;
        sum_e += e0 + e1;
        sum_o += o0 + o1;
        const u32 m = ((e0 > e1) ? e0 : e1) | ((o0 > o1) ? o0 : o1);   /* an upper bound is enough for a max */
        largest = (m > largest) ? m : largest;
    }

    part = warp_sum(part);
    sum_e = warp_sum(sum_e);
    sum_o = warp_sum(sum_o);
    largest = warp_max(largest);
    if ((threadIdx.x & 31u) == 0u) {
        if (part != 0ull) atomicAdd(sums + item, part);
        if (sum_e != 0ull) atomicAdd(entry_sums, sum_e);
        if (sum_o != 0ull) atomicAdd(entry_sums + 1, sum_o);
        if (largest != 0u) atomicMax(max_entry, largest);
    }
}

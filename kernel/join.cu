/*
 * The join: for every item of a batch, the dot product of its two tables.
 *
 * After the two enumerate_kernel launches of a batch (see enumerate.cu), item
 * i has an E table at e_tables + item_table[i] and an O table at the same
 * offset in o_tables, both item_entries[i] long and aligned entry for entry.
 * Its count of full-board solutions is sum over j of E[j] * O[j].
 *
 * Grid: blockIdx.x is the item, blockIdx.y a chunk of CHUNK consecutive
 * entries of it; blocks past the end of an item's table exit at once. Each
 * thread sums a strided share of the chunk, warps reduce with shuffles, and
 * each warp adds its part to the item's sum.
 */
typedef unsigned int u32;
typedef unsigned long long u64;

#define THREADS 256u
#define CHUNK   (THREADS * 64u)

extern "C" __global__ void join_kernel(
    const u64* __restrict__ item_table,        /* [n_items] offset of the item's tables */
    const u32* __restrict__ item_entries,      /* [n_items] length of the item's tables */
    const u32* __restrict__ e_tables,
    const u32* __restrict__ o_tables,
    u64* __restrict__ sums)                    /* [n_items] results, zero at launch */
{
    const u32 item = blockIdx.x;
    const u32 entries = item_entries[item];
    const u32 start = blockIdx.y * CHUNK;
    if (start >= entries) return;
    const u32 end = (start + CHUNK < entries) ? start + CHUNK : entries;
    const u32* e = e_tables + item_table[item];
    const u32* o = o_tables + item_table[item];

    u64 part = 0;
    for (u32 j = start + threadIdx.x; j < end; j += THREADS) part += (u64)e[j] * o[j];

    for (u32 d = 16; d > 0; d >>= 1) part += __shfl_down_sync(0xffffffffu, part, d);
    if ((threadIdx.x & 31u) == 0u && part != 0ull) atomicAdd(sums + item, part);
}

/*
 * Host build of enumerate.cu for the kernel emulation test
 * (`cargo test --features emulate`, see src/gpu/emulate.rs).
 *
 * The CUDA constructs the kernel uses are given plain C++ meanings, the
 * board macros become runtime values so one build serves every n, and the
 * kernel runs as a block of one thread, which claims every ticket in turn.
 * The join is trivial and is done by the test itself.
 */
#include <cstdint>

#define __device__
#define __forceinline__ inline
#define __global__
#define __launch_bounds__(...)
#define __shared__ static
#define __restrict__ __restrict

struct Dim3 { unsigned x, y, z; };
static Dim3 threadIdx = {0, 0, 0};

static inline int __popc(unsigned x) { return __builtin_popcount(x); }
static inline int __ffs(unsigned x) { return __builtin_ffs((int)x); }
static inline int __clz(unsigned x) { return x ? __builtin_clz(x) : 32; }
template <class T> static T atomicAdd(T* p, T v) { T old = *p; *p += v; return old; }
static inline unsigned __funnelshift_l(unsigned lo, unsigned hi, unsigned s) { s &= 31u; return (unsigned)((((unsigned long long)hi << 32) | lo) << s >> 32); }
static inline unsigned __funnelshift_r(unsigned lo, unsigned hi, unsigned s) { s &= 31u; return (unsigned)((((unsigned long long)hi << 32) | lo) >> s); }
/* Warp intrinsics for a warp of one lane (lane 0). */
template <class T> static T __shfl_sync(unsigned, T v, unsigned) { return v; }
static inline unsigned __ballot_sync(unsigned, bool pred) { return pred ? 1u : 0u; }
static inline void __syncwarp(unsigned = 0xffffffffu) {}
static inline unsigned __fns(unsigned mask, unsigned base, int offset) {   /* position of the offset-th set bit at or after base */
    for (unsigned i = base; i < 32; ++i) if ((mask >> i) & 1u && --offset == 0) return i;
    return 0xffffffffu;
}

static unsigned emu_n, emu_prefix, emu_mirror;
#define NQ_N      emu_n
#define NQ_PREFIX emu_prefix
#define NQ_MIRROR emu_mirror
#define NQ_BLOCK  1u
#define NQ_MIN_BLOCKS 0u
#define NQ_FRAMES 32u

#include "enumerate.cu"

extern "C" void emulate_enumerate(
    unsigned n, unsigned prefix_depth, unsigned mirror,
    u32 parity, u32 n_items, const u32* item_rows, const u64* item_table, const u32* item_odd_stride,
    const u64* ticket_base, u32* tables, const u16* rank_lo, const u16* rank_hi, u64* counters)
{
    emu_n = n;
    emu_prefix = prefix_depth;
    emu_mirror = mirror;
    threadIdx.x = 0;
    enumerate_kernel(parity, n_items, item_rows, item_table, item_odd_stride, ticket_base, tables, rank_lo, rank_hi, counters);
}

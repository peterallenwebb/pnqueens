//! The CUDA driver: compiles the kernels for this board, owns the device
//! buffers, and runs the plan one batch at a time.

use super::plan::{
    default_prefix_depth, uses_mirror, Batch, Launch, COUNTER_DEAD, COUNTER_LEAVES, COUNTER_STEPS, COUNTER_TICKETS, ENTRY_BYTES,
    MIN_PREFIX_DEPTH, N_COUNTERS,
};
use super::GpuOptions;
use crate::solution_table::{RANK_HI, RANK_LO};
use crate::RowCombo;
use cudarc::driver::sys::CUdevice_attribute::{CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;
use std::time::Instant;

const ENUMERATE_SRC: &str = include_str!("../../kernel/enumerate.cu");
const ENUMERATE_MODULE: &str = "pnqueens_enumerate";
const ENUMERATE_KERNEL: &str = "enumerate_kernel";
const JOIN_SRC: &str = include_str!("../../kernel/join.cu");
const JOIN_MODULE: &str = "pnqueens_join";
const JOIN_KERNEL: &str = "join_kernel";

/// Threads per block and table entries per block of the join kernel.
const JOIN_THREADS: u32 = 256;
const JOIN_CHUNK: u32 = JOIN_THREADS * 64;
/// Static shared memory available to one block on every supported GPU.
const SHARED_MEMORY_LIMIT: u32 = 48 * 1024;
/// Bytes of shared memory per stack frame per thread (four u32 words).
const FRAME_BYTES: u32 = 16;
/// The residency the enumerate kernel typically gets, in threads per
/// multiprocessor (eight warps); what the prefix-depth rule assumes.
const RESIDENT_THREADS_PER_SM: u32 = 256;

/// Threads per block and stack frames per thread for the enumerate kernel.
/// A row set has at most n rows, and the DFS below the prefix needs one
/// frame per row after it, so n - prefix frames always suffice. Blocks are
/// 128 threads when their stacks fit the static shared-memory limit, else 64.
fn block_shape(n: u32, prefix_depth: u32) -> (u32, u32) {
    let frames = n.saturating_sub(prefix_depth).max(1);
    let block = if 128 * frames * FRAME_BYTES <= SHARED_MEMORY_LIMIT { 128 } else { 64 };
    (block, frames)
}

/// Number of blocks to launch, and how many of them are resident on each
/// multiprocessor: the kernel's threads are persistent (each loops over
/// tickets), so we launch exactly the blocks that can be resident at once,
/// as reported by the driver for this compiled kernel and block size.
fn grid_size(dev: &CudaDevice, kernel: &CudaFunction, block_size: u32) -> Result<(u32, u32), String> {
    let sms = dev.attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).map_err(cuda_err)? as u32;
    let per_sm = kernel.occupancy_max_active_blocks_per_multiprocessor(block_size, 0, None).map_err(cuda_err)?;
    if per_sm == 0 {
        return Err("the kernel cannot be launched with this block size".into());
    }
    Ok((sms * per_sm, per_sm))
}

/// The enumerate kernel source for this run: the board and launch shape are
/// compile-time constants of the kernel.
fn enumerate_source(n: u32, prefix_depth: u32, block: u32, frames: u32) -> String {
    format!(
        "#define NQ_N {n}u\n#define NQ_PREFIX {prefix_depth}u\n#define NQ_BLOCK {block}u\n#define NQ_FRAMES {frames}u\n#define NQ_MIRROR {}\n{ENUMERATE_SRC}",
        u32::from(uses_mirror(n))
    )
}

fn join_source() -> String {
    format!("#define JOIN_THREADS {JOIN_THREADS}u\n#define JOIN_CHUNK {JOIN_CHUNK}u\n{JOIN_SRC}")
}

/// Compiles a kernel with NVRTC and loads it.
fn load_kernel(dev: &Arc<CudaDevice>, source: String, module: &'static str, name: &'static str) -> Result<CudaFunction, String> {
    let ptx = compile_ptx_with_opts(source, CompileOptions::default()).map_err(|e| format!("{name} compilation failed:\n{e}"))?;
    dev.load_ptx(ptx, module, &[name]).map_err(cuda_err)?;
    dev.get_func(module, name).ok_or_else(|| format!("{name} not found after loading"))
}

fn cuda_err(e: impl std::fmt::Display) -> String {
    format!("CUDA error: {e}")
}

/// A device set up for one board size: the compiled kernels, their launch
/// shape, and the buffers every batch reuses.
pub struct Gpu {
    dev: Arc<CudaDevice>,
    enumerate: CudaFunction,
    join: CudaFunction,
    /// Blocks and threads per block of an enumerate launch.
    grid: u32,
    block: u32,
    prefix_depth: u32,
    rank_lo: CudaSlice<u16>,
    rank_hi: CudaSlice<u16>,
    /// One buffer of tables per colour: parity 0 fills the first, parity 1 the second.
    tables: [CudaSlice<u32>; 2],
    /// Entries in each table buffer.
    budget_entries: usize,
    /// The enumerate kernel's ticket counter and statistics (see N_COUNTERS).
    counters: CudaSlice<u64>,
}

/// What one batch produced.
pub struct BatchResult {
    /// Each item's count of full-board solutions, unweighted.
    pub counts: Vec<u64>,
    /// Statistics of the two enumerate launches.
    pub stats: [LaunchStats; 2],
    pub join_seconds: f64,
}

impl Gpu {
    /// Opens the device, chooses the prefix depth (from the heaviest item
    /// unless overridden), compiles the kernels and allocates the buffers.
    pub fn new(n: u32, opts: &GpuOptions, heaviest: Option<&RowCombo>) -> Result<Self, String> {
        let dev = CudaDevice::new(opts.device).map_err(cuda_err)?;
        println!("device: CUDA {} ({})", opts.device, dev.name().map_err(cuda_err)?);

        let prefix_depth = match (opts.prefix_depth, heaviest) {
            (Some(depth), _) => depth,
            (None, None) => MIN_PREFIX_DEPTH,
            (None, Some(heaviest)) => {
                let sms = dev.attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).map_err(cuda_err)? as u32;
                let l2_bytes = dev.attribute(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE).map_err(cuda_err)? as u64;
                default_prefix_depth(n, heaviest, sms * RESIDENT_THREADS_PER_SM, l2_bytes)
            }
        };
        if prefix_depth == 0 {
            return Err("prefix depth must be at least 1".into());
        }
        // The kernel decodes a ticket's position within its item in 32 bits.
        let max_tickets = u64::from(n.div_ceil(2)).pow(prefix_depth);
        if max_tickets > u64::from(u32::MAX) {
            return Err(format!("prefix depth {prefix_depth} gives up to {max_tickets} tickets per row set; the limit is 2^32"));
        }

        let (block, frames) = block_shape(n, prefix_depth);
        let enumerate = load_kernel(&dev, enumerate_source(n, prefix_depth, block, frames), ENUMERATE_MODULE, ENUMERATE_KERNEL)?;
        let (grid, blocks_per_sm) = grid_size(&dev, &enumerate, block)?;
        println!(
            "gpu: prefix depth {prefix_depth}, table budget {} MiB, {grid} blocks of {block} threads ({blocks_per_sm} per SM), {frames} stack frames ({} KiB shared memory per block)",
            opts.table_budget_bytes >> 20,
            block * frames * FRAME_BYTES / 1024
        );
        let join = load_kernel(&dev, join_source(), JOIN_MODULE, JOIN_KERNEL)?;

        let rank_lo = dev.htod_sync_copy(&RANK_LO).map_err(cuda_err)?;
        let rank_hi_flat: Vec<u16> = RANK_HI.iter().flatten().copied().collect();
        let rank_hi = dev.htod_sync_copy(&rank_hi_flat).map_err(cuda_err)?;
        // Half the budget for each colour's tables.
        let budget_entries = opts.table_budget_bytes / ENTRY_BYTES / 2;
        let tables = [dev.alloc_zeros(budget_entries).map_err(cuda_err)?, dev.alloc_zeros(budget_entries).map_err(cuda_err)?];
        let counters = dev.alloc_zeros(N_COUNTERS).map_err(cuda_err)?;
        Ok(Gpu { dev, enumerate, join, grid, block, prefix_depth, rank_lo, rank_hi, tables, budget_entries, counters })
    }

    pub fn prefix_depth(&self) -> u32 {
        self.prefix_depth
    }

    /// Table memory available per colour, in entries.
    pub fn budget_entries(&self) -> usize {
        self.budget_entries
    }

    /// Runs the three launches of a batch: the even-square search of every
    /// item, the odd-square search, then the join. All go to the same
    /// stream, so each sees the tables the previous one completed.
    pub fn run_batch(&mut self, batch: &Batch) -> Result<BatchResult, String> {
        for tables in self.tables.iter_mut() {
            let mut used = tables.slice_mut(0..batch.used_entries.max(1));
            self.dev.memset_zeros(&mut used).map_err(cuda_err)?;
        }
        let stats = [self.enumerate(0, &batch.launches[0])?, self.enumerate(1, &batch.launches[1])?];
        let (counts, join_seconds) = self.join(batch)?;
        Ok(BatchResult { counts, stats, join_seconds })
    }

    /// One enumerate launch, timed, with its kernel counters collected.
    fn enumerate(&mut self, parity: usize, launch: &Launch) -> Result<LaunchStats, String> {
        let dev = &self.dev;
        let n_items = launch.rows.len() as u32;
        let rows = dev.htod_sync_copy(&launch.rows).map_err(cuda_err)?;
        let table_offset = dev.htod_sync_copy(&launch.table_offset).map_err(cuda_err)?;
        let odd_stride = dev.htod_sync_copy(&launch.odd_stride).map_err(cuda_err)?;
        let ticket_base = dev.htod_sync_copy(&launch.ticket_base).map_err(cuda_err)?;
        dev.memset_zeros(&mut self.counters).map_err(cuda_err)?;
        dev.synchronize().map_err(cuda_err)?;

        let start = Instant::now();
        let cfg = LaunchConfig { grid_dim: (self.grid, 1, 1), block_dim: (self.block, 1, 1), shared_mem_bytes: 0 };
        let params = (
            parity as u32,
            n_items,
            &rows,
            &table_offset,
            &odd_stride,
            &ticket_base,
            &mut self.tables[parity],
            &self.rank_lo,
            &self.rank_hi,
            &mut self.counters,
        );
        unsafe { self.enumerate.clone().launch(cfg, params).map_err(cuda_err)? };
        dev.synchronize().map_err(cuda_err)?;
        let counters = dev.dtoh_sync_copy(&self.counters).map_err(cuda_err)?;
        Ok(LaunchStats::from_counters(&counters, start.elapsed().as_secs_f64()))
    }

    /// The join of a batch: one grid column per item, enough chunks for the
    /// longest table. Returns the per-item counts and the time taken.
    fn join(&mut self, batch: &Batch) -> Result<(Vec<u64>, f64), String> {
        let start = Instant::now();
        let dev = &self.dev;
        let n_items = batch.items.len();
        let table_offset = dev.htod_sync_copy(&batch.launches[0].table_offset).map_err(cuda_err)?;
        let entries = dev.htod_sync_copy(&batch.entries).map_err(cuda_err)?;
        let mut sums: CudaSlice<u64> = dev.alloc_zeros(n_items).map_err(cuda_err)?;
        let chunks = batch.entries.iter().max().copied().unwrap_or(0).div_ceil(JOIN_CHUNK).max(1);
        let cfg = LaunchConfig { grid_dim: (n_items as u32, chunks, 1), block_dim: (JOIN_THREADS, 1, 1), shared_mem_bytes: 0 };
        let [e_tables, o_tables] = &self.tables;
        unsafe { self.join.clone().launch(cfg, (&table_offset, &entries, e_tables, o_tables, &mut sums)).map_err(cuda_err)? };
        dev.synchronize().map_err(cuda_err)?;
        let counts = dev.dtoh_sync_copy(&sums).map_err(cuda_err)?;
        Ok((counts, start.elapsed().as_secs_f64()))
    }
}

/// What one enumerate launch (or all launches of one colour) did, and how
/// long it took.
#[derive(Clone, Copy, Default)]
pub struct LaunchStats {
    pub seconds: f64,
    /// Tickets taken (equals the launch's total).
    pub tickets: u64,
    /// Tickets whose prefix had a conflict.
    pub dead: u64,
    /// Search steps below the prefix (loop iterations that were not ticket setup).
    pub steps: u64,
    /// Complete placements found.
    pub leaves: u64,
}

impl LaunchStats {
    fn from_counters(c: &[u64], seconds: f64) -> Self {
        LaunchStats { seconds, tickets: c[COUNTER_TICKETS], dead: c[COUNTER_DEAD], steps: c[COUNTER_STEPS], leaves: c[COUNTER_LEAVES] }
    }

    pub fn add(&mut self, o: &LaunchStats) {
        self.seconds += o.seconds;
        self.tickets += o.tickets;
        self.dead += o.dead;
        self.steps += o.steps;
        self.leaves += o.leaves;
    }
}

impl std::fmt::Display for LaunchStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dead_pct = if self.tickets > 0 { 100.0 * self.dead as f64 / self.tickets as f64 } else { 0.0 };
        write!(
            f,
            "{:.3}s, {} tickets ({dead_pct:.0}% dead), {:.3e} steps, {:.3e} leaves ({:.2e} steps/s, {:.2e} leaves/s)",
            self.seconds,
            self.tickets,
            self.steps as f64,
            self.leaves as f64,
            self.steps as f64 / self.seconds,
            self.leaves as f64 / self.seconds
        )
    }
}

//! CUDA path (feature `cuda`): the same computation as `count_queens`, with
//! both one-colour searches of every row set run on the GPU.
//!
//! The work is the list of feasible even-queen row sets R from
//! `get_feasible_even_row_combos`, each with its symmetry weight, exactly as
//! on the CPU. For each R we need
//!
//!   1. the table E[R, C] for all C            (`kernel/enumerate.cu`, parity 0),
//!   2. the table O[R̅, C̅] for all C            (`kernel/enumerate.cu`, parity 1),
//!   3. sum over C of E[R, C] * O[R̅, C̅]        (`kernel/join.cu`).
//!
//! Steps 1 and 2 are the same search (a transcription of `enumerate`) on the
//! two colours, each tallying its leaves into a table; step 2 indexes its
//! table by the complement of the leaf's columns, so the
//! two tables are aligned and step 3 is a plain dot product. The CPU path
//! (`get_erc` then `backtrack_odd_solutions`) instead looks E[R, C] up at each
//! odd-square leaf; on the GPU those random reads are the bottleneck, while
//! tallying is a fire-and-forget atomic, so both sides tabulate.
//!
//! Because one row set is far too little work to fill a GPU, row sets are
//! processed in batches: a batch takes as many items as fit a table memory
//! budget, all their E tables live side by side in one device buffer and
//! their O tables at the same offsets in another, and three kernel launches
//! per batch do the steps for every item. Within an enumerate launch, threads
//! are assigned by "tickets" (fixed column choices for the first few rows;
//! see the kernel header), so one launch has millions of independent threads.
//!
//! The kernel source is compiled at run time by NVRTC for whatever GPU is
//! present, with the board size and prefix depth baked in as macros.

use crate::rowset_log::{RowsetCount, RowsetLog};
use crate::solution_table::{RANK_HI, RANK_LO, SolutionTable};
use crate::{complement, get_feasible_even_row_combos, RowCombo};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;
use std::time::Instant;

const KERNEL_SRC: &str = include_str!("../kernel/enumerate.cu");
const KERNEL_NAME: &str = "enumerate_kernel";
const MODULE_NAME: &str = "pnqueens";
const JOIN_SRC: &str = include_str!("../kernel/join.cu");
const JOIN_NAME: &str = "join_kernel";
const JOIN_MODULE: &str = "pnqueens_join";
/// Threads per block and table entries per block of the join kernel (its
/// THREADS and CHUNK).
const JOIN_THREADS: u32 = 256;
const JOIN_CHUNK: u32 = JOIN_THREADS * 64;
/// Static shared memory available to one block on every supported GPU.
const SHARED_MEMORY_LIMIT: u32 = 48 * 1024;
/// Bytes of shared memory per stack frame per thread (four u32 words).
const FRAME_BYTES: u32 = 16;

/// Threads per block and stack frames per thread for the kernel. A row set
/// has at most n rows, and the DFS below the prefix needs one frame per row
/// after it, so n - prefix frames always suffice. Blocks are 128 threads
/// when their stacks fit the static shared-memory limit, else 64.
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
    use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
    let sms = dev.attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).map_err(cuda_err)? as u32;
    let per_sm = kernel.occupancy_max_active_blocks_per_multiprocessor(block_size, 0, None).map_err(cuda_err)?;
    if per_sm == 0 {
        return Err("the kernel cannot be launched with this block size".into());
    }
    Ok((sms * per_sm, per_sm))
}

pub struct GpuOptions {
    /// CUDA device index.
    pub device: usize,
    /// Rows of each row set whose columns are fixed by the thread's ticket.
    /// Deeper prefixes mean more, shorter tickets per item. None: choose by
    /// `default_prefix_depth`.
    pub prefix_depth: Option<u32>,
    /// Device memory for one batch's tables (both colours together), in bytes.
    pub table_budget_bytes: usize,
}

/// The smallest prefix depth worth using.
const MIN_PREFIX_DEPTH: u32 = 3;
/// Tickets in progress at a moment, as a multiple of the resident threads.
/// Threads take tickets in order, but a slow ticket keeps its thread on an
/// old item while the others move on, so the window is wider than one per
/// thread.
const TICKET_WINDOW_FACTOR: u64 = 2;

/// The default prefix depth for this board and GPU: the smallest depth at
/// which the tables of the items in flight fit the L2 cache.
///
/// About `threads` tickets are in progress at any moment (times the window
/// factor), consecutive ones, so the items in flight number about window /
/// tickets-per-item, and the kernel's table updates are spread over their
/// tables. While those fit L2 the updates are cheap; once they spill,
/// every update is a DRAM access and the kernel runs at half speed or worse
/// (measured at n = 23: 235 s with depth 3, 109 s with depth 4). Deeper
/// prefixes cost setup work for their many more, mostly dead, tickets, so
/// the depth is no larger than this needs. The heaviest item stands for all.
fn default_prefix_depth(n: u32, heaviest: &RowCombo, threads: u32, l2_bytes: u64) -> u32 {
    let table_bytes = SolutionTable::dimensions(n, heaviest.mask, 0).0 as u64 * std::mem::size_of::<u32>() as u64;
    let window = TICKET_WINDOW_FACTOR * u64::from(threads);
    let rows_bar = complement(n, heaviest.mask);
    let mut depth = MIN_PREFIX_DEPTH;
    loop {
        let tickets = ticket_count(n, heaviest.mask, 0, depth).min(ticket_count(n, rows_bar, 1, depth));
        let items_in_flight = window.div_ceil(tickets.max(1));
        let next = ticket_count(n, heaviest.mask, 0, depth + 1).max(ticket_count(n, rows_bar, 1, depth + 1));
        let deeper_is_allowed = depth < n && next <= u64::from(u32::MAX);
        // Stop when the tables fit, or when going deeper cannot help: one
        // item in flight is the least possible.
        if items_in_flight * table_bytes <= l2_bytes * 3 / 4 || items_in_flight <= 1 || !deeper_is_allowed {
            return depth;
        }
        depth += 1;
    }
}

fn cuda_err(e: impl std::fmt::Display) -> String {
    format!("CUDA error: {e}")
}

/// Whether the kernel applies the mirror symmetry of `backtrack_odd_solutions`
/// to the odd-square search: odd n only (n = 1 has no odd-square rows to restrict).
fn uses_mirror(n: u32) -> bool {
    n % 2 == 1 && n >= 3
}

/// Number of tickets for one search: the product, over the first
/// `prefix_depth` rows of `rows`, of the number of columns the ticket digit
/// for that row ranges over. That is the colour-compatible columns of the
/// row, except that with the mirror symmetry the first row of an odd-square
/// search is limited to the columns up to the centre. This must match the
/// kernel's prefix decoding exactly.
fn ticket_count(n: u32, rows: u32, parity: u32, prefix_depth: u32) -> u64 {
    let (m_even, m_odd) = (n.div_ceil(2), n / 2);
    let mid = (n - 1) / 2;
    let mut count = 1u64;
    let mut m = rows;
    let mut taken = 0;
    while m != 0 && taken < prefix_depth {
        let r = m.trailing_zeros();
        m &= m - 1;
        let col_parity = (r & 1) ^ parity;
        let radix = if taken == 0 && parity == 1 && uses_mirror(n) {
            (mid - col_parity) / 2 + 1
        } else if col_parity == 0 {
            m_even
        } else {
            m_odd
        };
        count *= u64::from(radix);
        taken += 1;
    }
    count
}

/// The per-item arrays of one enumerate launch, in the layout the kernel reads.
struct Launch {
    rows: Vec<u32>,
    table_offset: Vec<u64>,
    odd_stride: Vec<u32>,
    /// `ticket_base[i]` is the first ticket of item i; the last entry is the total.
    ticket_base: Vec<u64>,
}

/// The same arrays on the device.
struct LaunchDev {
    rows: CudaSlice<u32>,
    table_offset: CudaSlice<u64>,
    odd_stride: CudaSlice<u32>,
    ticket_base: CudaSlice<u64>,
}

impl Launch {
    fn new() -> Self {
        Launch { rows: Vec::new(), table_offset: Vec::new(), odd_stride: Vec::new(), ticket_base: vec![0] }
    }

    fn push(&mut self, rows: u32, table_offset: u64, odd_stride: u32, tickets: u64) {
        self.rows.push(rows);
        self.table_offset.push(table_offset);
        self.odd_stride.push(odd_stride);
        let next = self.ticket_base.last().unwrap() + tickets;
        self.ticket_base.push(next);
    }

    fn upload(&self, dev: &Arc<CudaDevice>) -> Result<LaunchDev, String> {
        Ok(LaunchDev {
            rows: dev.htod_sync_copy(&self.rows).map_err(cuda_err)?,
            table_offset: dev.htod_sync_copy(&self.table_offset).map_err(cuda_err)?,
            odd_stride: dev.htod_sync_copy(&self.odd_stride).map_err(cuda_err)?,
            ticket_base: dev.htod_sync_copy(&self.ticket_base).map_err(cuda_err)?,
        })
    }
}

/// Q(n) computed on the GPU. Every row set's count is recorded in `log`
/// (which keeps the weighted total) and the log is flushed after each batch.
/// Prints progress per batch.
pub fn count_queens_gpu(n: u32, opts: &GpuOptions, log: &mut RowsetLog) -> Result<(), String> {
    let start = Instant::now();

    // ---- work items, heaviest tables first so each batch holds items of similar size
    let mut items: Vec<RowCombo> = get_feasible_even_row_combos(n);
    let table_entries = |rows: u32| SolutionTable::dimensions(n, rows, 0).0;
    items.sort_by_key(|it| std::cmp::Reverse(table_entries(it.mask)));
    println!("PREPARED {}", items.len());

    let entry_bytes = std::mem::size_of::<u32>();
    // Half the budget for each colour's tables.
    let budget_entries = opts.table_budget_bytes / entry_bytes / 2;
    if let Some(biggest) = items.first() {
        let need = table_entries(biggest.mask);
        if need > budget_entries {
            return Err(format!(
                "table budget of {} MiB is too small for one pair of tables of {need} entries",
                opts.table_budget_bytes >> 20
            ));
        }
    }

    // ---- device setup: choose the prefix depth, compile the kernel for this n, upload the rank tables
    let dev = CudaDevice::new(opts.device).map_err(cuda_err)?;
    println!("device: CUDA {} ({})", opts.device, dev.name().map_err(cuda_err)?);
    let prefix_depth = match (opts.prefix_depth, items.first()) {
        (Some(depth), _) => depth,
        (None, None) => MIN_PREFIX_DEPTH,
        (None, Some(heaviest)) => {
            use cudarc::driver::sys::CUdevice_attribute::{CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT};
            let sms = dev.attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).map_err(cuda_err)? as u32;
            let l2_bytes = dev.attribute(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE).map_err(cuda_err)? as u64;
            // The residency this kernel typically gets: eight warps per multiprocessor.
            default_prefix_depth(n, heaviest, sms * 256, l2_bytes)
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
    let (block_size, frames) = block_shape(n, prefix_depth);
    let src = format!(
        "#define NQ_N {n}u\n#define NQ_PREFIX {prefix_depth}u\n#define NQ_BLOCK {block_size}u\n#define NQ_FRAMES {frames}u\n#define NQ_MIRROR {}\n{KERNEL_SRC}",
        u32::from(uses_mirror(n))
    );
    let ptx = compile_ptx_with_opts(src, CompileOptions::default())
        .map_err(|e| format!("kernel compilation failed:\n{e}"))?;
    dev.load_ptx(ptx, MODULE_NAME, &[KERNEL_NAME]).map_err(cuda_err)?;
    let kernel = dev.get_func(MODULE_NAME, KERNEL_NAME).ok_or("kernel not found after loading")?;
    let (grid_size, blocks_per_sm) = grid_size(&dev, &kernel, block_size)?;
    println!(
        "gpu: prefix depth {prefix_depth}, table budget {} MiB, {grid_size} blocks of {block_size} threads ({blocks_per_sm} per SM), {frames} stack frames ({} KiB shared memory per block)",
        opts.table_budget_bytes >> 20,
        block_size * frames * FRAME_BYTES / 1024
    );
    let join_ptx = compile_ptx_with_opts(JOIN_SRC, CompileOptions::default())
        .map_err(|e| format!("join kernel compilation failed:\n{e}"))?;
    dev.load_ptx(join_ptx, JOIN_MODULE, &[JOIN_NAME]).map_err(cuda_err)?;
    let join = dev.get_func(JOIN_MODULE, JOIN_NAME).ok_or("join kernel not found after loading")?;

    let rank_lo_d: CudaSlice<u16> = dev.htod_sync_copy(&RANK_LO).map_err(cuda_err)?;
    let rank_hi_flat: Vec<u16> = RANK_HI.iter().flatten().copied().collect();
    let rank_hi_d: CudaSlice<u16> = dev.htod_sync_copy(&rank_hi_flat).map_err(cuda_err)?;
    // One buffer of tables per colour: parity 0 fills the first, parity 1 the second.
    let mut tables_d: [CudaSlice<u32>; 2] =
        [dev.alloc_zeros(budget_entries).map_err(cuda_err)?, dev.alloc_zeros(budget_entries).map_err(cuda_err)?];
    // Kernel counters: [0] next ticket, then the statistics (see COUNTER_*).
    let mut counters_d: CudaSlice<u64> = dev.alloc_zeros(N_COUNTERS).map_err(cuda_err)?;
    let mut stats = [LaunchStats::default(), LaunchStats::default()]; // per colour, whole run
    let mut join_seconds = 0.0;

    // ---- batches
    let mut next_item = 0;
    let mut batch_number = 0;
    while next_item < items.len() {
        // Take items while their tables fit the budget.
        let mut launches = [Launch::new(), Launch::new()]; // parity 0: rows R; parity 1: rows R̅
        let mut item_entries: Vec<u32> = Vec::new(); // table length of each item, for the join
        let mut used_entries: usize = 0;
        let first_item = next_item;
        while next_item < items.len() {
            let item = &items[next_item];
            let (entries, odd_stride) = SolutionTable::dimensions(n, item.mask, 0);
            if used_entries + entries > budget_entries {
                break;
            }
            let offset = used_entries as u64;
            let rows_bar = complement(n, item.mask);
            launches[0].push(item.mask, offset, odd_stride, ticket_count(n, item.mask, 0, prefix_depth));
            launches[1].push(rows_bar, offset, odd_stride, ticket_count(n, rows_bar, 1, prefix_depth));
            item_entries.push(entries as u32);
            used_entries += entries;
            next_item += 1;
        }
        let batch_items = &items[first_item..next_item];
        let n_items = batch_items.len();
        batch_number += 1;

        // Zero this batch's tables and sums.
        for tables in tables_d.iter_mut() {
            let mut used = tables.slice_mut(0..used_entries.max(1));
            dev.memset_zeros(&mut used).map_err(cuda_err)?;
        }
        let mut sums_d: CudaSlice<u64> = dev.alloc_zeros(n_items).map_err(cuda_err)?;

        // All launches go to the same stream, so the join sees the completed
        // tables. Each launch is timed and its kernel counters collected.
        let mut batch_stats = [LaunchStats::default(), LaunchStats::default()];
        for parity in 0..2usize {
            let launch = launches[parity].upload(&dev)?;
            let cfg = LaunchConfig { grid_dim: (grid_size, 1, 1), block_dim: (block_size, 1, 1), shared_mem_bytes: 0 };
            dev.memset_zeros(&mut counters_d).map_err(cuda_err)?;
            dev.synchronize().map_err(cuda_err)?;
            let launch_start = Instant::now();
            unsafe {
                kernel
                    .clone()
                    .launch(
                        cfg,
                        (
                            parity as u32,
                            n_items as u32,
                            &launch.rows,
                            &launch.table_offset,
                            &launch.odd_stride,
                            &launch.ticket_base,
                            &mut tables_d[parity],
                            &rank_lo_d,
                            &rank_hi_d,
                            &mut counters_d,
                        ),
                    )
                    .map_err(cuda_err)?;
            }
            dev.synchronize().map_err(cuda_err)?;
            let counters: Vec<u64> = dev.dtoh_sync_copy(&counters_d).map_err(cuda_err)?;
            let s = LaunchStats::from_counters(&counters, launch_start.elapsed().as_secs_f64());
            batch_stats[parity] = s;
            stats[parity].add(&s);
        }

        // The join: one grid column per item, enough chunks for the longest table.
        let join_start = Instant::now();
        {
            let offset_d = dev.htod_sync_copy(&launches[0].table_offset).map_err(cuda_err)?;
            let entries_d = dev.htod_sync_copy(&item_entries).map_err(cuda_err)?;
            let chunks = item_entries.iter().max().copied().unwrap_or(0).div_ceil(JOIN_CHUNK).max(1);
            let cfg = LaunchConfig { grid_dim: (n_items as u32, chunks, 1), block_dim: (JOIN_THREADS, 1, 1), shared_mem_bytes: 0 };
            let [e_tables, o_tables] = &tables_d;
            unsafe {
                join.clone()
                    .launch(cfg, (&offset_d, &entries_d, e_tables, o_tables, &mut sums_d))
                    .map_err(cuda_err)?;
            }
            dev.synchronize().map_err(cuda_err)?;
        }
        let batch_join_seconds = join_start.elapsed().as_secs_f64();
        join_seconds += batch_join_seconds;

        // Record this batch's row sets; the flush makes them durable before
        // the next batch starts.
        let sums: Vec<u64> = dev.dtoh_sync_copy(&sums_d).map_err(cuda_err)?;
        for (item, &count) in batch_items.iter().zip(&sums) {
            let r = RowsetCount { mask: item.mask, weight: item.weight as u32, count };
            log.record(&r).map_err(|e| format!("cannot write row-set counts: {e}"))?;
        }
        log.flush().map_err(|e| format!("cannot write row-set counts: {e}"))?;
        println!(
            "BATCH {batch_number}: items {first_item}..{next_item} of {}, 2 x {} MiB of tables; even {}; odd {}; join {batch_join_seconds:.3}s; {:.1}s elapsed",
            items.len(),
            (used_entries * entry_bytes) >> 20,
            batch_stats[0],
            batch_stats[1],
            start.elapsed().as_secs_f64()
        );
    }
    println!("even total: {}", stats[0]);
    println!("odd total:  {}", stats[1]);
    println!("join total: {join_seconds:.3}s");
    Ok(())
}

/// Layout of the kernel's counter buffer. Index 0 is the ticket counter the
/// kernel dispenses work from; the rest are statistics it adds up at the end.
const COUNTER_TICKETS: usize = 1;
const COUNTER_DEAD: usize = 2;
const COUNTER_STEPS: usize = 3;
const COUNTER_LEAVES: usize = 4;
const N_COUNTERS: usize = 5;

/// What one launch (or all launches of one mode) did, and how long it took.
#[derive(Clone, Copy, Default)]
struct LaunchStats {
    seconds: f64,
    /// Tickets taken (equals the launch's total).
    tickets: u64,
    /// Tickets whose prefix had a conflict.
    dead: u64,
    /// Search steps below the prefix (loop iterations that were not ticket setup).
    steps: u64,
    /// Complete placements found.
    leaves: u64,
}

impl LaunchStats {
    fn from_counters(c: &[u64], seconds: f64) -> Self {
        LaunchStats {
            seconds,
            tickets: c[COUNTER_TICKETS],
            dead: c[COUNTER_DEAD],
            steps: c[COUNTER_STEPS],
            leaves: c[COUNTER_LEAVES],
        }
    }

    fn add(&mut self, o: &LaunchStats) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticket_count_is_the_product_of_prefix_radices() {
        // n = 7: 4 even columns, 3 odd. R = {0, 3, 4}; even squares.
        // row 0 (even) -> even columns (4); row 3 (odd) -> odd columns (3);
        // row 4 (even) -> even columns (4).
        assert_eq!(ticket_count(7, 0b11001, 0, 1), 4);
        assert_eq!(ticket_count(7, 0b11001, 0, 2), 12);
        assert_eq!(ticket_count(7, 0b11001, 0, 3), 48);
        assert_eq!(ticket_count(7, 0b11001, 0, 10), 48, "prefix deeper than |R| stops at |R|");
        // Odd squares swap the column parities, and for odd n the first row
        // of the odd-square search is limited to the columns up to the centre
        // (n = 7: centre column 3; row 0 uses odd columns, so {1, 3}).
        assert_eq!(ticket_count(7, 0b11001, 1, 3), 2 * 4 * 3);
        // n = 9, centre 4: row 0 on odd squares uses odd columns {1, 3};
        // row 1 uses even columns {0, 2, 4}.
        assert_eq!(ticket_count(9, 0b01, 1, 1), 2);
        assert_eq!(ticket_count(9, 0b10, 1, 1), 3);
        // Even n: no mirror restriction on the odd-square search.
        assert_eq!(ticket_count(8, 0b11001, 1, 3), 4 * 4 * 4);
        // No rows: a single ticket.
        assert_eq!(ticket_count(7, 0, 0, 3), 1);
    }
}

//! CUDA path (feature `cuda`): the same computation as `count_queens`, with
//! both one-colour searches of every row set run on the GPU.
//!
//! The work is the list of feasible even-queen row sets R from
//! `get_feasible_even_row_combos`, each with its symmetry weight, exactly as
//! on the CPU. For each R we need
//!
//!   1. the table E[R, C] for all C            (kernel mode 0, "tabulate"),
//!   2. sum over C of E[R, C] * O[R̅, C̅]        (kernel mode 1, "read"),
//!
//! where step 2 enumerates the odd-square placements on R̅ and, at each leaf
//! with column set C̅, reads E[R, C] from the table built in step 1. That is
//! precisely `get_erc` followed by `backtrack_odd_solutions`, and the kernel
//! (`kernel/enumerate.cu`) is a transcription of `enumerate`.
//!
//! Because one row set is far too little work to fill a GPU, row sets are
//! processed in batches: a batch takes as many items as fit a table memory
//! budget, all their E tables live side by side in one device buffer, and two
//! kernel launches per batch do step 1 for every item and then step 2 for
//! every item. Within a launch, threads are assigned by "tickets" (fixed
//! column choices for the first few rows; see the kernel header), so one
//! launch has millions of independent threads.
//!
//! The kernel source is compiled at run time by NVRTC for whatever GPU is
//! present, with the board size and prefix depth baked in as macros.

use crate::rowset_log::{RowsetCount, RowsetLog};
use crate::solution_table::{RANK_HI, RANK_LO, SolutionTable};
use crate::{complement, get_feasible_even_row_combos, RowCombo};
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::time::Instant;

const KERNEL_SRC: &str = include_str!("../kernel/enumerate.cu");
const KERNEL_NAME: &str = "enumerate_kernel";
const MODULE_NAME: &str = "pnqueens";
/// Static shared memory available to one block on every supported GPU.
const SHARED_MEMORY_LIMIT: u32 = 48 * 1024;
/// Bytes of shared memory per stack frame per thread (five u32 words).
const FRAME_BYTES: u32 = 20;

/// Threads per block and stack frames per thread for the kernel. A row set
/// has at most n rows, and the DFS below the prefix needs one frame per row
/// after it, so n - prefix frames always suffice. Blocks are 128 threads
/// when their stacks fit the static shared-memory limit, else 64.
fn block_shape(n: u32, prefix_depth: u32) -> (u32, u32) {
    let frames = n.saturating_sub(prefix_depth).max(1);
    let block = if 128 * frames * FRAME_BYTES <= SHARED_MEMORY_LIMIT { 128 } else { 64 };
    (block, frames)
}

/// Number of blocks to launch: the kernel's threads are persistent (each
/// loops over tickets), so we launch just enough blocks to fill every
/// multiprocessor, as limited by its shared memory and thread capacity.
fn grid_size(dev: &CudaDevice, block_size: u32, frames: u32) -> Result<u32, String> {
    use cudarc::driver::sys::CUdevice_attribute::*;
    let attr = |a| dev.attribute(a).map_err(cuda_err).map(|v| v as u32);
    let sms = attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
    let shared_per_sm = attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)?;
    let threads_per_sm = attr(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)?;
    let by_shared = shared_per_sm / (block_size * frames * FRAME_BYTES);
    let by_threads = threads_per_sm / block_size;
    Ok(sms * by_shared.min(by_threads).max(1))
}

pub struct GpuOptions {
    /// CUDA device index.
    pub device: usize,
    /// Rows of each row set whose columns are fixed by the thread's ticket.
    /// Deeper prefixes mean more, shorter threads per item.
    pub prefix_depth: u32,
    /// Device memory for one batch's tables, in bytes.
    pub table_budget_bytes: usize,
}

impl GpuOptions {
    /// The prefix depth that gives a few hundred thousand threads per batch at
    /// typical batch sizes: 3 rows, one more from n = 24, one more from n = 28.
    pub fn default_prefix_depth(n: u32) -> u32 {
        3 + u32::from(n >= 24) + u32::from(n >= 28)
    }
}

fn cuda_err(e: impl std::fmt::Display) -> String {
    format!("CUDA error: {e}")
}

/// Number of tickets for one search: the product, over the first
/// `prefix_depth` rows of `rows`, of the number of colour-compatible columns
/// in that row. This must match the kernel's prefix decoding exactly.
fn ticket_count(n: u32, rows: u32, parity: u32, prefix_depth: u32) -> u64 {
    let (m_even, m_odd) = (u64::from(n.div_ceil(2)), u64::from(n / 2));
    let mut count = 1u64;
    let mut m = rows;
    let mut taken = 0;
    while m != 0 && taken < prefix_depth {
        let r = m.trailing_zeros();
        m &= m - 1;
        let col_parity = (r & 1) ^ parity;
        count *= if col_parity == 0 { m_even } else { m_odd };
        taken += 1;
    }
    count
}

/// The per-item arrays of one kernel launch, in the layout the kernel reads.
struct Launch {
    rows: Vec<u32>,
    table_offset: Vec<u64>,
    odd_stride: Vec<u32>,
    /// `ticket_base[i]` is the first ticket of item i; the last entry is the total.
    ticket_base: Vec<u64>,
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
}

/// Q(n) computed on the GPU. Every row set's count is recorded in `log`
/// (which keeps the weighted total) and the log is flushed after each batch.
/// Prints progress per batch.
pub fn count_queens_gpu(n: u32, opts: &GpuOptions, log: &mut RowsetLog) -> Result<(), String> {
    let start = Instant::now();
    if opts.prefix_depth == 0 {
        return Err("prefix depth must be at least 1".into());
    }
    // The kernel decodes a ticket's position within its item in 32 bits.
    let max_tickets = u64::from(n.div_ceil(2)).pow(opts.prefix_depth);
    if max_tickets > u64::from(u32::MAX) {
        return Err(format!("prefix depth {} gives up to {max_tickets} tickets per row set; the limit is 2^32", opts.prefix_depth));
    }

    // ---- work items, heaviest tables first so each batch holds items of similar size
    let mut items: Vec<RowCombo> = get_feasible_even_row_combos(n);
    let table_entries = |rows: u32| SolutionTable::dimensions(n, rows, 0).0;
    items.sort_by_key(|it| std::cmp::Reverse(table_entries(it.mask)));
    println!("PREPARED {}", items.len());

    let entry_bytes = std::mem::size_of::<u32>();
    let budget_entries = opts.table_budget_bytes / entry_bytes;
    if let Some(biggest) = items.first() {
        let need = table_entries(biggest.mask);
        if need > budget_entries {
            return Err(format!(
                "table budget of {} MiB is too small for one table of {need} entries",
                opts.table_budget_bytes >> 20
            ));
        }
    }

    // ---- device setup: compile the kernel for this n, upload the rank tables
    let dev = CudaDevice::new(opts.device).map_err(cuda_err)?;
    println!("device: CUDA {} ({})", opts.device, dev.name().map_err(cuda_err)?);
    let (block_size, frames) = block_shape(n, opts.prefix_depth);
    let grid_size = grid_size(&dev, block_size, frames)?;
    println!(
        "gpu: prefix depth {}, table budget {} MiB, {grid_size} blocks of {block_size} threads, {frames} stack frames ({} KiB shared memory per block)",
        opts.prefix_depth,
        opts.table_budget_bytes >> 20,
        block_size * frames * FRAME_BYTES / 1024
    );

    let src = format!(
        "#define NQ_N {n}u\n#define NQ_PREFIX {}u\n#define NQ_BLOCK {block_size}u\n#define NQ_FRAMES {frames}u\n{KERNEL_SRC}",
        opts.prefix_depth
    );
    let ptx = compile_ptx_with_opts(src, CompileOptions::default())
        .map_err(|e| format!("kernel compilation failed:\n{e}"))?;
    dev.load_ptx(ptx, MODULE_NAME, &[KERNEL_NAME]).map_err(cuda_err)?;
    let kernel = dev.get_func(MODULE_NAME, KERNEL_NAME).ok_or("kernel not found after loading")?;

    let rank_lo_d: CudaSlice<u16> = dev.htod_sync_copy(&RANK_LO).map_err(cuda_err)?;
    let rank_hi_flat: Vec<u16> = RANK_HI.iter().flatten().copied().collect();
    let rank_hi_d: CudaSlice<u16> = dev.htod_sync_copy(&rank_hi_flat).map_err(cuda_err)?;
    let mut tables_d: CudaSlice<u32> = dev.alloc_zeros(budget_entries).map_err(cuda_err)?;
    let mut next_ticket_d: CudaSlice<u64> = dev.alloc_zeros(1).map_err(cuda_err)?;

    // ---- batches
    let mut next_item = 0;
    let mut batch_number = 0;
    while next_item < items.len() {
        // Take items while their tables fit the budget.
        let mut even = Launch::new(); // mode 0: E[R, ·] on even squares, rows R
        let mut odd = Launch::new(); // mode 1: odd squares on R̅, reading E[R, ·]
        let mut used_entries: usize = 0;
        let first_item = next_item;
        while next_item < items.len() {
            let item = &items[next_item];
            let (entries, odd_stride) = SolutionTable::dimensions(n, item.mask, 0);
            if used_entries + entries > budget_entries {
                break;
            }
            let offset = used_entries as u64;
            even.push(item.mask, offset, odd_stride, ticket_count(n, item.mask, 0, opts.prefix_depth));
            let rows_bar = complement(n, item.mask);
            odd.push(rows_bar, offset, odd_stride, ticket_count(n, rows_bar, 1, opts.prefix_depth));
            used_entries += entries;
            next_item += 1;
        }
        let batch_items = &items[first_item..next_item];
        let n_items = batch_items.len();
        batch_number += 1;

        // Zero this batch's tables and sums.
        {
            let mut used = tables_d.slice_mut(0..used_entries.max(1));
            dev.memset_zeros(&mut used).map_err(cuda_err)?;
        }
        let mut sums_d: CudaSlice<u64> = dev.alloc_zeros(n_items).map_err(cuda_err)?;

        // Both launches go to the same stream, so the read launch sees the
        // completed tables of the tabulate launch.
        for (mode, launch) in [(0u32, &even), (1u32, &odd)] {
            let rows_d = dev.htod_sync_copy(&launch.rows).map_err(cuda_err)?;
            let offset_d = dev.htod_sync_copy(&launch.table_offset).map_err(cuda_err)?;
            let stride_d = dev.htod_sync_copy(&launch.odd_stride).map_err(cuda_err)?;
            let base_d = dev.htod_sync_copy(&launch.ticket_base).map_err(cuda_err)?;
            let cfg = LaunchConfig { grid_dim: (grid_size, 1, 1), block_dim: (block_size, 1, 1), shared_mem_bytes: 0 };
            let parity = mode; // even squares for the tabulate launch, odd for the read launch
            dev.memset_zeros(&mut next_ticket_d).map_err(cuda_err)?;
            unsafe {
                kernel
                    .clone()
                    .launch(
                        cfg,
                        (
                            mode,
                            parity,
                            n_items as u32,
                            &rows_d,
                            &offset_d,
                            &stride_d,
                            &base_d,
                            &mut tables_d,
                            &mut sums_d,
                            &rank_lo_d,
                            &rank_hi_d,
                            &mut next_ticket_d,
                        ),
                    )
                    .map_err(cuda_err)?;
            }
        }
        dev.synchronize().map_err(cuda_err)?;

        // Record this batch's row sets; the flush makes them durable before
        // the next batch starts.
        let sums: Vec<u64> = dev.dtoh_sync_copy(&sums_d).map_err(cuda_err)?;
        for (item, &count) in batch_items.iter().zip(&sums) {
            let r = RowsetCount { mask: item.mask, weight: item.weight as u32, count };
            log.record(&r).map_err(|e| format!("cannot write row-set counts: {e}"))?;
        }
        log.flush().map_err(|e| format!("cannot write row-set counts: {e}"))?;
        println!(
            "BATCH {batch_number}: items {first_item}..{next_item} of {}, {} MiB of tables, {:.1}s elapsed",
            items.len(),
            (used_entries * entry_bytes) >> 20,
            start.elapsed().as_secs_f64()
        );
    }
    Ok(())
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
        // Odd squares swap the column parities.
        assert_eq!(ticket_count(7, 0b11001, 1, 3), 3 * 4 * 3);
        // No rows: a single ticket.
        assert_eq!(ticket_count(7, 0, 0, 3), 1);
    }
}

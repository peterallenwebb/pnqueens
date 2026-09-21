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
//! table by the complement of the leaf's columns, so the two tables are
//! aligned and step 3 is a plain dot product. The CPU path (`get_erc` then
//! `backtrack_odd_solutions`) instead looks E[R, C] up at each odd-square
//! leaf; on the GPU those random reads are the bottleneck, while tallying is
//! a fire-and-forget atomic, so both sides tabulate.
//!
//! Because one row set is far too little work to fill a GPU, row sets are
//! processed in batches: a batch takes as many items as fit a table memory
//! budget, all their E tables live side by side in one device buffer and
//! their O tables at the same offsets in another, and three kernel launches
//! per batch do the steps for every item. Within an enumerate launch, work
//! is assigned by "tickets" (fixed column choices for the first few rows;
//! see the kernel header), so one launch has millions of independent units.
//!
//! The module is split by what needs a device:
//!
//! - `plan`: the work plan (items, tables, tickets, batches), plain Rust;
//! - `device`: the CUDA driver, which compiles the kernels with NVRTC for
//!   whatever GPU is present and runs the plan one batch at a time;
//! - `emulate`: a test that runs the kernel source as host C++ against the
//!   plan, so kernel changes can be checked without a GPU
//!   (`cargo test --features emulate`).

pub mod plan;
#[cfg(feature = "cuda")]
mod device;
#[cfg(all(test, feature = "emulate"))]
mod emulate;

#[cfg(feature = "cuda")]
pub struct GpuOptions {
    /// CUDA device index.
    pub device: usize,
    /// Rows of each row set whose columns are fixed by the thread's ticket.
    /// Deeper prefixes mean more, shorter tickets per item. None: choose by
    /// `plan::default_prefix_depth`.
    pub prefix_depth: Option<u32>,
    /// Device memory for one batch's tables (both colours together), in bytes.
    pub table_budget_bytes: usize,
    /// Blocks of the enumerate kernel to keep resident per multiprocessor;
    /// None means as many as fit. Fewer is only useful for measurements.
    pub blocks_per_sm: Option<u32>,
}

/// What a GPU run has to say about itself: the device and the statistics of
/// its launches, for the manifest.
#[cfg(feature = "cuda")]
pub struct RunStats {
    pub gpu: String,
    /// The configuration the run actually used (the options may have left
    /// them to the defaults): a verification rerun must differ in these.
    pub prefix_depth: u32,
    pub blocks_per_sm: u32,
    pub even: device::LaunchStats,
    pub odd: device::LaunchStats,
    pub join_seconds: f64,
    pub max_entry: u32,
}

#[cfg(feature = "cuda")]
impl RunStats {
    pub fn to_json(&self) -> String {
        let launch = |s: &device::LaunchStats| {
            format!(
                "{{\"seconds\": {:.3}, \"tickets\": {}, \"dead\": {}, \"steps\": {}, \"leaves\": {}, \"tallied\": {}}}",
                s.seconds, s.tickets, s.dead, s.steps, s.leaves, s.tallied
            )
        };
        format!(
            "{{\"prefix_depth\": {}, \"blocks_per_sm\": {}, \"even\": {}, \"odd\": {}, \"join_seconds\": {:.3}, \"max_entry\": {}}}",
            self.prefix_depth,
            self.blocks_per_sm,
            launch(&self.even),
            launch(&self.odd),
            self.join_seconds,
            self.max_entry
        )
    }
}

/// The given row sets counted on the GPU. Every row set's count is recorded
/// in `log` (which keeps the weighted total) and the log is flushed after
/// each batch. Prints progress per batch.
#[cfg(feature = "cuda")]
pub fn count_queens_gpu(
    n: u32,
    opts: &GpuOptions,
    items: &[crate::work::RowSet],
    log: &mut crate::rowset_log::RowsetLog,
) -> Result<RunStats, String> {
    use crate::rowset_log::RowsetCount;
    use device::{Gpu, LaunchStats};
    use std::time::Instant;

    let start = Instant::now();
    let mut gpu = Gpu::new(n, opts, items.first())?;
    let batches = plan::Batches::new(n, items, gpu.prefix_depth(), gpu.budget_words())?;

    let mut totals = [LaunchStats::default(), LaunchStats::default()]; // per colour, whole run
    let mut join_seconds = 0.0;
    let mut max_entry = 0;
    for (number, batch) in batches.enumerate() {
        let result = gpu.run_batch(&batch)?;
        // Record the batch's row sets; the flush makes them durable before
        // the next batch starts.
        for (item, &count) in batch.items.iter().zip(&result.counts) {
            let r = RowsetCount { mask: item.mask, weight: item.weight, count };
            log.record(&r).map_err(|e| format!("cannot write row-set counts: {e}"))?;
        }
        log.flush().map_err(|e| format!("cannot write row-set counts: {e}"))?;
        for (total, stats) in totals.iter_mut().zip(&result.stats) {
            total.add(stats);
        }
        join_seconds += result.join_seconds;
        max_entry = max_entry.max(result.max_entry);
        println!(
            "BATCH {}: items {}..{} of {}, 2 x {} MiB of tables; even {}; odd {}; join {:.3}s; {:.1}s elapsed",
            number + 1,
            batch.first,
            batch.first + batch.items.len(),
            items.len(),
            (batch.used_words * plan::WORD_BYTES) >> 20,
            result.stats[0],
            result.stats[1],
            result.join_seconds,
            start.elapsed().as_secs_f64()
        );
    }
    println!("even total: {}", totals[0]);
    println!("odd total:  {}", totals[1]);
    println!("join total: {join_seconds:.3}s");
    println!("largest table entry: {max_entry} (16-bit entries; every batch's tables matched the tallies)");
    Ok(RunStats {
        gpu: gpu.name().to_string(),
        prefix_depth: gpu.prefix_depth(),
        blocks_per_sm: gpu.blocks_per_sm(),
        even: totals[0],
        odd: totals[1],
        join_seconds,
        max_entry,
    })
}

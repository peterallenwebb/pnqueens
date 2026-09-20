//! Kernel emulation (feature `emulate`): `kernel/enumerate.cu` compiled as
//! host C++ by build.rs through `kernel/emulate.cpp`, run on one thread
//! against the real work plan. This checks the kernel's ticket decoding,
//! search and table indexing, and their agreement with `plan`, against the
//! known counts without a GPU. Timing and residency need the real thing.

use super::plan::{table_entries, table_words, uses_mirror, work_items, Batches, COUNTER_LEAVES, COUNTER_TALLIED, COUNTER_TICKETS, N_COUNTERS};
use crate::solution_table::{RANK_HI, RANK_LO};
use crate::KNOWN;

unsafe extern "C" {
    /// `enumerate_kernel` as a block of one thread; the first three
    /// arguments are the kernel's compile-time constants.
    fn emulate_enumerate(
        n: u32,
        prefix_depth: u32,
        mirror: u32,
        parity: u32,
        n_items: u32,
        item_rows: *const u32,
        item_table: *const u64,
        item_odd_stride: *const u32,
        ticket_base: *const u64,
        tables: *mut u32,
        rank_lo: *const u16,
        rank_hi: *const u16,
        counters: *mut u64,
    );
}

/// Q(n) by the GPU plan with the emulated kernel, in batches of about
/// `batch_tables` heaviest tables' worth of memory.
fn count_by_emulation(n: u32, prefix_depth: u32, batch_tables: usize) -> u128 {
    let items = work_items(n);
    let budget = batch_tables * table_words(table_entries(n, items[0].mask));
    let rank_hi_flat: Vec<u16> = RANK_HI.iter().flatten().copied().collect();
    let mut total: u128 = 0;
    for batch in Batches::new(n, &items, prefix_depth, budget).unwrap() {
        let mut tables = [vec![0u32; batch.used_words.max(1)], vec![0u32; batch.used_words.max(1)]];
        for (parity, launch) in batch.launches.iter().enumerate() {
            let mut counters = [0u64; N_COUNTERS];
            unsafe {
                emulate_enumerate(
                    n,
                    prefix_depth,
                    u32::from(uses_mirror(n)),
                    parity as u32,
                    launch.rows.len() as u32,
                    launch.rows.as_ptr(),
                    launch.table_offset.as_ptr(),
                    launch.odd_stride.as_ptr(),
                    launch.ticket_base.as_ptr(),
                    tables[parity].as_mut_ptr(),
                    RANK_LO.as_ptr(),
                    rank_hi_flat.as_ptr(),
                    counters.as_mut_ptr(),
                );
            }
            assert_eq!(counters[COUNTER_TICKETS], launch.tickets(), "every ticket is taken exactly once");
            // The tables hold the tallied weights, as the join's check on the GPU requires.
            let in_tables: u64 = tables[parity].iter().map(|&w| u64::from(w & 0xFFFF) + u64::from(w >> 16)).sum();
            assert_eq!(in_tables, counters[COUNTER_TALLIED], "every tallied weight is in the tables");
            assert!(in_tables >= counters[COUNTER_LEAVES], "every leaf is tallied");
        }
        // The join, over packed 16-bit entries.
        for (i, item) in batch.items.iter().enumerate() {
            let offset = batch.launches[0].table_offset[i] as usize;
            let len = batch.words[i] as usize;
            let e = &tables[0][offset..offset + len];
            let o = &tables[1][offset..offset + len];
            let count: u64 = e
                .iter()
                .zip(o)
                .map(|(&e, &o)| u64::from(e & 0xFFFF) * u64::from(o & 0xFFFF) + u64::from(e >> 16) * u64::from(o >> 16))
                .sum();
            total += u128::from(item.weight as u32) * u128::from(count);
        }
    }
    total
}

#[test]
fn emulated_kernel_matches_known_counts() {
    for n in 1..=12 {
        for prefix_depth in 1..=4 {
            assert_eq!(count_by_emulation(n, prefix_depth, 3), KNOWN[n as usize - 1], "Q({n}) at prefix depth {prefix_depth}");
        }
    }
    // A prefix deeper than every row set, and a single batch.
    assert_eq!(count_by_emulation(9, 12, 1000), KNOWN[8]);
}

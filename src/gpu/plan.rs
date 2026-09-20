//! The work plan of a GPU run: which row sets to count, how their tables are
//! laid out, how each search is cut into tickets, and how items are grouped
//! into batches. Nothing here touches a device, so the plan is shared by the
//! CUDA driver and by the kernel emulation test, and is tested on any
//! machine.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use crate::solution_table::SolutionTable;
use crate::{complement, get_feasible_even_row_combos, RowCombo};

/// Bytes of one table entry: the kernels tally and join in u32.
pub const ENTRY_BYTES: usize = std::mem::size_of::<u32>();

/// Layout of the enumerate kernel's counter buffer: index 0 is the ticket
/// counter the kernel claims work from, the rest are the statistics it adds
/// up at the end. Must match the enum in `kernel/enumerate.cu`.
pub const COUNTER_TICKETS: usize = 1;
pub const COUNTER_DEAD: usize = 2;
pub const COUNTER_STEPS: usize = 3;
pub const COUNTER_LEAVES: usize = 4;
pub const N_COUNTERS: usize = 5;

/// The smallest prefix depth worth using.
pub const MIN_PREFIX_DEPTH: u32 = 3;

/// Tickets in progress at a moment, as a multiple of the resident threads.
/// Threads take tickets in order, but a slow ticket keeps its thread on an
/// old item while the others move on, so the window is wider than one per
/// thread.
const TICKET_WINDOW_FACTOR: u64 = 2;

/// The feasible even-queen row sets with their symmetry weights, heaviest
/// table first, so that each batch holds items of similar size.
pub fn work_items(n: u32) -> Vec<RowCombo> {
    let mut items = get_feasible_even_row_combos(n);
    items.sort_by_key(|item| std::cmp::Reverse(table_entries(n, item.mask)));
    items
}

/// The shape of the two tables of the item with row set R, as (entries,
/// odd stride). The E table of R on even squares and the O table of R̅ on
/// odd squares have the same shape: an odd-square placement on R̅ uses the
/// column parities R's even-square placements leave free, and the kernel
/// indexes the O table by the complement of the leaf's columns.
pub fn table_shape(n: u32, rows: u32) -> (usize, u32) {
    SolutionTable::dimensions(n, rows, 0)
}

/// Entries in each table of the item with row set `rows`.
pub fn table_entries(n: u32, rows: u32) -> usize {
    table_shape(n, rows).0
}

/// Whether the kernel applies the mirror symmetry of `backtrack_odd_solutions`
/// to the odd-square search: odd n only (n = 1 has no odd-square rows to restrict).
pub fn uses_mirror(n: u32) -> bool {
    n % 2 == 1 && n >= 3
}

/// Number of tickets for one search: the product, over the first
/// `prefix_depth` rows of `rows`, of the number of columns the ticket digit
/// for that row ranges over. That is the colour-compatible columns of the
/// row, except that with the mirror symmetry the first row of an odd-square
/// search is limited to the columns up to the centre. This must match the
/// kernel's `digit_radix` exactly.
pub fn ticket_count(n: u32, rows: u32, parity: u32, prefix_depth: u32) -> u64 {
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
pub fn default_prefix_depth(n: u32, heaviest: &RowCombo, threads: u32, l2_bytes: u64) -> u32 {
    let table_bytes = (table_entries(n, heaviest.mask) * ENTRY_BYTES) as u64;
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

/// The per-item arrays of one enumerate launch, in the layout the kernel reads.
pub struct Launch {
    /// Row set of each item: R for the even-square search, R̅ for the odd-square one.
    pub rows: Vec<u32>,
    /// Offset of each item's table in the colour's table buffer, in entries.
    pub table_offset: Vec<u64>,
    /// Odd stride of each item's table.
    pub odd_stride: Vec<u32>,
    /// `ticket_base[i]` is the first ticket of item i; the last entry is the total.
    pub ticket_base: Vec<u64>,
}

impl Launch {
    fn new() -> Self {
        Launch { rows: Vec::new(), table_offset: Vec::new(), odd_stride: Vec::new(), ticket_base: vec![0] }
    }

    fn push(&mut self, rows: u32, table_offset: u64, odd_stride: u32, tickets: u64) {
        self.rows.push(rows);
        self.table_offset.push(table_offset);
        self.odd_stride.push(odd_stride);
        let next = self.tickets() + tickets;
        self.ticket_base.push(next);
    }

    /// Total tickets of the launch.
    pub fn tickets(&self) -> u64 {
        *self.ticket_base.last().unwrap()
    }
}

/// One batch of work: a run of consecutive items of the sorted list whose
/// tables fit the budget, with the launch arrays for both colours.
pub struct Batch<'a> {
    /// Position of the batch's first item in the full list, for progress messages.
    pub first: usize,
    pub items: &'a [RowCombo],
    /// Per colour: [0] searches R on even squares, [1] searches R̅ on odd squares.
    pub launches: [Launch; 2],
    /// Table length of each item, for the join.
    pub entries: Vec<u32>,
    /// Entries of table memory the batch uses, per colour.
    pub used_entries: usize,
}

/// Cuts the sorted item list into batches that fit the table budget.
pub struct Batches<'a> {
    n: u32,
    items: &'a [RowCombo],
    next: usize,
    prefix_depth: u32,
    budget_entries: usize,
}

impl<'a> Batches<'a> {
    /// `budget_entries` is the table memory available per colour, in entries.
    pub fn new(n: u32, items: &'a [RowCombo], prefix_depth: u32, budget_entries: usize) -> Result<Self, String> {
        let heaviest = items.iter().map(|item| table_entries(n, item.mask)).max().unwrap_or(0);
        if heaviest > budget_entries {
            return Err(format!(
                "table budget of {} MiB is too small for one pair of tables of {heaviest} entries",
                (2 * budget_entries * ENTRY_BYTES) >> 20
            ));
        }
        Ok(Batches { n, items, next: 0, prefix_depth, budget_entries })
    }
}

impl<'a> Iterator for Batches<'a> {
    type Item = Batch<'a>;

    fn next(&mut self) -> Option<Batch<'a>> {
        if self.next >= self.items.len() {
            return None;
        }
        let first = self.next;
        let mut launches = [Launch::new(), Launch::new()];
        let mut entries = Vec::new();
        let mut used = 0;
        while let Some(item) = self.items.get(self.next) {
            let (len, odd_stride) = table_shape(self.n, item.mask);
            if used + len > self.budget_entries {
                break;
            }
            let rows_bar = complement(self.n, item.mask);
            launches[0].push(item.mask, used as u64, odd_stride, ticket_count(self.n, item.mask, 0, self.prefix_depth));
            launches[1].push(rows_bar, used as u64, odd_stride, ticket_count(self.n, rows_bar, 1, self.prefix_depth));
            entries.push(len as u32);
            used += len;
            self.next += 1;
        }
        Some(Batch { first, items: &self.items[first..self.next], launches, entries, used_entries: used })
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

    #[test]
    fn batches_cover_the_items_in_order_within_budget() {
        let n = 12;
        let items = work_items(n);
        let heaviest = table_entries(n, items[0].mask);
        let budget = 3 * heaviest;
        assert!(Batches::new(n, &items, 3, heaviest - 1).is_err(), "a budget below the heaviest table is refused");

        let mut seen = 0;
        for batch in Batches::new(n, &items, 3, budget).unwrap() {
            assert_eq!(batch.first, seen);
            assert!(!batch.items.is_empty());
            assert!(batch.used_entries <= budget);
            assert_eq!(batch.entries.len(), batch.items.len());
            for launch in &batch.launches {
                assert_eq!(launch.rows.len(), batch.items.len());
                assert_eq!(launch.ticket_base.len(), batch.items.len() + 1);
                assert!(launch.ticket_base.windows(2).all(|w| w[0] < w[1]), "every item has at least one ticket");
            }
            // Tables are laid out back to back, the same offsets for both colours.
            let mut offset = 0;
            for (i, item) in batch.items.iter().enumerate() {
                assert_eq!(batch.launches[0].table_offset[i], offset);
                assert_eq!(batch.launches[1].table_offset[i], offset);
                assert_eq!(batch.launches[0].rows[i], item.mask);
                assert_eq!(batch.launches[1].rows[i], complement(n, item.mask));
                assert_eq!(batch.entries[i] as usize, table_entries(n, item.mask));
                offset += u64::from(batch.entries[i]);
            }
            assert_eq!(offset as usize, batch.used_entries);
            seen += batch.items.len();
        }
        assert_eq!(seen, items.len());
    }

    #[test]
    fn default_prefix_depth_on_a_5090() {
        // 170 SMs at 256 resident threads, 96 MB of L2: depth 3 up to n = 22,
        // 4 for n = 23..26, 5 for n = 27..29 (see cuda-notes.md).
        let threads = 170 * 256;
        let l2 = 96 << 20;
        for (n, depth) in [(20, 3), (22, 3), (23, 4), (24, 4), (26, 4), (27, 5), (28, 5), (29, 5)] {
            let items = work_items(n);
            assert_eq!(default_prefix_depth(n, &items[0], threads, l2), depth, "n = {n}");
        }
    }
}

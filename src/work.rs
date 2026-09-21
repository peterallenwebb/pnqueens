//! What a count works through: the feasible even-queen row sets with their
//! symmetry weights, in the order every run uses, the fingerprint that
//! identifies that list, and the selection of one phase's share of it.
//!
//! A phase is a set of positions in the list (every `stride`-th item from
//! `phase`), so phases of a run are only compatible when they were cut from
//! the same list: the fingerprint travels with every record and the merge
//! requires it to be the same everywhere.

use crate::solution_table::SolutionTable;

/// A row set with the number of row sets its symmetry orbit stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowSet {
    /// Bit r set: a queen on an even square of row r.
    pub mask: u32,
    pub weight: u32,
}

/// Complement of an n-bit mask.
pub fn complement(n: u32, x: u32) -> u32 {
    !x & ((1 << n) - 1)
}

/// Reflection of an n-bit mask (row r to row n - 1 - r).
pub fn reflect(n: u32, x: u32) -> u32 {
    x.reverse_bits() >> (32 - n)
}

/// The feasible even-queen row sets, one representative per symmetry orbit
/// with the orbit's size as its weight, in increasing order of mask. Only
/// row sets with (even rows) - (odd rows) = n mod 2 can carry queens (see
/// the module comment of main.rs); of a set and its reflection only the
/// smaller is kept (weight 2), and for even n likewise the reflection of
/// the complement, which stands for the odd-square queens (weight 4).
pub fn feasible_row_sets(n: u32) -> Vec<RowSet> {
    let mut sets = Vec::new();
    for mask in 0..(1u32 << n) {
        let k00 = (mask & 0x5555_5555).count_ones();
        let k11 = (mask & 0xAAAA_AAAA).count_ones();
        if k00 != k11 + n % 2 {
            continue;
        }
        let reflection = reflect(n, mask);
        if mask > reflection {
            continue;
        }
        if !n.is_multiple_of(2) {
            sets.push(RowSet { mask, weight: if mask == reflection { 1 } else { 2 } });
        } else {
            let rc = reflect(n, complement(n, mask));
            if mask <= rc {
                sets.push(RowSet { mask, weight: if mask == reflection || mask == rc { 2 } else { 4 } });
            }
        }
    }
    sets
}

/// The row sets in the order runs process them: heaviest table first, so
/// that a GPU batch holds items of similar size and a phase samples heavy
/// and light items alike. The sort is stable, so the order is a function of
/// n alone.
pub fn work_items(n: u32) -> Vec<RowSet> {
    let mut items = feasible_row_sets(n);
    items.sort_by_key(|item| std::cmp::Reverse(SolutionTable::dimensions(n, item.mask, 0).0));
    items
}

/// A 64-bit fingerprint of n and the item list in order (FNV-1a over the
/// little-endian bytes of n, then of every mask and weight).
pub fn fingerprint(n: u32, items: &[RowSet]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |v: u32| {
        for byte in v.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    eat(n);
    for item in items {
        eat(item.mask);
        eat(item.weight);
    }
    h
}

/// One phase's share of the items: the positions congruent to `phase`
/// modulo `stride`. The `stride` phases of a stride partition the list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selection {
    pub stride: u32,
    pub phase: u32,
}

impl Selection {
    pub const ALL: Selection = Selection { stride: 1, phase: 0 };

    pub fn is_all(&self) -> bool {
        self.stride == 1
    }

    pub fn pick<'a>(&self, items: &'a [RowSet]) -> impl Iterator<Item = &'a RowSet> + 'a {
        items.iter().skip(self.phase as usize).step_by(self.stride as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_account_for_every_feasible_row_set() {
        for n in 1..=14 {
            let unreduced = (0..(1u32 << n))
                .filter(|&m| (m & 0x5555_5555).count_ones() == (m & 0xAAAA_AAAA).count_ones() + n % 2)
                .count();
            let weighted: u32 = feasible_row_sets(n).iter().map(|r| r.weight).sum();
            assert_eq!(weighted as usize, unreduced, "n = {n}");
        }
    }

    #[test]
    fn phases_partition_the_items() {
        let items = work_items(12);
        for stride in [1, 2, 7, 100] {
            let mut seen = vec![0u32; items.len()];
            for phase in 0..stride {
                for item in (Selection { stride, phase }).pick(&items) {
                    let pos = items.iter().position(|i| i == item).unwrap();
                    seen[pos] += 1;
                }
            }
            assert!(seen.iter().all(|&c| c == 1), "stride {stride}");
        }
    }

    #[test]
    fn fingerprints_are_stable() {
        // Golden values: a change here means the item order changed, which
        // makes records of different builds incompatible. Change consciously.
        assert_eq!(fingerprint(8, &work_items(8)), 0x7569_0c3e_1bb0_29f0);
        assert_eq!(fingerprint(12, &work_items(12)), 0x78f1_f517_705c_9b68);
        assert_ne!(fingerprint(12, &work_items(12)), fingerprint(13, &work_items(13)));
    }
}

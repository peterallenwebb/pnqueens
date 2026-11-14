//! The conventional count of Q(n): one queen per row, backtracking down the
//! rows with the attacked columns kept as bitmasks. It is here as the
//! baseline the decomposition in main.rs is measured against, and shares
//! nothing with it (cpu-comparison.md).
//!
//! The tricks are the usual ones. The squares of the current row attacked
//! by the queens above it are three masks: the columns taken, and the two
//! diagonal families, each of which shifts by one column when the search
//! steps down a row. The free columns of a row are then one AND-NOT, and
//! candidates are taken lowest bit first. The last row is not searched:
//! every free column in it is one solution, so the row above it adds a
//! population count. The mirror c -> n-1-c maps solutions to solutions and
//! fixes none (a fixed solution would have two queens in a row), so the
//! first row's queen is placed only left of centre and the count doubled;
//! for odd n the centre column is placed too, and the second row's queen is
//! the one kept left of centre.

/// Q(n) for 1 <= n <= 31 by plain backtracking, on the calling thread.
pub fn count(n: u32) -> u128 {
    assert!((1..=31).contains(&n));
    if n == 1 {
        return 1;
    }
    let all = (1u32 << n) - 1;
    let left = (1u32 << (n / 2)) - 1;
    // The first queen left of centre: half the solutions, their mirror images the rest.
    let mut half = search(all, left, 0, 0, 0, n - 1);
    if !n.is_multiple_of(2) {
        // The first queen in the centre column, the second left of centre.
        let c = 1u32 << (n / 2);
        let (cols, ld, rd) = (c, c << 1, c >> 1);
        half += search(all, left & !(cols | ld | rd), cols, ld, rd, n - 2);
    }
    2 * u128::from(half)
}

/// The completions of a partial placement. `cols`, `ld` and `rd` are the
/// columns of the current row attacked by the queens above it (columns taken,
/// and the two diagonal families), `free` the columns of the current row
/// still to be tried, and `below` the number of rows under the current one.
fn search(all: u32, mut free: u32, cols: u32, ld: u32, rd: u32, below: u32) -> u64 {
    if below == 0 {
        return u64::from(free.count_ones());
    }
    let mut total = 0;
    while free != 0 {
        let bit = free & free.wrapping_neg();
        free ^= bit;
        let cols = cols | bit;
        let ld = (ld | bit) << 1;
        let rd = (rd | bit) >> 1;
        total += search(all, all & !(cols | ld | rd), cols, ld, rd, below - 1);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KNOWN;

    #[test]
    fn conventional_count_matches_the_published_values() {
        // Q(1) .. Q(14); larger n take too long in a debug build.
        for (i, &q) in KNOWN[..14].iter().enumerate() {
            let n = i as u32 + 1;
            assert_eq!(count(n), q, "Q({n})");
        }
    }
}

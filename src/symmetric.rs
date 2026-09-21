//! An independent check on a count. The dihedral group of the square acts on
//! solutions and no solution is fixed by a reflection, so every orbit has 8,
//! 4 or 2 solutions: 4 when the solution is fixed by the 180° rotation only,
//! 2 when by the 90° rotation. With k8, k4, k2 orbits of each size,
//!
//!     Q = 8 k8 + 4 k4 + 2 k2   and   S2 = 4 k4 + 2 k2,
//!
//! where S2 is the number of solutions fixed by the 180° rotation, so
//! Q ≡ S2 (mod 8). S2 is counted here directly, by a search that shares
//! nothing with the checkerboard decomposition: a queen at (r, c) forces
//! one at (n-1-r, n-1-c), so the first half of the rows decides everything.

use rayon::prelude::*;

/// The number of solutions fixed by the 180° rotation, for 1 <= n <= 31.
pub fn rotation_fixed_solutions(n: u32) -> u64 {
    assert!((1..=31).contains(&n));
    let mut board = Board { n, cols: 0, d1: 0, d2: 0 };
    if !n.is_multiple_of(2) {
        // The centre square is its own image and must hold a queen.
        let m = (n - 1) / 2;
        board.place(m, m);
    }
    let half = n / 2;
    if half == 0 {
        return 1;
    }
    // The first row's choices in parallel; each is its own search.
    board
        .free(0)
        .into_par_iter()
        .map(|c| {
            let mut b = board;
            if b.place_pair(0, c) { b.count(1, half) } else { 0 }
        })
        .sum()
}

/// Queens placed so far: the columns taken and the diagonals occupied, in
/// board coordinates (bit c - r + n - 1 and bit c + r), so that a queen in
/// any row can be tested against them.
#[derive(Clone, Copy)]
struct Board {
    n: u32,
    cols: u32,
    d1: u64,
    d2: u64,
}

impl Board {
    fn attacked(&self, r: u32) -> u32 {
        self.cols | (self.d1 >> (self.n - 1 - r)) as u32 | (self.d2 >> r) as u32
    }

    /// The columns of row r no placed queen attacks.
    fn free(&self, r: u32) -> Vec<u32> {
        let full = (1u32 << self.n) - 1;
        let mut free = full & !self.attacked(r);
        let mut out = Vec::new();
        while free != 0 {
            out.push(free.trailing_zeros());
            free &= free - 1;
        }
        out
    }

    fn place(&mut self, r: u32, c: u32) {
        self.cols |= 1 << c;
        self.d1 |= 1u64 << (c + self.n - 1 - r);
        self.d2 |= 1u64 << (c + r);
    }

    /// Places a queen at (r, c) and its image at (n-1-r, n-1-c), if the two
    /// do not attack each other (they share a diagonal when c = r or
    /// c + r = n - 1) and the image's square is free. r is in the first
    /// half, so the image is in a later row than any queen placed so far
    /// except the other images, which the masks cover.
    fn place_pair(&mut self, r: u32, c: u32) -> bool {
        let (rp, cp) = (self.n - 1 - r, self.n - 1 - c);
        if c == r || c + r == self.n - 1 || (self.attacked(rp) >> cp) & 1 == 1 {
            return false;
        }
        self.place(r, c);
        self.place(rp, cp);
        true
    }

    /// Completions of rows r .. half-1 (each with its image).
    fn count(&self, r: u32, half: u32) -> u64 {
        if r == half {
            return 1;
        }
        let mut total = 0;
        for c in self.free(r) {
            let mut b = *self;
            if b.place_pair(r, c) {
                total += b.count(r + 1, half);
            }
        }
        total
    }
}

/// Every solution of the n-queens problem as its column per row, by plain
/// backtracking. For tests: independent of everything else in this program.
#[cfg(test)]
pub fn brute_force_solutions(n: u32) -> Vec<Vec<u32>> {
    fn go(n: u32, row: u32, cols: u32, ld: u32, rd: u32, placed: &mut Vec<u32>, out: &mut Vec<Vec<u32>>) {
        if row == n {
            out.push(placed.clone());
            return;
        }
        let full = (1u32 << n) - 1;
        let mut free = full & !(cols | ld | rd);
        while free != 0 {
            let bit = free & free.wrapping_neg();
            free ^= bit;
            placed.push(bit.trailing_zeros());
            go(n, row + 1, cols | bit, (ld | bit) << 1, (rd | bit) >> 1, placed, out);
            placed.pop();
        }
    }
    let mut out = Vec::new();
    go(n, 0, 0, 0, 0, &mut Vec::new(), &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KNOWN;

    #[test]
    fn rotation_fixed_count_matches_brute_force() {
        for n in 1..=11 {
            let solutions = brute_force_solutions(n);
            assert_eq!(solutions.len() as u128, KNOWN[n as usize - 1], "Q({n}) by brute force");
            let fixed = solutions
                .iter()
                .filter(|s| (0..n).all(|r| s[(n - 1 - r) as usize] == n - 1 - s[r as usize]))
                .count() as u64;
            assert_eq!(rotation_fixed_solutions(n), fixed, "S2({n})");
        }
    }

    #[test]
    fn the_mod_8_identity_holds_on_the_published_values() {
        for n in 1..=22 {
            let s2 = u128::from(rotation_fixed_solutions(n));
            assert_eq!(KNOWN[n as usize - 1] % 8, s2 % 8, "Q({n}) ≡ S2 (mod 8)");
        }
    }
}

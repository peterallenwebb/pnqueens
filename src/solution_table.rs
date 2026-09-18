//! Fast storage for E[R, ·] indexed by column set. This allows us to avoid
//! hashing overhead.
//!
//! For a fixed row set R and color, every placement uses exactly `k_even`
//! even columns and `k_odd` odd columns (for even squares, k_even = |R ∩ even
//! rows| and k_odd = |R ∩ odd rows|; for odd squares the roles swap). So a
//! column set C is a pair (k_even-subset of the even columns, k_odd-subset of
//! the odd columns), and there are exactly
//!
//!     binom(m_even, k_even) * binom(m_odd, k_odd)
//!
//! of them, where m_even = ceil(n/2) and m_odd = floor(n/2). We map each C to a
//! distinct index in that range by ranking the two subsets in the
//! combinatorial number system (colex order, the order Algorithm T produces):
//!
//!     rank({c_1 < c_2 < ... < c_k}) = sum_i binom(c_i, i)
//!
//! and combining `rank_even * binom(m_odd, k_odd) + rank_odd`.
//!
//! Column sets are handled in "dense" form: even column c is bit c/2 of the
//! even mask, odd column c is bit (c-1)/2 of the odd mask (`c >> 1` in both
//! cases). Searches can maintain these masks incrementally; [`split_columns`]
//! converts from a full n-bit mask when needed. Ranking a dense mask is two
//! table lookups (low byte, high byte given the low byte's popcount), so an
//! index costs a few nanoseconds.

/// The count type for table entries. Individual E[R, C] values are far
/// smaller than Q(n), so u32 is safe. An overflow panics.
pub type Count = u32;

/// Maximum number of columns of one parity: n <= 32 gives at most 16 even
/// columns and 16 odd columns, so dense masks fit in 16 bits.
const MAX_DENSE_BITS: usize = 16;

/// BINOM[m][k] = binom(m, k) for 0 <= m, k <= 16.
const BINOM: [[u32; MAX_DENSE_BITS + 1]; MAX_DENSE_BITS + 1] = build_binom();

const fn build_binom() -> [[u32; MAX_DENSE_BITS + 1]; MAX_DENSE_BITS + 1] {
    let mut b = [[0u32; MAX_DENSE_BITS + 1]; MAX_DENSE_BITS + 1];
    let mut m = 0;
    while m <= MAX_DENSE_BITS {
        b[m][0] = 1;
        let mut k = 1;
        while k <= m {
            b[m][k] = b[m - 1][k - 1] + if k < m { b[m - 1][k] } else { 0 };
            k += 1;
        }
        m += 1;
    }
    b
}

/// binom(m, k), zero when k > m. Only valid for m <= 16.
pub const fn binom(m: u32, k: u32) -> u32 {
    if k > m { 0 } else { BINOM[m as usize][k as usize] }
}

/// RANK_LO[lo] = rank contribution of the low byte of a dense mask.
/// (Public so the GPU path can upload the same tables the CPU uses.)
pub const RANK_LO: [u16; 256] = build_rank_lo();

/// RANK_HI[hi][k_lo] = rank contribution of the high byte, given that the low
/// byte has k_lo bits set (which offsets the subscript i of every high bit).
pub const RANK_HI: [[u16; 9]; 256] = build_rank_hi();

const fn build_rank_lo() -> [u16; 256] {
    let mut t = [0u16; 256];
    let mut lo = 0usize;
    while lo < 256 {
        let mut r = 0u32;
        let mut i = 0u32;
        let mut pos = 0u32;
        while pos < 8 {
            if (lo >> pos) & 1 == 1 {
                i += 1;
                r += BINOM[pos as usize][i as usize];
            }
            pos += 1;
        }
        t[lo] = r as u16;
        lo += 1;
    }
    t
}

const fn build_rank_hi() -> [[u16; 9]; 256] {
    let mut t = [[0u16; 9]; 256];
    let mut hi = 0usize;
    while hi < 256 {
        let mut k_lo = 0usize;
        while k_lo <= 8 {
            let mut r = 0u32;
            let mut i = k_lo as u32;
            let mut pos = 0u32;
            while pos < 8 {
                if (hi >> pos) & 1 == 1 {
                    i += 1;
                    r += BINOM[(8 + pos) as usize][i as usize];
                }
                pos += 1;
            }
            t[hi][k_lo] = r as u16;
            k_lo += 1;
        }
        hi += 1;
    }
    t
}

/// Combinadic rank of a dense (<= 16-bit) mask: two table lookups.
#[inline]
pub fn rank(dense: u16) -> u32 {
    let lo = (dense & 0xFF) as usize;
    let hi = (dense >> 8) as usize;
    RANK_LO[lo] as u32 + RANK_HI[hi][lo.count_ones() as usize] as u32
}

/// Gathers the even-position bits of `x` into the low 16 bits:
/// bit 2i of the input becomes bit i of the output.
#[inline]
pub fn gather_even_bits(mut x: u32) -> u16 {
    x &= 0x5555_5555;
    x = (x | (x >> 1)) & 0x3333_3333;
    x = (x | (x >> 2)) & 0x0F0F_0F0F;
    x = (x | (x >> 4)) & 0x00FF_00FF;
    x = (x | (x >> 8)) & 0x0000_FFFF;
    x as u16
}

/// Splits a full n-bit column mask into dense (even, odd) masks.
#[inline]
pub fn split_columns(cols: u32) -> (u16, u16) {
    (gather_even_bits(cols), gather_even_bits(cols >> 1))
}

/// Counts indexed by column set, for column sets with a fixed number of even
/// and odd columns on an n-board. See the module docs for the indexing scheme.
#[derive(Debug, Clone)]
pub struct SolutionTable {
    n: u32,
    /// Number of even columns in every stored column set.
    k_even: u32,
    /// Number of odd columns in every stored column set.
    k_odd: u32,
    /// binom(m_odd, k_odd): the multiplier for the even rank.
    odd_stride: u32,
    counts: Vec<Count>,
}

impl SolutionTable {
    /// A zeroed table for column sets with exactly `k_even` even columns and
    /// `k_odd` odd columns on an n-board. If either count exceeds the number
    /// of columns of that parity, the table is empty (no such column sets).
    pub fn new(n: u32, k_even: u32, k_odd: u32) -> Self {
        assert!((1..=32).contains(&n), "board size must be in 1..=32");
        let (m_even, m_odd) = Self::column_counts(n);
        let odd_stride = binom(m_odd, k_odd);
        let len = binom(m_even, k_even) as usize * odd_stride as usize;
        SolutionTable { n, k_even, k_odd, odd_stride, counts: vec![0; len] }
    }

    /// A zeroed table sized for the column sets reachable by placing one queen
    /// in each row of `rows` on squares of the given color (`parity` 0 for
    /// even squares, 1 for odd). Even squares pair even rows with even
    /// columns; odd squares pair even rows with odd columns.
    pub fn for_rows(n: u32, rows: u32, parity: u32) -> Self {
        let (k_even, k_odd) = Self::column_usage(rows, parity);
        Self::new(n, k_even, k_odd)
    }

    /// The size of the table `for_rows` would build, as (number of entries,
    /// odd stride), without allocating it. The GPU path uses this to lay out
    /// many tables in one device buffer.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn dimensions(n: u32, rows: u32, parity: u32) -> (usize, u32) {
        let (m_even, m_odd) = Self::column_counts(n);
        let (k_even, k_odd) = Self::column_usage(rows, parity);
        let odd_stride = binom(m_odd, k_odd);
        (binom(m_even, k_even) as usize * odd_stride as usize, odd_stride)
    }

    /// (even columns used, odd columns used) by placements on `rows` with the
    /// given colour.
    fn column_usage(rows: u32, parity: u32) -> (u32, u32) {
        assert!(parity <= 1);
        let even_rows = (rows & 0x5555_5555).count_ones();
        let odd_rows = (rows & 0xAAAA_AAAA).count_ones();
        if parity == 0 { (even_rows, odd_rows) } else { (odd_rows, even_rows) }
    }

    /// (number of even columns, number of odd columns) on an n-board.
    fn column_counts(n: u32) -> (u32, u32) {
        (n.div_ceil(2), n / 2)
    }

    /// Index of the column set given as dense (even, odd) masks.
    #[inline]
    pub fn index_dense(&self, even: u16, odd: u16) -> usize {
        debug_assert_eq!(even.count_ones(), self.k_even, "wrong number of even columns");
        debug_assert_eq!(odd.count_ones(), self.k_odd, "wrong number of odd columns");
        (rank(even) * self.odd_stride + rank(odd)) as usize
    }

    /// Adds one to the count for the column set given as dense masks.
    #[inline]
    pub fn increment_dense(&mut self, even: u16, odd: u16) {
        let i = self.index_dense(even, odd);
        let slot = &mut self.counts[i];
        *slot = slot.checked_add(1).expect("SolutionTable count overflow");
    }

    /// Count for the column set given as dense masks.
    #[inline]
    pub fn get_dense(&self, even: u16, odd: u16) -> Count {
        self.counts[self.index_dense(even, odd)]
    }

    /// Dense masks of the complement of the given dense column set, i.e. the
    /// columns of this board *not* in it. Used by the odd-square search, which
    /// reaches a leaf holding C̅ and needs E[R, C].
    #[inline]
    pub fn complement_dense(&self, even: u16, odd: u16) -> (u16, u16) {
        let (m_even, m_odd) = Self::column_counts(self.n);
        let even_full = ((1u32 << m_even) - 1) as u16;
        let odd_full = ((1u32 << m_odd) - 1) as u16;
        (!even & even_full, !odd & odd_full)
    }

    /// Count for the complement of the column set given as dense masks.
    #[inline]
    pub fn get_complement_dense(&self, even: u16, odd: u16) -> Count {
        let (even, odd) = self.complement_dense(even, odd);
        self.get_dense(even, odd)
    }
}

//! Per-row-set results on disk: the records of a computation.
//!
//! Q(n) is a sum over the feasible even-queen row sets R of
//! weight(R) * count(R), where count(R) = sum over C of E[R, C] * O[R̅, C̅].
//! Both the CPU and the GPU paths record every (R, weight, count) as it is
//! produced, so a finished run can be re-summed, compared row set by row set
//! with another run, or audited one row set at a time (`--rowset`).
//!
//! A file is CSV with a provenance line, a header line and one line per row
//! set:
//!
//!     # pnqueens n=8 stride=1 phase=0 workmap=76f14ac1f9ff5e3b items=71 ...
//!     n,rowset_hex,weight,count,even_leaves,odd_leaves
//!     8,0000000f,4,0,16,4
//!
//! `rowset_hex` is the row bitmask (bit r set = a queen on an even square of
//! row r), `weight` the size of its symmetry orbit, `count` the unweighted
//! count, `even_leaves` the number of even-square placements on those rows
//! and `odd_leaves` the number of odd-square placements on the other rows
//! (the two solution sets whose join the count is; each can be checked
//! without the tables). Lines are written in the order results arrive and
//! flushed when the caller asks (after each GPU batch), so an interrupted
//! run keeps the row sets it finished. `n` is repeated on every line so
//! files from different machines can be concatenated and checked together.
//! The provenance line holds `key=value` pairs (see `count` in main.rs).
//! Files from before 2026-09-22 have neither the provenance line nor the
//! leaf columns; they are read without leaf counts.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// One row set's result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowsetCount {
    pub mask: u32,
    pub weight: u32,
    pub count: u64,
    /// The numbers of even-square and odd-square placements joined; None in
    /// records from before the columns existed.
    pub leaves: Option<[u64; 2]>,
}

impl RowsetCount {
    /// This row set's contribution to Q(n).
    pub fn weighted(&self) -> u64 {
        u64::from(self.weight).checked_mul(self.count).expect("weighted count overflows u64")
    }

    /// Whether two records of the same row set say the same: the same
    /// count, and the same leaf counts when both have them.
    pub fn agrees_with(&self, other: &RowsetCount) -> bool {
        self.weight == other.weight && self.count == other.count && match (self.leaves, other.leaves) {
            (Some(a), Some(b)) => a == b,
            _ => true,
        }
    }
}

/// Writes row-set results as they arrive and keeps the running total. The
/// total is u128: Q(29) exceeds u64, though every row set's count fits.
pub struct RowsetLog {
    n: u32,
    out: Box<dyn Write>,
    total: u128,
    written: usize,
}

impl RowsetLog {
    /// A log to a new file, headed by the provenance line. Refuses to
    /// overwrite an existing file: a finished run's results should not be
    /// lost to a rerun with the same name.
    pub fn create(path: &Path, n: u32, provenance: &str) -> io::Result<Self> {
        let file = File::create_new(path)?;
        let mut log = Self::new(n, Box::new(BufWriter::new(file)));
        writeln!(log.out, "# {provenance}")?;
        writeln!(log.out, "n,rowset_hex,weight,count,even_leaves,odd_leaves")?;
        Ok(log)
    }

    /// A log that only keeps the total (for tests).
    #[cfg(test)]
    pub fn discard(n: u32) -> Self {
        Self::new(n, Box::new(io::sink()))
    }

    fn new(n: u32, out: Box<dyn Write>) -> Self {
        RowsetLog { n, out, total: 0, written: 0 }
    }

    pub fn record(&mut self, r: &RowsetCount) -> io::Result<()> {
        let [even, odd] = r.leaves.expect("a record written by this program has its leaf counts");
        writeln!(self.out, "{},{:08x},{},{},{},{}", self.n, r.mask, r.weight, r.count, even, odd)?;
        self.total += u128::from(r.weighted());
        self.written += 1;
        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.out.flush()
    }

    /// Sum of weight * count over everything recorded so far.
    pub fn total(&self) -> u128 {
        self.total
    }

    /// Number of row sets recorded so far.
    pub fn written(&self) -> usize {
        self.written
    }
}

/// The `key=value` pairs of a file's provenance line.
pub type Provenance = BTreeMap<String, String>;

/// Everything read from one or more record files: n, the rows of all files
/// in order, and each file's provenance (empty for a file without the line).
pub struct Records {
    pub n: u32,
    pub rows: Vec<RowsetCount>,
    pub provenance: Vec<(String, Provenance)>,
}

/// Reads record files written by `RowsetLog`. All lines must have the same n.
pub fn read_rowset_files(paths: &[String]) -> Result<Records, String> {
    let mut n: Option<u32> = None;
    let mut rows = Vec::new();
    let mut provenance = Vec::new();
    for path in paths {
        let file = File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let mut prov = Provenance::new();
        for (i, line) in BufReader::new(file).lines().enumerate() {
            let line = line.map_err(|e| format!("{path}: {e}"))?;
            let line = line.trim();
            if let Some(rest) = line.strip_prefix('#') {
                for pair in rest.split_whitespace() {
                    if let Some((k, v)) = pair.split_once('=') {
                        prov.insert(k.to_string(), v.to_string());
                    }
                }
                continue;
            }
            if line.is_empty() || line.starts_with("n,") {
                continue; // header (or a blank line from concatenation)
            }
            let bad = || format!("{path}:{}: cannot parse '{line}'", i + 1);
            let fields: Vec<&str> = line.split(',').collect();
            let (n_str, mask_hex, weight_str, count_str, leaves) = match fields.as_slice() {
                [n, m, w, c, e, o] => (n, m, w, c, Some((e, o))),
                [n, m, w, c] => (n, m, w, c, None),
                _ => return Err(bad()),
            };
            let line_n: u32 = n_str.parse().map_err(|_| bad())?;
            let parse = |s: &str| s.parse::<u64>().map_err(|_| bad());
            let row = RowsetCount {
                mask: u32::from_str_radix(mask_hex, 16).map_err(|_| bad())?,
                weight: weight_str.parse().map_err(|_| bad())?,
                count: parse(count_str)?,
                leaves: match leaves {
                    Some((e, o)) => Some([parse(e)?, parse(o)?]),
                    None => None,
                },
            };
            match n {
                None => n = Some(line_n),
                Some(prev) if prev != line_n => {
                    return Err(format!("{path}:{}: n = {line_n}, but earlier lines have n = {prev}", i + 1));
                }
                Some(_) => {}
            }
            rows.push(row);
        }
        provenance.push((path.clone(), prov));
    }
    let n = n.ok_or("no row-set results found")?;
    Ok(Records { n, rows, provenance })
}

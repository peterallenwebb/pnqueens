//! Per-row-set results on disk.
//!
//! Q(n) is a sum over the feasible even-queen row sets R of
//! weight(R) * count(R), where count(R) = sum over C of E[R, C] * O[R̅, C̅].
//! Both the CPU and the GPU paths record every (R, weight, count) as it is
//! produced, so a finished run can be re-summed, compared row set by row set
//! with another run, or resumed by hand from what an interrupted run wrote.
//!
//! The file is CSV with one header line and one line per row set:
//!
//!     n,rowset_hex,weight,count
//!     8,0000000f,4,0
//!
//! `rowset_hex` is the row bitmask (bit r set = a queen on an even square of
//! row r), `weight` the size of its symmetry orbit, `count` the unweighted
//! count. Lines are written in the order results arrive and flushed when the
//! caller asks (after each GPU batch), so an interrupted run keeps the row
//! sets it finished. `n` is repeated on every line so files from different
//! machines can be concatenated and checked together.

use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// One row set's result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowsetCount {
    pub mask: u32,
    pub weight: u32,
    pub count: u64,
}

impl RowsetCount {
    /// This row set's contribution to Q(n).
    pub fn weighted(&self) -> u64 {
        u64::from(self.weight).checked_mul(self.count).expect("weighted count overflows u64")
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
    /// A log to a new file. Refuses to overwrite an existing file: a finished
    /// run's results should not be lost to a rerun with the same name.
    pub fn create(path: &Path, n: u32) -> io::Result<Self> {
        let file = File::create_new(path)?;
        let mut log = Self::new(n, Box::new(BufWriter::new(file)));
        writeln!(log.out, "n,rowset_hex,weight,count")?;
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
        writeln!(self.out, "{},{:08x},{},{}", self.n, r.mask, r.weight, r.count)?;
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

/// Reads row-set results from one or more files written by `RowsetLog`.
/// All lines must have the same n. Returns (n, rows).
pub fn read_rowset_files(paths: &[String]) -> Result<(u32, Vec<RowsetCount>), String> {
    let mut n: Option<u32> = None;
    let mut rows = Vec::new();
    for path in paths {
        let file = File::open(path).map_err(|e| format!("{path}: {e}"))?;
        for (i, line) in BufReader::new(file).lines().enumerate() {
            let line = line.map_err(|e| format!("{path}: {e}"))?;
            let line = line.trim();
            if line.is_empty() || line.starts_with("n,") {
                continue; // header (or a blank line from concatenation)
            }
            let bad = || format!("{path}:{}: cannot parse '{line}'", i + 1);
            let fields: Vec<&str> = line.split(',').collect();
            let [n_str, mask_hex, weight_str, count_str] = fields.as_slice() else {
                return Err(bad());
            };
            let line_n: u32 = n_str.parse().map_err(|_| bad())?;
            let row = RowsetCount {
                mask: u32::from_str_radix(mask_hex, 16).map_err(|_| bad())?,
                weight: weight_str.parse().map_err(|_| bad())?,
                count: count_str.parse().map_err(|_| bad())?,
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
    }
    let n = n.ok_or("no row-set results found")?;
    Ok((n, rows))
}

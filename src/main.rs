//! This program counts the number of solutions to the n-queens problem for a
//! given n. This initial comment establishes some notation and theory.
//!
//! Let E[R, C] be the number of solutions, including partial solutions, to the
//! n-queens problem, subject to these restrictions:
//!  - Queens are mutually non-attacking, as usual
//!  - Queens are only placed on squares with r + c = 0 mod 2 ("even squares")
//!  - Queens are placed on rows in set R, columns in set C, and only those
//!
//! Define O[R, C] in the same way, but restricting queens to squares with 
//! r + c = 1 mod 2 ("odd squares").
//!
//! Intuitively, E counts the solutions restricted to one checkerboard color,
//! and O the other color.
//!
//! Denote the complements of R and C by R̅ and C̅.
//! 
//! Note that a partial solution on even squares will never diagonally attack a
//! partial solution on odd squares, and that two such solutions combine to an
//! n-queens solutions exactly when one is chosen from the solutions counted by
//! E[R, C] and the other from the solutions counted by O[R̅, C̅]. Thus the total
//! number of solutions to the n-queens problem is given by the sum of over all
//! possible sets R and C of the product E[R, C] * O[R̅, C̅].
//!
//! Now we'll make some observations which reduce the work of calcuating the sum
//! by identifying many values of R and C which could never yield a solution.
//! 
//! Partition the squares into four parity classes according to the value
//! of (r mod 2, c mod 2). Under this partition, the classes represented by (0,0)
//! and (1,1) are the even squares definded above, and those represented by (0,1)
//! and (1,0) are the odd squares.
//!
//! For a full solution of the n-queens problem, consider the numbers k00, k01,
//! k10, k11 of queens on squares of the corresponding parity classes. Note
//! that k00 + k01 counts the number of even rows on the board, while k01 + k11
//! counts the number of odd columns. Therefore k00 + k10 = ceil(n/2) and
//! k11 + k10 = floor(n/2). It follows that k00 - k11 = n mod 2. By a similar
//! argument, k01 = k10.
//!
//! This means we do not have to check all possible row and column sets, but only
//! those compatible with the constraints on parity classes.
//! 
//! Note the identities:
//! - |R ∩ even rows| = k00
//! - |R ∩ odd rows| = k11
//! - |R̅ ∩ even rows| = k01
//! - |R̅ ∩ odd rows| = k10
//!
//! For even n, flipping the board horizontally maps black squares to white, and
//! so square queens to odd square queens, and solutions with even queens on R
//! to solutions with odd queens on R, which are just the even solutions on R̅.
//!
//! Note also:
//! E[R, C] = E[R, rev C] for odd n, since the mirror preserves color (exploit inside sum)
//! E[R, C] = O[R, rev C] for even n, by horizontal reflection, which exchanges colors

#[cfg(feature = "cuda")]
mod gpu;
mod rowset_log;
mod solution_table;

use rayon::prelude::*;
use rowset_log::{read_rowset_files, RowsetCount, RowsetLog};
use solution_table::{split_columns, SolutionTable};
use std::path::PathBuf;

fn main() {
    let args = parse_args();
    if let Some(files) = &args.sum_files {
        retabulate(files);
        return;
    }
    let n = args.n;
    let out = args.out.clone().unwrap_or_else(|| PathBuf::from(format!("rowsets_{n}.csv")));
    let mut log = RowsetLog::create(&out, n).unwrap_or_else(|e| fail(&format!("cannot create {}: {e}", out.display())));
    if args.cuda {
        count_queens_on_gpu(&args, &mut log);
    } else {
        count_queens(n, &mut log);
    }
    log.flush().unwrap_or_else(|e| fail(&format!("cannot write {}: {e}", out.display())));
    println!("{} row-set counts written to {}", log.written(), out.display());
    report(n, log.total(), true);
}

/// Re-tabulate Q(n) from row-set files and check that they cover
/// every feasible row set exactly once with the right weight.
fn retabulate(files: &[String]) {
    let (n, rows) = read_rowset_files(files).unwrap_or_else(|e| fail(&e));
    let expected = get_feasible_even_row_combos(n);
    let mut seen: std::collections::HashMap<u32, u32> = expected.iter().map(|rc| (rc.mask, rc.weight as u32)).collect();
    let mut total: u128 = 0;
    for row in &rows {
        match seen.remove(&row.mask) {
            Some(weight) if weight == row.weight => {}
            Some(weight) => fail(&format!("row set {:08x} has weight {}, expected {weight}", row.mask, row.weight)),
            None if expected.iter().any(|rc| rc.mask == row.mask) => fail(&format!("row set {:08x} appears twice", row.mask)),
            None => fail(&format!("row set {:08x} is not a feasible row set for n = {n}", row.mask)),
        }
        total += u128::from(row.weighted());
    }
    println!("{} of {} row sets present", rows.len(), expected.len());
    report(n, total, seen.is_empty());
}

/// Prints the result, labelled against the known value when the count is
/// complete and n is in the known range.
fn report(n: u32, total: u128, complete: bool) {
    let status = match (complete, known(n)) {
        (false, _) => " PARTIAL",
        (true, Some(k)) if k == total => " ok",
        (true, Some(_)) => " MISMATCH",
        (true, None) => "",
    };
    println!("Q({n}): {total}{status}");
}

fn fail(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
fn count_queens_on_gpu(args: &Args, log: &mut RowsetLog) {
    let opts = gpu::GpuOptions {
        device: args.device,
        prefix_depth: args.prefix_depth.unwrap_or_else(|| gpu::GpuOptions::default_prefix_depth(args.n)),
        table_budget_bytes: args.table_mem_mib << 20,
    };
    gpu::count_queens_gpu(args.n, &opts, log).unwrap_or_else(|e| fail(&e));
}

#[cfg(not(feature = "cuda"))]
fn count_queens_on_gpu(_args: &Args, _log: &mut RowsetLog) {
    fail("this binary was built without the cuda feature (cargo build --release --features cuda)");
}

/// Q(n) for 1 <= n <= 27, OEIS A000170. Used to label results and in tests.
const KNOWN: [u128; 27] = [
    1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184, 14772512, 95815104,
    666090624, 4968057848, 39029188884, 314666222712, 2691008701644, 24233937684440,
    227514171973736, 2207893435808352, 22317699616364044, 234907967154122528,
];

fn known(n: u32) -> Option<u128> {
    KNOWN.get(n as usize - 1).copied()
}

// Q(n): the sum over feasible even-queen row sets R of E[R, C] * O[R̅, C̅],
// weighted by the size of each R's symmetry orbit. Every row set's count is
// recorded in `log`, which keeps the weighted total.
fn count_queens(n: u32, log: &mut RowsetLog) {
    let ercs = get_feasible_even_row_combos(n);

    println!("PREPARED {}", ercs.len());

    // Each row combo is an independent search, so count them in parallel;
    // rayon's work stealing absorbs the large variation in per-combo cost.
    // `collect` keeps the input order, so the log is in row-set order.
    let results: Vec<RowsetCount> = ercs
        .par_iter()
        .map(|erc| RowsetCount { mask: erc.mask, weight: erc.weight as u32, count: count_solutions(n, erc) })
        .collect();
    for r in &results {
        log.record(r).unwrap_or_else(|e| fail(&format!("cannot write row-set counts: {e}")));
    }
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct Args {
    /// Board size. To simplify bitmask arithmetic, we enforce 1 <= n <= 31.
    n: u32,
    /// Run on a CUDA GPU instead of the CPU.
    cuda: bool,
    /// CUDA device index (--device).
    device: usize,
    /// Ticket prefix depth (--prefix); None means the GPU path's default for n.
    prefix_depth: Option<u32>,
    /// Device memory for one batch's tables, in MiB (--table-mem).
    table_mem_mib: usize,
    /// Where to write the per-row-set counts (--out); None means rowsets_<n>.csv.
    out: Option<PathBuf>,
    /// `--sum FILE...`: re-tabulate from row-set files instead of counting.
    sum_files: Option<Vec<String>>,
}

// Parse the command line. Exit with a usage message on any problem.
fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let usage = || -> ! {
        eprintln!(
            "usage: {0} <n> [--cuda] [--device I] [--prefix P] [--table-mem MiB] [--out FILE]\n\
             \x20      {0} --sum FILE...\n\
             \x20 1 <= n <= 31\n\
             \x20 --cuda        Run on a CUDA GPU (build with --features cuda)\n\
             \x20 --device I    (CUDA mode only) CUDA device index (default 0)\n\
             \x20 --prefix P    (CUDA mode only) Rows fixed per GPU thread (default 3, +1 at n >= 24, +1 at n >= 28)\n\
             \x20 --table-mem M (CUDA mode only) Device memory for tables per batch, MiB (default 8192)\n\
             \x20 --out FILE    Per-row-set counts, CSV (default rowsets_<n>.csv; never overwritten)\n\
             \x20 --sum FILE... Re-tabulate Q(n) from row-set files and check they are complete",
            argv.first().map_or("pnqueens", String::as_str)
        );
        std::process::exit(2);
    };
    let mut args = Args { n: 0, cuda: false, device: 0, prefix_depth: None, table_mem_mib: 8192, out: None, sum_files: None };
    let mut n = None;
    let mut rest = argv.iter().skip(1);
    while let Some(arg) = rest.next() {
        let mut value = |what: &str| -> String { rest.next().cloned().unwrap_or_else(|| { eprintln!("{what} needs a value"); usage() }) };
        match arg.as_str() {
            "--cuda" => args.cuda = true,
            "--device" => args.device = value("--device").parse().unwrap_or_else(|_| usage()),
            "--prefix" => args.prefix_depth = Some(value("--prefix").parse().unwrap_or_else(|_| usage())),
            "--table-mem" => args.table_mem_mib = value("--table-mem").parse().unwrap_or_else(|_| usage()),
            "--out" => args.out = Some(PathBuf::from(value("--out"))),
            "--sum" => {
                // Everything after --sum is a file name.
                let files: Vec<String> = rest.by_ref().cloned().collect();
                if files.is_empty() {
                    usage();
                }
                args.sum_files = Some(files);
            }
            other => match other.parse::<u32>() {
                Ok(v) if (1..=31).contains(&v) && n.is_none() => n = Some(v),
                _ => usage(),
            },
        }
    }
    if args.sum_files.is_some() {
        if n.is_some() || args.cuda || args.out.is_some() {
            usage();
        }
        return args;
    }
    args.n = n.unwrap_or_else(|| usage());
    args
}

struct RowCombo {
    pub mask: u32,
    pub weight: i32,
}

fn get_feasible_even_row_combos(n: u32) -> Vec<RowCombo> {
    let mut feasible_combos: Vec<RowCombo> = Vec::new();
    let mut rowset: u32 = 0;
    while rowset < (1 << n) {
        // To be feasible, the count of rows with r = 0 mod 2, less the
        // count of rows with r = 1 mod 2, must be 0 for even n, and 1 for odd.
        let k00 = (rowset & 0x5555_5555).count_ones() as i32;
        let k11 = (rowset & 0xAAAA_AAAA).count_ones() as i32;
        if k00 - k11 == (n % 2) as i32 {
            // We also cull combos that are symmetric under reflection if they are
            // lexically greater than their reflection. Marking the latter as
            // double-weighted to compensate.
            let reflection = reflect(n, rowset);

            if rowset <= reflection {
                if n % 2 == 1 {
                    feasible_combos.push(RowCombo { mask: rowset, weight: if rowset == reflection { 1 } else { 2 } });
                } else {
                    let rc = reflect(n, complement(n, rowset));
                    if rowset <= rc {
                        let weight = if rowset == reflection || rowset == rc { 2 } else { 4 };
                        feasible_combos.push(RowCombo { mask: rowset, weight });
                    }
                }
            }
        }

        rowset += 1;
    }
    return feasible_combos;
}

fn count_solutions(n: u32, even_queen_rows: &RowCombo) -> u64 {
    match get_erc(n, even_queen_rows.mask) {
        None => 0,
        Some(erc) => backtrack_odd_solutions(n, &erc, complement(n, even_queen_rows.mask)),
    }
}

// Computes the table E[R, C] for all C, indexed by column set. Returns None
// when there are no even-square placements on R at all (every entry would be
// zero), so the caller can skip the odd-square search.
fn get_erc(n: u32, row_bitmask: u32) -> Option<SolutionTable> {
    let mut table = SolutionTable::for_rows(n, row_bitmask, 0);
    let leaves = enumerate(n, row_bitmask, 0, |cols| {
        let (even, odd) = split_columns(cols);
        table.increment_dense(even, odd);
    });
    (leaves > 0).then_some(table)
}

// Complement an n-bit bitmask
fn complement(n: u32, x: u32) -> u32 {
    return !x & ((1 << n) - 1);
}

// Reflect an n-bit bitmask
fn reflect(n: u32, x: u32) -> u32 {
    return x.reverse_bits() >> (32 - n);
}

// Run a conventional backtracking search to find all partial n-queens solutions
// which are counted by O[R̅, C̅]. When one is found, add the corresponding value
// of E[R, C] to the count. Return the total count. The parameter rcomp is R̅, the
// available row mask for odd-square queens.
fn backtrack_odd_solutions(n: u32, erc: &SolutionTable, rcomp: u32) -> u64 {
    // Same enumeration as get_erc, but on odd squares; a leaf with column set
    // C̅ contributes E[R, C] where C is the complement of C̅.
    let mut total = 0u64;
    enumerate(n, rcomp, 1, |cols| {
        let (even, odd) = split_columns(cols);
        total += erc.get_complement_dense(even, odd) as u64;
    });
    total
}

// Enumerates every placement of one queen per row of `rows`, all on squares of
// the given parity and mutually non-attacking, calling `leaf(cols)` with the
// column bitmask of each. Returns the number of leaves.
//
// The search is an explicit-stack depth-first search over the rows of R only.
// Level t holds the row r_t; `pmask[t]` is the colour-compatible column mask
// for that row and `gap[t]` the distance to the next row of R. `cols`, `ld`,
// `rd` are the attacked columns and the two diagonal families as seen from the
// current row, so stepping to the next row of R shifts the diagonals by the
// gap. `st_free[t]` holds the candidates not yet tried at level t.
fn enumerate(n: u32, rows: u32, parity: u32, mut leaf: impl FnMut(u32)) -> u64 {
    let full: u32 = (1 << n) - 1;

    let mut row = [0u32; 32];
    let mut l = 0usize;
    let mut m = rows;
    while m != 0 {
        row[l] = m.trailing_zeros();
        m &= m - 1;
        l += 1;
    }
    if l == 0 {
        // No rows: the single empty placement.
        leaf(0);
        return 1;
    }

    let mut pmask = [0u32; 32];
    let mut gap = [0u32; 32];
    for t in 0..l {
        let r = row[t];
        // Colour-compatible columns: c ≡ r (mod 2) on even squares, c ≢ r on odd.
        pmask[t] = if (r & 1) == parity { 0x5555_5555 } else { 0xAAAA_AAAA } & full;
        gap[t] = if t + 1 < l { row[t + 1] - r } else { 0 };
    }

    let mut st_free = [0u32; 32];
    let mut st_cols = [0u32; 32];
    let mut st_ld = [0u32; 32];
    let mut st_rd = [0u32; 32];
    st_free[0] = pmask[0];
    let last = l - 1;
    let mut leaves = 0u64;
    let mut t = 0usize;
    loop {
        let free = st_free[t];
        if free == 0 {
            if t == 0 {
                break;
            }
            t -= 1;
            continue;
        }
        let bit = free & free.wrapping_neg();
        st_free[t] = free ^ bit;
        let cols = st_cols[t] | bit;
        if t == last {
            leaf(cols);
            leaves += 1;
            continue;
        }
        let sh = gap[t];
        let ld = (st_ld[t] | bit) << sh;
        let rd = (st_rd[t] | bit) >> sh;
        t += 1;
        st_cols[t] = cols;
        st_ld[t] = ld;
        st_rd[t] = rd;
        st_free[t] = pmask[t] & !(cols | ld | rd);
    }
    leaves
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_queen_counts() {
        // Q(1) .. Q(15); larger n take too long in a debug build.
        for (i, &q) in KNOWN[..15].iter().enumerate() {
            let n = i as u32 + 1;
            let mut log = RowsetLog::discard(n);
            count_queens(n, &mut log);
            assert_eq!(log.total(), q, "Q({n})");
        }
    }
}

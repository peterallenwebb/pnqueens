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
//!
//! The first of these is used inside the sum for odd n. The mirror c -> n-1-c
//! fixes R and preserves colour, so it pairs each odd-square placement on R̅
//! (column set C̅) with its mirror image (column set rev C̅), and both look up
//! the same value E[R, C] = E[R, rev C]. The odd-square search therefore
//! places the queen of its first row only in columns left of centre, counting
//! each placement twice, plus the centre column, counted once. See
//! `backtrack_odd_solutions` and `mirror_columns`.

#[cfg(any(feature = "cuda", test))]
mod gpu;
mod rowset_log;
mod solution_table;
mod symmetric;
mod work;

use rayon::prelude::*;
use rowset_log::{read_rowset_files, RowsetCount, RowsetLog};
use sha2::{Digest, Sha256};
use solution_table::{split_columns, SolutionTable};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;
use work::{feasible_row_sets, fingerprint, work_items, RowSet, Selection};

/// Exit status: 0 when every check that applied passed, 1 when one failed
/// (`fail` and the checks below), 3 when records were incomplete.
const EXIT_FAILED: i32 = 1;
const EXIT_PARTIAL: i32 = 3;

fn main() {
    let args = parse_args();
    let code = match &args.mode {
        Mode::Count => count(&args),
        Mode::Version => {
            let id = Identity::of_this_run();
            println!("pnqueens commit {} binary {}", id.commit, id.binary_sha256);
            0
        }
        Mode::Items => {
            print_items(args.n);
            0
        }
        Mode::RowSet(mask) => {
            audit_rowset(args.n, *mask);
            0
        }
        Mode::Mod8(total) => {
            if mod8_check(args.n, *total) { 0 } else { EXIT_FAILED }
        }
        Mode::Sum(files) => retabulate(files),
    };
    std::process::exit(code);
}

/// Q(n), or one phase of it: the selected row sets are counted, on the GPU
/// or the CPU, into a record file headed by the run's provenance, and a JSON
/// manifest describes the run. A complete count is checked against the
/// known value and the mod-8 identity; the exit status says whether it passed.
fn count(args: &Args) -> i32 {
    let n = args.n;
    let start = Instant::now();
    let items = work_items(n);
    let workmap = fingerprint(n, &items);
    let selected: Vec<RowSet> = args.selection.pick(&items).copied().collect();
    let out = args.out.clone().unwrap_or_else(|| default_output(n, args.selection));
    let manifest_path = args.manifest.clone().unwrap_or_else(|| out.with_extension("json"));
    let identity = Identity::of_this_run();
    let provenance = format!(
        "pnqueens n={n} stride={} phase={} workmap={workmap:016x} items={} device={} commit={} binary={}",
        args.selection.stride,
        args.selection.phase,
        selected.len(),
        if args.cuda { "cuda" } else { "cpu" },
        identity.commit,
        identity.binary_sha256
    );
    let mut log = RowsetLog::create(&out, n, &provenance).unwrap_or_else(|e| fail(&format!("cannot create {}: {e}", out.display())));
    println!("PREPARED {} of {} row sets (stride {}, phase {}, work map {workmap:016x})", selected.len(), items.len(), args.selection.stride, args.selection.phase);
    let device = if args.cuda { count_on_gpu(args, &selected, &mut log) } else { count_on_cpu(n, &selected, &mut log) };
    log.flush().unwrap_or_else(|e| fail(&format!("cannot write {}: {e}", out.display())));
    println!("{} row-set counts written to {}", log.written(), out.display());
    let passed = if args.selection.is_all() {
        report(n, log.total()) & mod8_check(n, Some(log.total()))
    } else {
        println!("Q({n}) phase {} of {}: {} PARTIAL", args.selection.phase, args.selection.stride, log.total());
        true
    };
    let manifest = Manifest {
        n,
        selection: args.selection,
        workmap,
        items: selected.len(),
        identity,
        device,
        options: format!(
            "{{\"prefix\": {}, \"blocks_per_sm\": {}, \"table_mem_mib\": {}}}",
            json_opt(args.prefix_depth),
            json_opt(args.blocks_per_sm),
            args.table_mem_mib
        ),
        wall_s: start.elapsed().as_secs_f64(),
        row_sets: log.written(),
        weighted_total: log.total(),
    };
    std::fs::write(&manifest_path, manifest.to_json()).unwrap_or_else(|e| fail(&format!("cannot write {}: {e}", manifest_path.display())));
    println!("manifest written to {}", manifest_path.display());
    if passed { 0 } else { EXIT_FAILED }
}

/// `rowsets_<n>.csv`, or `rowsets_<n>_s<stride>_p<phase>.csv` for a phase.
fn default_output(n: u32, selection: Selection) -> PathBuf {
    if selection.is_all() {
        PathBuf::from(format!("rowsets_{n}.csv"))
    } else {
        PathBuf::from(format!("rowsets_{n}_s{}_p{}.csv", selection.stride, selection.phase))
    }
}

fn fail(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1);
}

// ---- provenance and the manifest

/// What identifies the program and the machine that produced a record.
struct Identity {
    /// The source commit the binary was built from (build.rs), "-dirty" if
    /// the tree had uncommitted changes.
    commit: String,
    /// SHA-256 of the running executable.
    binary_sha256: String,
    host: String,
}

impl Identity {
    fn of_this_run() -> Self {
        let binary_sha256 = std::env::current_exe()
            .and_then(std::fs::read)
            .map(|bytes| format!("{:x}", Sha256::digest(&bytes)))
            .unwrap_or_else(|_| "unknown".into());
        let host = std::process::Command::new("hostname")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|| "unknown".into());
        Identity { commit: env!("PNQUEENS_COMMIT").to_string(), binary_sha256, host }
    }
}

/// What the run has to say about the device it ran on, for the manifest.
struct Device {
    name: &'static str,
    /// The GPU's name, if any.
    gpu: Option<String>,
    /// The device path's statistics as a JSON value ("null" for the CPU).
    stats: String,
}

/// The JSON description of a run, written next to its record file.
struct Manifest {
    n: u32,
    selection: Selection,
    workmap: u64,
    items: usize,
    identity: Identity,
    device: Device,
    options: String,
    wall_s: f64,
    row_sets: usize,
    weighted_total: u128,
}

impl Manifest {
    fn to_json(&self) -> String {
        format!(
            "{{\n  \"program\": \"pnqueens\",\n  \"n\": {},\n  \"stride\": {},\n  \"phase\": {},\n  \"workmap\": \"{:016x}\",\n  \"items\": {},\n  \
             \"commit\": {},\n  \"binary_sha256\": {},\n  \"host\": {},\n  \"device\": \"{}\",\n  \"gpu\": {},\n  \"options\": {},\n  \
             \"wall_s\": {:.3},\n  \"row_sets\": {},\n  \"weighted_total\": \"{}\",\n  \"stats\": {}\n}}\n",
            self.n,
            self.selection.stride,
            self.selection.phase,
            self.workmap,
            self.items,
            json_string(&self.identity.commit),
            json_string(&self.identity.binary_sha256),
            json_string(&self.identity.host),
            self.device.name,
            self.device.gpu.as_deref().map_or("null".to_string(), json_string),
            self.options,
            self.wall_s,
            self.row_sets,
            self.weighted_total,
            self.device.stats
        )
    }
}

fn json_string(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_opt(v: Option<u32>) -> String {
    v.map_or("null".to_string(), |v| v.to_string())
}

// ---- the other modes

/// The size of the work and its fingerprint, without counting anything.
fn print_items(n: u32) {
    let items = work_items(n);
    println!("n={n} items={} workmap={:016x}", items.len(), fingerprint(n, &items));
}

/// One row set's count on the CPU, printed as the record line it should
/// match: the audit of a single line of a run's records, needing no GPU.
fn audit_rowset(n: u32, mask: u32) {
    let Some(row_set) = feasible_row_sets(n).into_iter().find(|r| r.mask == mask) else {
        fail(&format!("{mask:08x} is not a representative feasible row set for n = {n} (see feasible_row_sets)"));
    };
    let start = Instant::now();
    let count = count_solutions(n, &row_set);
    println!("{n},{mask:08x},{},{count}", row_set.weight);
    eprintln!("({:.1}s on the CPU)", start.elapsed().as_secs_f64());
}

/// The number of solutions fixed by the 180° rotation, and, given a total,
/// whether Q ≡ S2 (mod 8) holds for it (see symmetric.rs). False on FAIL.
fn mod8_check(n: u32, total: Option<u128>) -> bool {
    let start = Instant::now();
    let s2 = symmetric::rotation_fixed_solutions(n);
    match total {
        Some(q) => {
            let pass = q % 8 == u128::from(s2) % 8;
            println!("mod-8 check: {} (Q ≡ {}, S2 = {s2} ≡ {} mod 8, {:.1}s)", if pass { "PASS" } else { "FAIL" }, q % 8, s2 % 8, start.elapsed().as_secs_f64());
            pass
        }
        None => {
            println!("S2({n}) = {s2} ({:.1}s)", start.elapsed().as_secs_f64());
            true
        }
    }
}

/// Re-tabulates Q(n) from record files and checks them: one work map across
/// the files, the phases of a stride present, duplicate row sets in
/// agreement, every feasible row set exactly once with the right weight, the
/// total against the published value, and the mod-8 identity. Returns the
/// exit status: 0 when complete and every check passed, EXIT_PARTIAL when
/// incomplete; a failed check exits EXIT_FAILED.
fn retabulate(files: &[String]) -> i32 {
    let records = read_rowset_files(files).unwrap_or_else(|e| fail(&e));
    let n = records.n;

    // Provenance: every file that carries a work map must carry the same one.
    let workmaps: std::collections::BTreeSet<&str> = records.provenance.iter().filter_map(|(_, p)| p.get("workmap").map(String::as_str)).collect();
    if workmaps.len() > 1 {
        fail(&format!("the files were cut from different work maps: {}", workmaps.into_iter().collect::<Vec<_>>().join(", ")));
    }
    let this_workmap = format!("{:016x}", fingerprint(n, &work_items(n)));
    if let Some(&recorded) = workmaps.iter().next()
        && recorded != this_workmap
    {
        println!("note: the records' work map {recorded} is not this binary's {this_workmap}; the row sets are checked regardless");
    }
    // Phases: with one stride across the files, which phases are present.
    let strides: std::collections::BTreeSet<u32> = records.provenance.iter().filter_map(|(_, p)| p.get("stride")?.parse().ok()).collect();
    if let (Some(&stride), 1) = (strides.iter().next(), strides.len()) {
        let mut phases: Vec<u32> = records.provenance.iter().filter_map(|(_, p)| p.get("phase")?.parse().ok()).collect();
        phases.sort_unstable();
        phases.dedup();
        let missing: Vec<u32> = (0..stride).filter(|p| phases.binary_search(p).is_err()).collect();
        println!("phases: {} of {stride} present{}", phases.len(), if missing.is_empty() { String::new() } else { format!(", missing {}", summarize(&missing)) });
    } else if strides.len() > 1 {
        fail(&format!("the files have different strides: {strides:?}"));
    }

    // Row sets: each feasible one exactly once, or several times in agreement.
    let expected: BTreeMap<u32, u32> = feasible_row_sets(n).iter().map(|r| (r.mask, r.weight)).collect();
    let mut seen: BTreeMap<u32, RowsetCount> = BTreeMap::new();
    let mut verified = 0usize;
    for row in &records.rows {
        match expected.get(&row.mask) {
            Some(&weight) if weight == row.weight => {}
            Some(weight) => fail(&format!("row set {:08x} has weight {}, expected {weight}", row.mask, row.weight)),
            None => fail(&format!("row set {:08x} is not a representative feasible row set for n = {n}", row.mask)),
        }
        match seen.insert(row.mask, *row) {
            None => {}
            Some(earlier) if earlier == *row => verified += 1,
            Some(earlier) => fail(&format!("CONFLICT: row set {:08x} recorded as {} and as {}", row.mask, earlier.count, row.count)),
        }
    }
    let total: u128 = seen.values().map(|r| u128::from(r.weighted())).sum();
    let complete = seen.len() == expected.len();
    println!(
        "{} of {} row sets present in {} files ({} recorded more than once, all in agreement)",
        seen.len(),
        expected.len(),
        files.len(),
        verified
    );
    if !complete {
        println!("Q({n}): {total} PARTIAL");
        return EXIT_PARTIAL;
    }
    if report(n, total) & mod8_check(n, Some(total)) { 0 } else { EXIT_FAILED }
}

/// "3, 7-9, 12" for a sorted list.
fn summarize(xs: &[u32]) -> String {
    let mut parts = Vec::new();
    let mut i = 0;
    while i < xs.len() {
        let mut j = i;
        while j + 1 < xs.len() && xs[j + 1] == xs[j] + 1 {
            j += 1;
        }
        parts.push(if j > i { format!("{}-{}", xs[i], xs[j]) } else { xs[i].to_string() });
        i = j + 1;
    }
    parts.join(", ")
}

/// Prints a complete count, labelled against the published value or the
/// reference project's value when there is one. False when it differs.
fn report(n: u32, total: u128) -> bool {
    let (status, passed) = match (known(n), reference(n)) {
        (Some(k), _) if k == total => (" ok", true),
        (Some(_), _) => (" MISMATCH", false),
        (None, Some(r)) if r == total => (" matches the reference project's value", true),
        (None, Some(_)) => (" DIFFERS from the reference project's value", false),
        (None, None) => (" (no reference value)", true),
    };
    println!("Q({n}): {total}{status}");
    passed
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

/// Values computed by another program (the reference project's balanced
/// checkerboard join on a fleet of RTX 5090s, 2026-09-11), not yet published.
fn reference(n: u32) -> Option<u128> {
    (n == 28).then_some(2_564_269_152_506_723_832)
}

// ---- counting

/// The selected row sets counted on the CPU, each an independent search,
/// in parallel. `collect` keeps the order, so the records are in item order.
fn count_on_cpu(n: u32, items: &[RowSet], log: &mut RowsetLog) -> Device {
    let results: Vec<RowsetCount> = items
        .par_iter()
        .map(|item| RowsetCount { mask: item.mask, weight: item.weight, count: count_solutions(n, item) })
        .collect();
    for r in &results {
        log.record(r).unwrap_or_else(|e| fail(&format!("cannot write row-set counts: {e}")));
    }
    Device { name: "cpu", gpu: None, stats: "null".into() }
}

#[cfg(feature = "cuda")]
fn count_on_gpu(args: &Args, items: &[RowSet], log: &mut RowsetLog) -> Device {
    let opts = gpu::GpuOptions {
        device: args.device,
        prefix_depth: args.prefix_depth,
        table_budget_bytes: args.table_mem_mib << 20,
        blocks_per_sm: args.blocks_per_sm,
    };
    let stats = gpu::count_queens_gpu(args.n, &opts, items, log).unwrap_or_else(|e| fail(&e));
    Device { name: "cuda", gpu: Some(stats.gpu.clone()), stats: stats.to_json() }
}

#[cfg(not(feature = "cuda"))]
fn count_on_gpu(_args: &Args, _items: &[RowSet], _log: &mut RowsetLog) -> Device {
    fail("this binary was built without the cuda feature (cargo build --release --features cuda)");
}

/// One row set's count: sum over C of E[R, C] * O[R̅, C̅], on the CPU.
fn count_solutions(n: u32, row_set: &RowSet) -> u64 {
    match get_erc(n, row_set.mask) {
        None => 0,
        Some(erc) => backtrack_odd_solutions(n, &erc, work::complement(n, row_set.mask)),
    }
}

// Computes the table E[R, C] for all C, indexed by column set. Returns None
// when there are no even-square placements on R at all (every entry would be
// zero), so the caller can skip the odd-square search.
fn get_erc(n: u32, row_bitmask: u32) -> Option<SolutionTable> {
    let mut table = SolutionTable::for_rows(n, row_bitmask, 0);
    let leaves = enumerate(n, row_bitmask, 0, u32::MAX, |cols| {
        let (even, odd) = split_columns(cols);
        table.increment_dense(even, odd);
    });
    (leaves > 0).then_some(table)
}

// Run a conventional backtracking search to find all partial n-queens solutions
// which are counted by O[R̅, C̅]. When one is found, add the corresponding value
// of E[R, C] to the count. Return the total count. The parameter rcomp is R̅, the
// available row mask for odd-square queens.
fn backtrack_odd_solutions(n: u32, erc: &SolutionTable, rcomp: u32) -> u64 {
    // Same enumeration as get_erc, but on odd squares; a leaf with column set
    // C̅ contributes E[R, C] where C is the complement of C̅.
    let count = |first_row_columns: u32| {
        let mut total = 0u64;
        enumerate(n, rcomp, 1, first_row_columns, |cols| {
            let (even, odd) = split_columns(cols);
            total += erc.get_complement_dense(even, odd) as u64;
        });
        total
    };
    match mirror_columns(n, rcomp) {
        // Odd n (see the module docs): placements with the first queen left
        // of centre stand for themselves and their mirror images.
        Some((left, centre)) => 2 * count(left) + count(centre),
        None => count(u32::MAX),
    }
}

// The column masks the odd-square search restricts its first row to, when the
// mirror symmetry applies: (columns left of centre, the centre column). It
// applies for odd n, where the mirror c -> n-1-c preserves square colour;
// for even n it swaps colours and cannot be used this way. It also needs a
// first row to restrict: with no rows there is one empty placement, which is
// its own mirror image and must be counted once.
fn mirror_columns(n: u32, rows: u32) -> Option<(u32, u32)> {
    if n.is_multiple_of(2) || rows == 0 {
        return None;
    }
    let mid = (n - 1) / 2;
    Some(((1 << mid) - 1, 1 << mid))
}

// Enumerates every placement of one queen per row of `rows`, all on squares of
// the given parity and mutually non-attacking, calling `leaf(cols)` with the
// column bitmask of each. Returns the number of leaves. The queen of the first
// row is restricted to `first_row_columns` (pass u32::MAX for no restriction).
//
// The search is an explicit-stack depth-first search over the rows of R only.
// Level t holds the row r_t; `pmask[t]` is the colour-compatible column mask
// for that row and `gap[t]` the distance to the next row of R. `cols`, `ld`,
// `rd` are the attacked columns and the two diagonal families as seen from the
// current row, so stepping to the next row of R shifts the diagonals by the
// gap. `st_free[t]` holds the candidates not yet tried at level t.
fn enumerate(n: u32, rows: u32, parity: u32, first_row_columns: u32, mut leaf: impl FnMut(u32)) -> u64 {
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
    st_free[0] = pmask[0] & first_row_columns;
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

// ---- the command line

enum Mode {
    /// Count the selected row sets (the default).
    Count,
    /// `--version`: the commit the binary was built from and its SHA-256.
    Version,
    /// `--items`: the item count and work map fingerprint.
    Items,
    /// `--rowset HEX`: one row set on the CPU.
    RowSet(u32),
    /// `--mod8 [TOTAL]`: the rotation-fixed count, and the check of a total.
    Mod8(Option<u128>),
    /// `--sum FILE...`: re-tabulate and check record files.
    Sum(Vec<String>),
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct Args {
    /// Board size. To simplify bitmask arithmetic, we enforce 1 <= n <= 31.
    n: u32,
    mode: Mode,
    /// Run on a CUDA GPU instead of the CPU.
    cuda: bool,
    /// CUDA device index (--device).
    device: usize,
    /// Ticket prefix depth (--prefix); None means the GPU path's default for n.
    prefix_depth: Option<u32>,
    /// Device memory for one batch's tables, in MiB (--table-mem).
    table_mem_mib: usize,
    /// Resident blocks per multiprocessor (--blocks-per-sm); None means as many as fit.
    blocks_per_sm: Option<u32>,
    /// The phase of the work to count (--stride, --phase).
    selection: Selection,
    /// Where to write the per-row-set counts (--out).
    out: Option<PathBuf>,
    /// Where to write the run's manifest (--manifest); None means next to the counts.
    manifest: Option<PathBuf>,
}

// Parse the command line. Exit with a usage message on any problem.
fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let usage = || -> ! {
        eprintln!(
            "usage: {0} <n> [--cuda] [--device I] [--prefix P] [--table-mem MiB] [--blocks-per-sm B]\n\
             \x20                 [--stride S --phase P] [--out FILE] [--manifest FILE]\n\
             \x20      {0} <n> --items\n\
             \x20      {0} <n> --rowset HEX\n\
             \x20      {0} <n> --mod8 [TOTAL]\n\
             \x20      {0} --sum FILE...\n\
             \x20      {0} --version\n\
             \x20 1 <= n <= 31\n\
             \x20 --cuda        Run on a CUDA GPU (build with --features cuda)\n\
             \x20 --device I    (CUDA mode only) CUDA device index (default 0)\n\
             \x20 --prefix P    (CUDA mode only) Rows fixed per GPU ticket (default: the least that keeps the tables in flight within L2)\n\
             \x20 --table-mem M (CUDA mode only) Device memory for tables per batch, MiB (default 8192)\n\
             \x20 --blocks-per-sm B (CUDA mode only) Resident blocks per multiprocessor (default: as many as fit; fewer only for measurements)\n\
             \x20 --stride S --phase P  Count only the row sets at positions P, P+S, P+2S, ... of the work list (phase P of S)\n\
             \x20 --out FILE    Per-row-set counts, CSV (default rowsets_<n>[_s<S>_p<P>].csv; never overwritten)\n\
             \x20 --manifest FILE  The run's manifest, JSON (default: the counts file with .json)\n\
             \x20 --items       Print the number of work items and the work map fingerprint\n\
             \x20 --version     Print the source commit the binary was built from and its SHA-256\n\
             \x20 --rowset HEX  Count one row set on the CPU and print its record line (an audit of one line of the records)\n\
             \x20 --mod8 [TOTAL]  Count the 180°-symmetric solutions S2; with TOTAL, check TOTAL ≡ S2 (mod 8)\n\
             \x20 --sum FILE... Re-tabulate Q(n) from record files and check them (coverage, weights, phases, duplicates, mod 8)",
            argv.first().map_or("pnqueens", String::as_str)
        );
        std::process::exit(2);
    };
    let mut args = Args {
        n: 0,
        mode: Mode::Count,
        cuda: false,
        device: 0,
        prefix_depth: None,
        table_mem_mib: 8192,
        blocks_per_sm: None,
        selection: Selection::ALL,
        out: None,
        manifest: None,
    };
    let mut n = None;
    let mut modes = 0;
    let mut rest = argv.iter().skip(1).peekable();
    while let Some(arg) = rest.next() {
        let mut value = |what: &str| -> String { rest.next().cloned().unwrap_or_else(|| { eprintln!("{what} needs a value"); usage() }) };
        match arg.as_str() {
            "--cuda" => args.cuda = true,
            "--device" => args.device = value("--device").parse().unwrap_or_else(|_| usage()),
            "--prefix" => args.prefix_depth = Some(value("--prefix").parse().unwrap_or_else(|_| usage())),
            "--table-mem" => args.table_mem_mib = value("--table-mem").parse().unwrap_or_else(|_| usage()),
            "--blocks-per-sm" => args.blocks_per_sm = Some(value("--blocks-per-sm").parse().unwrap_or_else(|_| usage())),
            "--stride" => args.selection.stride = value("--stride").parse().unwrap_or_else(|_| usage()),
            "--phase" => args.selection.phase = value("--phase").parse().unwrap_or_else(|_| usage()),
            "--out" => args.out = Some(PathBuf::from(value("--out"))),
            "--manifest" => args.manifest = Some(PathBuf::from(value("--manifest"))),
            "--items" => {
                args.mode = Mode::Items;
                modes += 1;
            }
            "--version" => {
                args.mode = Mode::Version;
                modes += 1;
            }
            "--rowset" => {
                args.mode = Mode::RowSet(u32::from_str_radix(&value("--rowset"), 16).unwrap_or_else(|_| usage()));
                modes += 1;
            }
            "--mod8" => {
                // An optional total follows.
                let total = rest.peek().and_then(|t| t.parse::<u128>().ok());
                if total.is_some() {
                    rest.next();
                }
                args.mode = Mode::Mod8(total);
                modes += 1;
            }
            "--sum" => {
                // Everything after --sum is a file name.
                let files: Vec<String> = rest.by_ref().cloned().collect();
                if files.is_empty() {
                    usage();
                }
                args.mode = Mode::Sum(files);
                modes += 1;
            }
            other => match other.parse::<u32>() {
                Ok(v) if (1..=31).contains(&v) && n.is_none() => n = Some(v),
                _ => usage(),
            },
        }
    }
    if modes > 1 || args.selection.stride == 0 || args.selection.phase >= args.selection.stride {
        usage();
    }
    if matches!(args.mode, Mode::Sum(_) | Mode::Version) {
        if n.is_some() || args.cuda || args.out.is_some() {
            usage();
        }
        return args;
    }
    args.n = n.unwrap_or_else(|| usage());
    args
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
            count_on_cpu(n, &work_items(n), &mut log);
            assert_eq!(log.total(), q, "Q({n})");
        }
    }

    #[test]
    fn phases_add_up_to_the_count() {
        let n = 10;
        let items = work_items(n);
        let mut total = 0u128;
        for phase in 0..7 {
            let selected: Vec<RowSet> = (Selection { stride: 7, phase }).pick(&items).copied().collect();
            let mut log = RowsetLog::discard(n);
            count_on_cpu(n, &selected, &mut log);
            total += log.total();
        }
        assert_eq!(total, KNOWN[n as usize - 1]);
    }

    #[test]
    fn one_row_set_matches_a_direct_count() {
        // Every row set's count, times its weight, is the number of solutions
        // whose even-square queens lie in the row sets of its symmetry orbit.
        // Checked on brute-force solutions for n = 8 and 9.
        for n in [8u32, 9] {
            let solutions = symmetric::brute_force_solutions(n);
            let even_rows = |sol: &Vec<u32>| (0..n).filter(|&r| (r + sol[r as usize]).is_multiple_of(2)).fold(0u32, |m, r| m | 1 << r);
            let by_row_set: BTreeMap<u32, u64> = solutions.iter().map(even_rows).fold(BTreeMap::new(), |mut m, mask| {
                *m.entry(mask).or_default() += 1;
                m
            });
            let mut covered = 0u64;
            for row_set in feasible_row_sets(n) {
                let orbit: std::collections::BTreeSet<u32> = [
                    row_set.mask,
                    work::reflect(n, row_set.mask),
                    work::reflect(n, work::complement(n, row_set.mask)),
                    work::complement(n, row_set.mask),
                ]
                .into_iter()
                .filter(|&m| n.is_multiple_of(2) || m == row_set.mask || m == work::reflect(n, row_set.mask))
                .collect();
                let direct: u64 = orbit.iter().map(|m| by_row_set.get(m).copied().unwrap_or(0)).sum();
                let counted = count_solutions(n, &row_set);
                assert_eq!(u64::from(row_set.weight) * counted, direct, "n = {n}, row set {:08x}", row_set.mask);
                covered += direct;
            }
            assert_eq!(covered, solutions.len() as u64);
        }
    }
}

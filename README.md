# pnqueens

Counts n-queens solutions by splitting every solution into its queens on even squares and its queens on odd squares, which cannot attack each other diagonally, so that Q(n) is a sum over row sets R of

    Σ_C E[R, C] · O[R̄, C̄]

with E and O the numbers of one-color partial solutions on given rows and columns (the derivation is the comment at the top of `src/main.rs`). Each row set is one bounded search whose leaves are tallied into a dense table, and the two tables of a row set are joined by a dot product. There is a CPU path and a CUDA path; every count is recorded row set by row set so that it can be audited (VALIDATION.md), and a run can be spread over a fleet of GPU boxes (RUNBOOK.md).

## Results

| n | Q(n) | RTX 5090 hours | |
|---|---|---|---|
| 27 | 234,907,967,154,122,528 | 27.5 | reproduces the published value (OEIS A000170); a rehearsal |
| 28 | 2,564,269,152,506,723,832 | 124.7 | agrees with the reference project's value, computed by a different program |
| 29 | 28,898,919,943,828,158,328 | 1,636.6 | no other computation to compare with |

Q(28) was completed on 2026-09-22 and Q(29) on 2026-09-30, on a fleet of rented RTX 5090 boxes, both with one binary (source commit `eedf00645cfd`). Both passed every check that applies to them: each feasible row set recorded exactly once with its weight, a sample of the phases recomputed in a different kernel configuration in exact agreement, and the mod-8 identity against an independent count of the 180°-symmetric solutions. The release of each count is the merged record of every row set (`rowsets_28.csv`, 10,034,104 row sets; `rowsets_29.csv`, 38,781,096) with its phase manifests and SHA-256 sums. VALIDATION.md states what each count passed and how to audit a release on a CPU; RUNBOOK.md has the measured rates and what the long run taught; `cuda-notes.md` 2.12 has the kernel's production measurements.

## Layout

| path | contents |
|---|---|
| `src/main.rs` | the derivation, the CPU search, the command line, the checks of records |
| `src/work.rs` | the feasible row sets and their weights, the work list, its fingerprint, phases |
| `src/solution_table.rs` | dense tables indexed by colex ranks of column sets |
| `src/rowset_log.rs` | the per-row-set record files |
| `src/symmetric.rs` | the independent count of 180°-symmetric solutions (the mod-8 check) |
| `src/conventional.rs` | Q(n) by conventional bitmask backtracking, the baseline of the CPU comparison |
| `src/gpu/` | the CUDA path: the work plan, the driver, the kernel emulation test |
| `kernel/` | the CUDA kernels (`enumerate.cu`, `join.cu`) and their host-C++ shim |
| `tools/` | fleet operation: ship, worker, presign, sync, status, fleet |
| `cuda-notes.md` | the GPU path's design, optimization history and measurements |
| `cpu-comparison.md` | the decomposition against conventional backtracking, both on one CPU thread |
| `cpu-comparison-times.tex`, `cpu-comparison-times-linear.tex` | pgfplots figures of the two counters' times from that comparison, on a log and on a linear scale, for a LaTeX document |

## Build, test, run

```
cargo build --release                                  # CPU only
cargo test --release                                   # fast tests
cargo test --release --features emulate                # plus the kernel emulated on the CPU (needs a C++ compiler)
cargo build --release --features cuda                  # the GPU path (NVIDIA driver + libnvrtc at run time)

./target/release/pnqueens 16                           # Q(16) on the CPU: 14772512 ok
./target/release/pnqueens 16 --threads 1               # the same on one thread
./target/release/pnqueens 16 --conventional            # the same by conventional backtracking, one thread
./target/release/pnqueens 10 --conventional --repeat 1000   # the mean time of 1000 counts
./target/release/pnqueens 24 --cuda                    # Q(24) on an RTX 5090 in about five minutes
./target/release/pnqueens 28 --cuda --stride 600 --phase 7   # one phase of a fleet run
./target/release/pnqueens --sum rowsets_28_s600_p*.csv       # re-tabulate and check records
./target/release/pnqueens 28 --rowset 0aaaaaaa         # one row set on the CPU, as a record line: count and the two leaf counts
./target/release/pnqueens 28 --mod8 2564269152506723832
./target/release/pnqueens --sum rowsets_29.csv                # audit the Q(29) release: about a quarter of an hour
```

The toolchain is pinned (`rust-toolchain.toml`) so every box builds the same binary; the source commit is recorded in it at build time.

## License

MIT; see `LICENSE`.

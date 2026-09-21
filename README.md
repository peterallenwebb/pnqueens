# pnqueens

Counts n-queens solutions by splitting every solution into its queens on
even squares and its queens on odd squares, which cannot attack each other
diagonally, so that Q(n) is a sum over row sets R of

    Σ_C E[R, C] · O[R̄, C̄]

with E and O the numbers of one-color partial solutions on given rows and
columns (the derivation is the comment at the top of `src/main.rs`). Each row
set is one bounded search whose leaves are tallied into a dense table, and
the two tables of a row set are joined by a dot product. There is a CPU path
and a CUDA path; every count is recorded row set by row set so that it can be
audited (VALIDATION.md), and a run can be spread over a fleet of GPU boxes
(RUNBOOK.md).

## Layout

| path | contents |
|---|---|
| `src/main.rs` | the derivation, the CPU search, the command line, the checks of records |
| `src/work.rs` | the feasible row sets and their weights, the work list, its fingerprint, phases |
| `src/solution_table.rs` | dense tables indexed by colex ranks of column sets |
| `src/rowset_log.rs` | the per-row-set record files |
| `src/symmetric.rs` | the independent count of 180°-symmetric solutions (the mod-8 check) |
| `src/gpu/` | the CUDA path: the work plan, the driver, the kernel emulation test |
| `kernel/` | the CUDA kernels (`enumerate.cu`, `join.cu`) and their host-C++ shim |
| `tools/` | fleet operation: ship, worker, presign, sync, status, fleet |
| `cuda-notes.md` | the GPU path's design, optimization history and measurements |

## Build, test, run

```
cargo build --release                                  # CPU only
cargo test --release                                   # fast tests
cargo test --release --features emulate                # plus the kernel emulated on the CPU (needs a C++ compiler)
cargo build --release --features cuda                  # the GPU path (NVIDIA driver + libnvrtc at run time)

./target/release/pnqueens 16                           # Q(16) on the CPU: 14772512 ok
./target/release/pnqueens 24 --cuda                    # Q(24) on an RTX 5090 in about five minutes
./target/release/pnqueens 28 --cuda --stride 600 --phase 7   # one phase of a fleet run
./target/release/pnqueens --sum rowsets_28_s600_p*.csv       # re-tabulate and check records
./target/release/pnqueens 28 --rowset 0aaaaaaa         # one row set on the CPU, as a record line
./target/release/pnqueens 28 --mod8 2564269152506723832
```

The toolchain is pinned (`rust-toolchain.toml`) so every box builds the same
binary; the source commit is recorded in it at build time.

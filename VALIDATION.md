# Validation

What is checked, by what, and what each check establishes. The aim is that a
reader can verify the tests on a laptop in seconds, the GPU path in minutes
on any NVIDIA GPU, and any single line of a published count in minutes on a
CPU, with nothing but this repository.

## 1. Tests (`cargo test --release`, under five seconds)

| test | establishes |
|---|---|
| `known_queen_counts` | the decomposition on the CPU equals OEIS A000170 for n ≤ 15 |
| `one_row_set_matches_a_direct_count` | every row set's count, times its weight, equals the number of brute-force solutions whose even-square queens lie in that row set's orbit (n = 8, 9), and the orbits cover every solution |
| `weights_account_for_every_feasible_row_set` | the symmetry weights add up to the number of feasible row sets without the symmetry, n ≤ 14 |
| `phases_add_up_to_the_count`, `phases_partition_the_items` | the phases of a stride partition the work and their partials sum to Q(n) |
| `fingerprints_are_stable` | golden work-map fingerprints for n = 8, 12: a change to the item ordering must be a conscious act |
| `rotation_fixed_count_matches_brute_force` | the independent counter of 180°-symmetric solutions equals a brute-force count, n ≤ 11 |
| `the_mod_8_identity_holds_on_the_published_values` | Q ≡ S₂ (mod 8) on OEIS values, n ≤ 22 |
| `batches_cover_the_items_in_order_within_budget`, `ticket_count_is_the_product_of_prefix_radices`, `default_prefix_depth_on_a_5090` | the GPU work plan |
| `emulated_kernel_matches_known_counts` (`--features emulate`) | the CUDA kernel source, compiled as host C++, reproduces Q(1 … 12) through the real batching at four prefix depths, with the tables matching the tallies |

`python3 tools/test_tools.py` exercises the worker and status tools on the
CPU: resume, a replacement worker, a verification rerun, the refusal of a
foreign work map, and the merged records passing `--sum`.

## 2. The ladder and the race detector

Every binary that produces a production count first runs `pnqueens N --cuda`
for n = 16 … 22 on every box (`fleet.py ladder`) and must print `ok` (equal
to OEIS) at every step, and runs under the CUDA race detector at n = 10
(`fleet.py racecheck`: `compute-sanitizer --tool racecheck`), which must
report zero hazards. The detector targets what neither the counts nor the
kernel's tally check can: shared memory written by one lane and read by
another without the ordering the memory model requires. The warp's prefix
pool is such a channel, and a review in September 2026 found it relied on a
warp vote for ordering, which votes do not provide; the detector reported
over a million hazards at n = 12 before the two `__syncwarp` barriers in
`refill_pool` and none after (`cuda-notes.md`, 2.11).

## 3. Production counts

Four checks apply to a finished count, all made by the binary itself from
the records (`pnqueens --sum rowsets_*.csv`). Every outcome is an exit
status: 0 when the records are complete and every check passed, 1 when any
check failed, 3 when the records are incomplete; `status.py` and
`fleet.py status` propagate it, so a failed check cannot pass unnoticed
through the tooling. A full count (`pnqueens N`) applies checks 2 and 3 to
its own total and exits 1 on a failure; `--mod8 TOTAL` exits 1 on FAIL.

1. **Coverage and weights.** Every feasible row set appears exactly once with
   its symmetry weight, or more than once with identical records; a
   disagreement is a CONFLICT and no total is printed. The phases of the
   stride are all present. Every file carries the same work-map fingerprint.
   A record is `n,rowset_hex,weight,count,even_leaves,odd_leaves`: the count
   and the sizes of the two one-colour solution sets it joins. No row set
   may have a count without placements on both colours.
2. **The published value** for n ≤ 27, and for n = 28 the value of the
   reference project (2,564,269,152,506,723,832, computed in 2026 by a
   different program: a read-side join of the same decomposition on a fleet
   of RTX 5090s). Q(25) … Q(27) are rehearsed with exactly the production
   machinery before a new term is attempted.
3. **The mod-8 identity** Q ≡ S₂ (mod 8), with S₂ the number of solutions
   fixed by the 180° rotation, counted by a search that shares nothing with
   the decomposition (`symmetric.rs`; `pnqueens N --mod8 TOTAL`). This is a
   limited consistency check: an error that is a multiple of 8 passes it.
   For a count with no reference value, Q(29), the weight rests on the
   other three checks and on the per-row-set records themselves.
4. **A verification sample.** A few percent of the phases are recomputed in
   another configuration of the kernel (prefix depth, residency), which
   changes the ticket decomposition and the order of every table update but
   must reproduce every row-set count exactly (RUNBOOK.md). The manifests
   record the effective configuration, and `status.py` warns when a
   duplicate ran in the same one as the original.

**Provenance.** Every record file carries the binary's source commit, the
SHA-256 of the binary, the host, the device, the options, and the work-map
fingerprint; every phase's manifest adds the launch statistics, the largest
table entry and the wall time. The source commit is the last commit that
touched what the binary is built from (`src`, `kernel`, the Cargo files,
`build.rs`, the toolchain pin), so commits to tools or documents leave the
binary, its hash and its recorded commit unchanged, and the binary
rehearsed is provably the binary of the production run; the ship step
refuses a dirty tree, so the tools and documents on a box are a commit too.

**The kernel's own check.** Table entries are 16 bits, two to a word. The
join sums the entries of each colour and the host requires the sum to equal
the weights the searches tallied; an entry that overflowed would have carried
into its neighbour and changed the sum, so every batch of every run is
proven free of overflow (the largest entries seen are 291 at n = 22, 982 at
n = 24, at most 2,224 at n = 25, measured as a bound). Per-row-set counts are u64, totals u128.

## 4. Auditing a release

A release is the merged CSV of every row set of a count, its manifests, and
the SHA-256 of the CSV. With a build of this repository:

```
cargo build --release
./target/release/pnqueens --sum rowsets_28.csv          # coverage, weights, total, mod 8: minutes
./target/release/pnqueens 28 --rowset 0aaaaaaa            # any one line, recomputed on the CPU: minutes
./target/release/pnqueens 28 --items                      # the work map the records name
```

`--rowset` prints the exact line the records must contain for that row
set: pick lines at random, or the heaviest, or the ones you distrust. The
two leaf counts on the line can also be checked by a program of a few dozen
lines that counts non-attacking placements on one colour of the given rows
(the test `one_row_set_matches_a_direct_count` contains one), which tests
the search without the tables or the join; the count tests all three. The
whole count on a CPU is out of reach for n ≥ 26, but every part of it is
within reach of anyone, which is what the per-row-set records are for.

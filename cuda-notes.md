# CUDA path: design, optimization history and measurements

Notes on `src/gpu/`, `kernel/enumerate.cu` and `kernel/join.cu`: what the
GPU path does, how it got to its current shape, what was measured along the
way (including the ideas that did not work), and what to try next.

The host side is split by what needs a device: `src/gpu/plan.rs` is the work
plan (items, table shapes, ticket counts, batches) in plain Rust;
`src/gpu/device.rs` is the CUDA driver (NVRTC compilation, buffers, the three
launches of a batch); `src/gpu/mod.rs` runs the plan batch by batch and logs
the results. `src/gpu/emulate.rs` with `kernel/emulate.cpp` compiles the
enumerate kernel as host C++ and runs it against the plan, so kernel logic
can be checked on any machine (`cargo test --features emulate`).

All timings are on a rented RTX 5090 (170 SMs, 96 MB L2, 100 KB shared
memory per SM, 32 GB), wall time of `pnqueens <n> --cuda` unless stated. The
reference implementation we compare against ran on the same GPU.

## 1. What the GPU path computes

Same decomposition as the CPU path. For each feasible even-queen row set R
(with its symmetry weight) we need

1. `E[R, C]` for all column sets C: one-colour placements on even squares,
   one queen per row of R;
2. `O[R̅, C̅]` for all C: placements on odd squares, rows R̅;
3. `sum_C E[R, C] · O[R̅, C̅]`, the number of full solutions with even-square
   queens exactly in rows R.

Tables are dense arrays indexed by the colex rank of the column set, exactly
as `SolutionTable` does on the CPU (split the mask into even/odd column
masks, rank each with the two-level `RANK_LO`/`RANK_HI` tables,
`index = rank(even) * odd_stride + rank(odd)`). On the GPU the entries are
16 bits, two per 32-bit word, tallied with one 32-bit atomic of the weight
shifted into place (2.10); the join sums the entries of each colour and the
host requires the sums to equal the weights the searches tallied, which an
overflowed entry (a carry into its neighbour) would break. Largest entries
seen: 291 at n = 22, 982 at n = 24, 2,224 at n = 25 (odd tables carry the
mirror weight 2); the reference project saw 3,662 unweighted at n = 28.

### Kernels

- `enumerate_kernel` (`kernel/enumerate.cu`): the one-colour search, a
  transcription of the CPU `enumerate` (bitmask backtracking over columns,
  left and right diagonals). Its one `parity` argument is the colour: parity
  0 tallies leaves into the item's E table; parity 1 tallies into its O
  table at the index of the *complement* of the leaf's columns, so the two
  tables are aligned entry for entry.
- `join_kernel` (`kernel/join.cu`): per item, the dot product of its two
  tables. Grid `(items, chunks)`, warp shuffle reduction, one 64-bit atomic
  per warp.

### Work division

Row sets ("items") are sorted heaviest table first and processed in batches
that fit a table memory budget (`--table-mem`, default 8 GiB, split between
the two colours). Per batch: zero the tables, launch the even-square search
for all items, the odd-square search for all items, then the join; copy the
per-item sums back and append them to the row-set log.

Within a launch, work is cut into **tickets**: a ticket fixes the columns of
the first `prefix_depth` rows of the item as a mixed-radix number. Threads
are **persistent**: each loops taking the next ticket from a global atomic
counter, decoding its prefix (tickets with a conflict in the prefix are
"dead" and finished at once), and running the depth-first search below it.
The whole kernel body is one loop in which each iteration is either a ticket
setup or a single search step, so the lanes of a warp stay together even
though tickets differ enormously in cost.

The search state (columns taken, the two diagonal families, the row being
placed and its columns still to try) lives in registers. The diagonals are
kept in board coordinates as two 64-bit masks (bit `c - r + n - 1` and bit
`c + r`), so placing and removing a queen are the same XOR and a pop
recomputes the parent's state from the child's; a stack frame is one byte
per thread (the column placed) in shared memory. Residency is then bounded
by registers (53–56 → 9 blocks of 128 per SM, 36 warps), not shared memory.

Ticket setup (item lookup, prefix decode, discarding dead tickets) is done
per warp, all lanes together: when a lane needs work and the warp's pool is
empty, lane 0 claims 32 consecutive tickets, every lane decodes one, and the
live prefixes go to a per-warp pool in shared memory; a lane then picks one
up with a few loads. Which lanes need work and which pool slots are live is
tracked with warp votes at the top of every iteration. (Per-lane setup was
measured to cost far more than its instruction count: a lane in the long
setup path makes the whole warp wait, and at prefix 4–5 most tickets are
dead, so some lane was in it almost every iteration.)

Mirror symmetry for odd n (`NQ_MIRROR`): the odd-square search restricts the
first row's queen to columns up to the centre; a leaf left of centre has
weight 2, one in the centre column weight 1. Same as the CPU
`backtrack_odd_solutions`.

Compile-time constants baked in by NVRTC per run: `NQ_N`, `NQ_PREFIX`,
`NQ_BLOCK`, `NQ_FRAMES`, `NQ_MIRROR`. The host reads back per-launch
counters (tickets, dead tickets, search steps, leaves) and prints per-batch
and per-colour rates: these were the key instrument for everything below.

## 2. Optimization history

Starting point for this round: a working persistent-thread kernel with a
five-word shared-memory stack, whose mode 1 *read* E[R, C] at each odd-square
leaf. It was 1.3× slower than the reference at n = 22 and 1.67× at n = 24.

### 2.1 Measurement first

- Nsight Compute is blocked in the container (`ERR_NVGPUCTRPERM`), so the
  kernel got its own counters: per-thread tallies of tickets, dead tickets,
  steps and leaves, added to a device buffer at exit; the host times each
  launch.
- Static analysis with `nvcc -Xptxas -v` and `cuobjdump -sass`: 38–40
  registers, no spills, no local memory, ~370 SASS instructions. A push step
  is ~40 instructions (4 `LDS`, 5 `STS`), a pop ~7, a leaf ~55 of which ~35
  are `table_index` (two bit gathers, two rank lookups). Leaves are 20–24%
  of steps.
- Finding: at n = 22 the read side was 76% of the time and 30% slower per
  step than the tabulate side; its leaf ended in `LDG` followed three
  instructions later by the `IADD.64` that consumed it.

### 2.2 Deferred read-side accumulate (kept briefly, then superseded)

Hypothesis: the leaf stalls on load latency. Parking the loaded value and
adding it at the next leaf gained 8%; bypassing L1 (`__ldcg`) gained
another 8%; together 18% (read side 13.0 → 10.6 s at n = 22). A 4-deep
queue was *worse* than none. So latency was a minor factor. Three
experiments settled it:

| experiment (n = 22, read-side trees) | time |
|---|---|
| real table read | 13.1 s |
| load from a fixed address (pure compute) | 8.2 s |
| `RED` atomic instead of the load | 8.4 s |
| tabulate-side trees with a load instead of `RED` | 4.1 → 6.6 s |

Random 4-byte table reads cap at ≈ 2.2–2.7 × 10¹⁰/s on this GPU however
many are in flight per thread; fire-and-forget atomics are nearly free.
Evict-first loads (`__ldcs`) were far slower (19 s), so L2 hits matter.

### 2.3 Tabulate both colours, join by dot product

Mode 1 stopped reading and started tallying into its own table (aligned via
the complement index); `join_kernel` does the per-item dot product. The join
costs 0.14 s at n = 22 and 2 s at n = 24; the table budget is halved per
colour.

| n | before | after |
|---|---|---|
| 22 | 15.4 s | 13.4 s (reference 13.5 s) |
| 23 | 169 s | 182 s (regression, see 2.5) |
| 24 | 613 s | 498 s |

### 2.4 Residency: four-word frames and the occupancy API

The n = 23 regression led to checking residency properly. CUDA reserves 1 KB
of shared memory per block; with five-word frames at 20 frames × 64 threads
the hand arithmetic said 4 blocks per SM but only 3 were resident (6 warps),
and the fourth ran as a late second wave. `grid_size` now uses
`cuOccupancyMaxActiveBlocksPerMultiprocessor`. Dropping the row word from
the frame (16 B/frame) makes every configuration through n = 28 run
2 × 128 threads (8 warps) per SM.

| n | five-word | four-word |
|---|---|---|
| 22 | 13.40 s | 13.35 s |
| 24 | 498 s | 393 s |
| 23 | 182 s | 235 s (still wrong, see next) |

### 2.5 The L2 footprint of the items in flight

n = 23 at prefix 3 stayed slow whatever the batch size (8, 16, 24 GB budgets:
181 s each), yet ran at full speed at prefix 4. The mechanism: tickets are
handed out in order, so the tickets in progress at any moment are a window
of roughly (resident threads × a small factor) consecutive tickets. At
n = 23 an item has only ~1,700 tickets at prefix 3, so the window spans
~100 items of 1.7 MB tables — more than the 96 MB L2 — and every table atomic
becomes a DRAM read-modify-write. n = 22 has the same ticket density but
half-size tables (fits); n = 24 at prefix 4 has 120k tickets per item
(window of a few items).

Confirmed the hard way: claiming tickets in chunks of 16 per thread (to cut
counter atomics) widened the window 16× and made n = 22 **4.6× slower**
(13.3 → 61.6 s). Reverted, and recorded in the kernel header.

The default prefix depth is now chosen by this rule (`default_prefix_depth`):
the smallest depth ≥ 3 at which `(window / tickets per heaviest item) ×
heaviest table size ≤ ¾ L2`, with window = 2 × (SMs × 256) threads, and
stopping early if a single item is already in flight or the 2³² tickets per
item limit would be exceeded. It yields depth 3 for n ≤ 22, 4 for 23–26,
5 for 27–29.

### 2.6 Cheap deep prefixes: galloping item lookup

Deeper prefixes mean many more, mostly dead, tickets, and each ticket did a
~15-step binary search over `ticket_base` with dependent global loads. A
thread's tickets only increase, so its item never moves backwards: the
kernel now gallops forward from the previous item and bisects the bracket
(a couple of cached reads). n = 22 at prefix 4: 19.5 → 16.4 s; n = 23 at
prefix 4: 112 → 109 s.

### 2.7 Current state

| n | start of this round | now | reference |
|---|---|---|---|
| 20 | 1.3 s | 1.1 s | 1.6 s |
| 21 | 4.4 s | 4.2 s | 5.2 s |
| 22 | 17.0 s | 13.4 s | 13.5 s |
| 23 | ~169 s | 109 s | — |
| 24 | ~690 s | 388 s | 439 s |

Both colours run at 1.7–1.9 × 10¹¹ search steps/s at every n measured,
i.e. the per-n anomalies are gone, so time tracks the size of the search
tree. Across n = 20..24 the kernel time grows as Q(n)^0.78 (exponent 0.773
to 0.787 on each two-step pair). With Q(28)/Q(27) ≈ 10.95 and Q(29)/Q(28)
≈ 11.4 extrapolated from the known ratios, that gives n = 26 ≈ 3.8 h,
n = 27 ≈ 37 h, Q(28) ≈ 150–170 5090-hours and Q(29) ≈ 1550–1900 h.
Unmeasured risk: the heaviest table is 47 MB at n = 28 and 88 MB at n = 29
against a 96 MB L2; if the atomics spill to DRAM the n = 23 depth-3
experience says up to ~2× slower. An n = 26 run would pin the exponent.

### 2.8 Hardware counters at last (2026-09-20)

A vast.ai *VM* (Ubuntu 22.04, driver 580.105, root) allows profiling, unlike
the earlier containers. Nsight Compute 2026.3.1 and the CUDA 12.8 pieces
(`cuda-nvrtc-12-8`, `cuda-nvrtc-dev-12-8`, `cuda-nvcc-12-8`,
`cuda-cuobjdump-12-8`, `cuda-nvdisasm-12-8`) come from the NVIDIA apt repo
that the image already has configured; `ncu` lives in
`/opt/nvidia/nsight-compute/2026.3.1`. Note the report extension is now
`.ncu-repz`.

Ladder after the clarity refactor of the same day, kernel time
(even + odd + join), all counts correct:

| n | time | steps/s even / odd |
|---|---|---|
| 20 | 0.49 s | 1.3 / 1.7 × 10¹¹ |
| 21 | 3.5 s | 1.8 / 1.5 × 10¹¹ |
| 22 | 12.5 s | 1.7 / 1.8 × 10¹¹ |
| 23 | 104 s | 1.7 / 1.7 × 10¹¹ |
| 24 | 383 s | 1.8 / 1.9 × 10¹¹ |

Same as before the refactor (2.7). The GPU held 2.58–2.85 GHz at 290–480 W
of its 575 W limit, 75 °C: no throttling. Dead tickets: 48–51% at prefix 3,
70–72% at prefix 4.

Static (ptxas for sm_120 from generic PTX, as NVRTC + driver JIT do it):
376 SASS instructions, 37 registers, no spills, 38.9 KB static shared
memory per 128-thread block; 4 LDS, 9 STS, 16 LDG, 6 RED, 1 ATOM, 19 BRA,
9 BSSY/BSYNC. Compiling directly for sm_120 gives 368 instructions and 38
registers, so a `--gpu-architecture` option is not a lever.

`ncu` on the first even and odd launches of n = 22 (`--table-mem 4096`,
`--clock-control none`), both launches alike:

| metric | value |
|---|---|
| issue slots busy | 39% (one instruction per 2.5 cycles per scheduler) |
| active warps per scheduler | 2.0 of 12 (theoretical occupancy 16.7%, limited by shared memory) |
| eligible warps per scheduler | 0.47; 60% of cycles have no eligible warp |
| warp cycles per issued instruction | 5.0: 1.0 issuing, 1.7 fixed-latency dependency, 0.71 short scoreboard (shared memory), 0.67 branch resolving, 0.3–0.4 long scoreboard (global), 0.2 not selected |
| active threads per warp instruction | 12.8 of 32 (12.1 after predication) |
| pipes, % of peak | ALU-heavy 40, LSU 34, ALU 24, ADU 10 |
| L2 hit rate | 95–96%; DRAM 18–24% of peak; L1 hit 72–76% |
| block limit | shared memory 2 blocks/SM; registers would allow 12 |

Reading: the kernel is latency-bound with two warps per scheduler, every
stall type is exposed, and only 40% of the lanes of an issued instruction
do work (the pop / leaf / push paths of one loop iteration are executed one
after another, each with its own lanes). Memory is not a limit at this n.
The two ceilings are issue (ALU-heavy at 40% and LSU at 34% would saturate
at about 2.5× the current issue rate) and divergence (up to 2.5× fewer
warp instructions if lanes converged). More warps (frames out of shared
memory) buys the first factor; shorter and fewer paths per step buy into
the second.

### 2.9 Register-resident search, byte frames, warp-cooperative ticket pool (2026-09-20)

The stack design above (section 1) replaced the four-word shared-memory
frames. Kernel time in seconds, even / odd side, sorted by configuration
(`--prefix P --blocks-per-sm B`); the old kernel is the section 2.8 ladder.
Counts correct in every run.

| n | old kernel | new, per-lane ticket claiming (sweep 1) | new, warp pool (sweep 3) |
|---|---|---|---|
| 22 | p3 b2: 4.0 / 8.3 | p3 b2: 4.0 / 8.3; p4 b2: 6.2 / 8.7; p4 b4: 6.2 / 7.3; p4 b8: 6.9 / 9.1; p4 b12: 9.0 / 13.0; p5 b12: 17.0 / 21.4; p3 b4..b12: 8.6 / 20 … 18.7 / 37.5 | p3 b2: 4.6 / 10.7; p4 b2: 3.7 / 7.8; **p4 b4: 2.9 / 5.8**; p4 b8: 3.8 / 8.2; p4 b9: 4.6 / 10.4; p5 b4: 3.6 / 6.1; **p5 b9: 3.5 / 5.7** |
| 23 | p4 b2: 66 / 37 | — | p5 b9 (default): 48 / 30 |
| 24 | p4 b2: 120 / 261 | p4 b2: 119 / 257 | p4 b2: 117 / 264; p4 b4: 139 / 332; **p5 b4: 92 / 196**; p5 b8: 91 / 194; p5 b9: 95 / 203 |

So: 1.3–1.4× at n = 23 and 24 (78 s and 291 s against 104 s and 383 s), and
1.35× at n = 22 with the right prefix. Three effects were disentangled on
the way:

- **The register design itself is free.** At the old residency and prefix,
  both versions of the new kernel time the old one to within 2%.
- **Prefix depth sets an L2 cliff that moves with residency.** More
  threads in flight span more items; once their tables exceed roughly
  15–30 MB the table atomics become DRAM read-modify-writes (profile:
  DRAM 29% of peak at p4 b9 against 4.8% at p5 b9, `lg_throttle` 2.1 vs
  0.5 cycles). This is why prefix 3 collapses above 2 blocks per SM, prefix
  4 above 4, and why n = 24 needs prefix 5 to use more than 2 blocks. The
  rule in `plan::default_prefix_depth` now uses window factor 3 and a
  quarter of L2, which reproduces every fast/slow point measured here. Note
  the odd-square search of an odd n has fewer tickets per item (mirror), so
  its window is wider: at n = 25 (old kernel, prefix 4) the odd side ran at
  1.2 × 10¹¹ steps/s against 1.8 on the even side. A per-colour depth would
  fix that.
- **Per-lane ticket setup does not scale.** With one ticket claimed per
  lane, prefix 4 cost the even side +2.2 s at every residency (sweep 1),
  far more than its instruction count: the setup path (item search, five
  digits with two integer divisions each, dead check) ran for one lane
  while 31 waited, in most iterations. The warp-cooperative pool removed
  that (p4 b2: 6.2 → 3.7 s) and also divides the counter atomics by 32.

Profile of the first even launch at n = 22 (`--table-mem 4096`), old p3 b2
against new p4 b9 and p5 b9: warps per SM 7.9 → 35.5; issue slots busy
38.6% → 48%; eligible warps per scheduler 0.47 → 1.5 / 1.2; cycles per
issued instruction 5.0 → 18.4; active threads per warp instruction 12.8 →
13.0 / 13.7 (divergence unchanged); executed instructions 2.66 → 3.12 /
3.76 × 10¹⁰ (the dead-ticket decode: +17% at p4, +41% at p5); SM clock
2.87 → 2.56 GHz under the profiler, and 2808 → 2747 MHz mean in the sweeps
at 504 W mean, 580 W peak, i.e. at the power limit. Top stalls of the new
kernel: `mio_throttle` 3.7 / 1.9 (the per-iteration vote and the shared
pool), `long_scoreboard` 2.8 / 7.5 (global loads in refill and pickup),
`short_scoreboard` 2.4, `not_selected` 2.0 / 1.5, `math_pipe_throttle`
1.4 / 1.0 (ALU-heavy at 52% of peak). ALU is now the busiest pipe and the
GPU is near its power cap, so further gains must come from fewer
instructions per step, not more warps: constant-divisor digit decode (the
radices are compile-time constants), pool slots that carry the item's
rows, stride and table offset (no global loads at pickup), voting every
few iterations instead of every one, and then the leaf's `table_index`
(item 3 of section 5).

Instrumentation lesson: two sweeps' worth of timings were silently
contaminated by stray processes because `pkill -f` matched the ssh shell
that issued it; `pgrep -af "pnq[u]eens"` before every timing run.

### 2.10 Packed 16-bit tables and instruction cuts (2026-09-20, container box)

Changes: table entries packed two to a word (halving the tables in flight);
ticket digits decoded with literal divisors (a multiply and a shift instead
of a runtime division); pool slots carrying the item's rows, stride and
table offset (a pickup touches no global memory); the diagonal shifts as
funnel shifts (one instruction each, the shift is always below 32); and the
handout vote every fourth iteration. The kernel now takes 68 registers,
which fits 8 blocks per SM; `__launch_bounds__(128, 9)` brings it to 54
registers without spilling, but the ninth block does not pay for the
tighter allocation (n = 22: 3.00 / 6.04 s bounded against 2.85 / 5.74
unbounded), so the bound is applied only when `--blocks-per-sm` asks for a
residency. Kernel seconds, even / odd, at the defaults the recalibrated
rule now picks (prefix 4 at n = 22, 5 at 23–26):

| n | old kernel | 2.9 (best config) | now (defaults) | now vs old |
|---|---|---|---|---|
| 22 | 4.0 / 8.3 | 2.9 / 5.8 | 2.85 / 5.74 | 1.45× |
| 23 | 66 / 37 | 48 / 30 | 47 / 25.5 | 1.42× |
| 24 | 120 / 261 | 92 / 196 | 89.5 / 192 | 1.35× |
| 25 | 2,145 / 1,787 | — | 1,695 / 940 | 1.49× |

The packing did what it was for: prefix 4 at full residency no longer falls
off the L2 cliff at n = 22 (2.85 / 5.74 against 4.65 / 10.4 in 2.9), and the
odd side of n = 25, which the old kernel ran at 1.2 × 10¹¹ steps/s because
the mirror leaves it a wider ticket window, now runs at 2.3 × 10¹¹ like the
even side. The instruction cuts themselves bought only 2–3% at n = 24, where
dead-ticket decoding was a small share to begin with. The rate is now
2.3–2.6 × 10¹¹ steps/s on both colours at every n from 22 to 25, so the
remaining limiters are what the 2.9 profile showed apart from decoding:
divergence (13 of 32 lanes active per instruction), the leaf's
`table_index`, and the power limit. (No profiler on this container box:
ERR_NVGPUCTRPERM, as in the earlier ones.)

### 2.11 The pool's missing barriers (2026-09-21)

An outside review found that the warp's prefix pool (2.9) was written by one
lane and read by another with only `__ballot_sync` between them. A warp
vote orders execution, not memory; only `__syncwarp` (`bar.warp.sync`)
guarantees that lanes see each other's shared-memory stores. It had never
produced a wrong count, because a converged warp executes in lockstep and
shared memory is coherent within the SM, but the compiler was free to move
the stores past the vote. `compute-sanitizer --tool racecheck` made the
defect visible: 15,576 warp-level RAW hazards at n = 8, 159,324 at n = 10,
1,326,204 at n = 12, all on the pool. Two `__syncwarp()` in `refill_pool`
(on entry, ordering the previous pool's reads before its reuse; before the
vote, ordering the stores before the reads) bring that to zero at no
measurable cost. Racecheck cannot run on a large n (it instruments every
shared-memory access of a persistent kernel: n = 14 did not finish in half
an hour), so the checklist uses n = 10. The kernel's tally check cannot see
this class of error, since a search on a corrupted prefix tallies
consistently with itself, and the one-lane emulator has no races at all.

## 3. Things that did not work (keep for reviewers)

- 4-deep software-pipelined read queue: slightly worse than no queue
  (extra moves and register hazards). Moot after 2.3.
- `__ldg` (read-only path) for table reads: no change. `__ldcs`
  (evict-first): much worse.
- Larger table budgets to fight the n = 23 slowdown: no effect (it is the
  in-flight window, not the batch, that matters).
- Chunked ticket claiming: 4.6× slower at n = 22 (L2 footprint, 2.5).
- Prefix 4 everywhere: right for n ≥ 23, ~3 s slower at n = 22.

## 4. Verification practice

- `cargo test --features emulate` before anything goes near a GPU: the
  kernel source, compiled as host C++, must reproduce Q(1…12) at prefix
  depths 1…4 through the real batching and ticket plan.
- `Q(n)` for n = 1…18 checked on the GPU after every kernel change, and
  n = 20…24 in every timing ladder, against `KNOWN`.
- Per-row-set counts are written to `rowsets_<n>.csv` per batch;
  `--sum` re-tabulates and checks completeness.
- Timing runs go under `nohup` to a log file on the box: an interrupted SSH
  session earlier lost a full ladder's output, and two accidentally
  concurrent ladders produced unusable timings. Check `pgrep` before
  launching.

## 5. Future work

In rough order of expected value per unit of effort.

1. **Residency beyond 8 warps.** We use 8 of 48 warps per SM; the step path
   is a dependent chain of shared-memory loads and stores, so more warps
   should hide latency almost linearly until issue-bound. Shared memory per
   thread is the limit (4 words × frames). Options: (a) cap frames at about
   n/2 + 2 and give items with more rows a deeper per-item prefix
   (`item_prefix[]` array; the kernel already decodes a runtime depth), which
   at 12 frames allows 12 warps and at 9 frames 16; needs a 64-bit ticket
   decode or a per-item ticket limit for very tall (and nearly empty) row
   sets; (b) three-word frames by storing the placed bit instead of `cols`
   and keeping `cols` in a register. Measure warps vs. steps/s first with a
   `--frames` override.
2. **Unroll the last row.** When the next row is the last row of the item,
   iterate its free columns inline instead of pushing a frame: removes the
   push/pop pair and one shared-memory round trip per last-row frame, and
   makes each leaf a tight inner-loop iteration. Leaves are 20–24% of steps
   and the last level is most of the tree, so a 30–40% reduction in steps is
   plausible.
3. **Cheaper `table_index`.** ~35 instructions per leaf (two 5-step bit
   gathers and two rank lookups) is ~25% of issued instructions. Since all
   columns of a row share a parity, the dense even/odd column masks could be
   maintained incrementally in registers or frames, leaving only the two
   rank lookups at the leaf. Trade against frame size (item 1).
4. **Ticket counter.** One same-address atomic per ticket. Warp-aggregated
   claiming (`__activemask` + one atomic per group of lanes that need a
   ticket at the same instant) keeps the in-flight window tight, unlike
   chunking, and cuts atomics several-fold. Only matters at deep prefixes.
5. **Large-n table sizes.** At n = 28 the heaviest tables are 47 MB and at
   n = 29 88 MB, comparable to L2; even one item in flight will partly miss.
   Ideas: order tickets within an item so that concurrently running tickets
   touch nearby table regions (the colex rank is monotone in the high
   columns, so ordering prefix digits by column may help), or split an
   item's table across two launches by a fixed first-row column.
6. **Batch-level overlap.** The three launches per batch are serialized
   with host synchronizations between them for timing. Running the join and
   the next batch's zeroing on a second stream, or dropping the mid-batch
   syncs when not instrumenting, would recover the few percent of idle GPU
   between launches at small n.
7. **Profiling with counters.** If a machine with `ncu` permissions becomes
   available, a single "SpeedOfLight / WarpStateStats / MemoryWorkload"
   capture of a mode 0 launch at n = 22 would confirm the issue-bound vs.
   latency-bound picture behind item 1 and size the L2 hit rate behind
   item 5.
8. **Multi-GPU and restart.** Q(28) is hundreds of GPU-hours: the row-set
   log already makes the computation restartable per item; add a `--skip`
   of already-logged row sets and a device/shard split so several GPUs can
   share one run and a crash costs one batch.

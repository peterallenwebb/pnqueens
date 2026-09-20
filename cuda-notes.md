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
`index = rank(even) * odd_stride + rank(odd)`).

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

The DFS stack lives in shared memory, frame `f` of thread `t` at word
`[f][t]` of four arrays (columns taken, the two diagonal masks, columns of
the row still to try). The frame's row is kept in a register and recomputed
on pop (previous set bit of the row mask). Shared memory per thread is what
bounds residency, so frames are kept to four words.

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

# The decomposition against conventional backtracking, on one CPU thread

How much faster is the checkerboard decomposition than the conventional
count? This note measures both on one thread of one CPU, from n = 6, where a
count takes microseconds and is repeated a thousand times to be timed, to
n = 20 for the conventional count and n = 21 for the decomposition, so that
the ratio and how it changes with n can be seen. Measured 2026-10-29 and 30.

## The two counters

**Conventional** (`src/conventional.rs`, `pnqueens N --conventional`). One
queen per row, backtracking down the rows. The squares of the current row
attacked by the queens above it are three bitmasks: the columns taken and the
two diagonal families, each shifted by one column per row. The free columns
of a row are one AND-NOT; candidates are taken lowest bit first; the last row
is not searched, since each free column in it is one solution. The mirror
c → n-1-c fixes no solution, so the first row's queen is placed only left of
centre and the count doubled (for odd n the centre column is placed too, with
the second row's queen kept left of centre). This is the textbook fast
counter: no further symmetry (the full group of the square would save up to
another factor of four at some bookkeeping cost), no SIMD, no threads.

**Decomposition** (the CPU path, `pnqueens N --threads 1`). Exactly what the
production CPU path does, on one thread: for each representative feasible
row set R, the even-square search (`enumerate`) tallies its leaves into the
dense table E[R, ·]; the odd-square search on the complementary rows looks up
E[R, C] at each of its leaves and adds it to the row set's count; the count
is weighted by the size of the row set's symmetry orbit (2 or 4), and for odd
n the odd-square search places its first queen only up to the centre column.
The search inside both colours is the same bitmask backtracking as the
conventional counter, over a subset of the rows and half the columns. (The
GPU path differs in form, tallying both colours into tables and joining them
with a dot product, but visits the same leaves.)

Both counters use symmetry to a comparable degree (a factor of two to four),
so the comparison is between the two ways of searching, not between two
amounts of symmetry.

## Method

| | |
|---|---|
| machine | Apple M1 Pro (8 performance + 2 efficiency cores), macOS 26.6.2, otherwise idle |
| toolchain | rustc 1.98.1 (pinned), `cargo build --release` |
| source | commit dc5610a plus the changes that added `conventional.rs`, `--conventional` and `--threads` (the binary records `eedf00645cfd-dirty`: the last commit that touched the sources, with uncommitted changes) |
| timed | the search alone, as the program's `counted in` line: for the decomposition from the first row set to the last (the work list, the record file and the mod-8 check are outside it); for the conventional counter the whole count. Process start-up is outside both. |
| repetitions | n ≤ 13: the mean per count of 1000 counts in one process (`--repeat 1000`), the median of three such rounds, which agree within about 10% (28% at n = 6, where a count takes microseconds); n = 14, 15: best of three single runs; one run above |
| order | all decomposition runs n = 8 … 21, then all conventional runs n = 8 … 19, one process at a time; the repeated runs were a second sweep, n = 6 … 13, in the same order; the conventional n = 20 run came last, on its own |

The one-colour leaf counts come from the record files of the decomposition
runs (`even_leaves` + `odd_leaves`, summed over the representative row sets:
the leaves actually visited, before weighting). The commands were

```
pnqueens N --threads 1 --out rowsets_N.csv                  # n ≥ 14
pnqueens N --conventional
pnqueens N --threads 1 --repeat 1000 --out rowsets_N.csv    # n ≤ 13
pnqueens N --conventional --repeat 1000
```

## Results

| n | Q(n) | row sets | one-colour leaves | leaves per solution | conventional | decomposition | speedup | conventional growth | decomposition growth |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 4 | 7 | 37 | 9.25 | 0.2 µs | 4.6 µs | 0.04× | – | – |
| 7 | 40 | 19 | 163 | 4.08 | 0.6 µs | 8.4 µs | 0.07× | 3.4× | 1.8× |
| 8 | 92 | 23 | 330 | 3.59 | 2.1 µs | 11.8 µs | 0.18× | 3.8× | 1.4× |
| 9 | 352 | 66 | 1,794 | 5.10 | 8.5 µs | 29.1 µs | 0.29× | 4.1× | 2.5× |
| 10 | 724 | 71 | 3,584 | 4.95 | 75.8 µs | 66.2 µs | 1.1× | 8.9× | 2.3× |
| 11 | 2,680 | 236 | 24,201 | 9.03 | 455.2 µs | 418.5 µs | 1.1× | 6.0× | 6.3× |
| 12 | 14,200 | 252 | 54,580 | 3.84 | 2.52 ms | 1.28 ms | 2.0× | 5.5× | 3.1× |
| 13 | 73,712 | 868 | 412,213 | 5.59 | 13.4 ms | 6.97 ms | 1.9× | 5.3× | 5.4× |
| 14 | 365,596 | 890 | 962,001 | 2.63 | 78 ms | 21 ms | 3.7× | 5.8× | 3.0× |
| 15 | 2,279,184 | 3,235 | 8,326,689 | 3.65 | 487 ms | 132 ms | 3.7× | 6.2× | 6.3× |
| 16 | 14,772,512 | 3,299 | 20,795,310 | 1.41 | 3.25 s | 427 ms | 7.6× | 6.7× | 3.2× |
| 17 | 95,815,104 | 12,190 | 194,333,014 | 2.03 | 22.8 s | 2.89 s | 7.9× | 7.0× | 6.8× |
| 18 | 666,090,624 | 12,283 | 509,089,301 | 0.76 | 169 s | 9.93 s | 17.0× | 7.4× | 3.4× |
| 19 | 4,968,057,848 | 46,252 | 5,153,490,879 | 1.04 | 1,315 s | 72.4 s | 18.2× | 7.8× | 7.3× |
| 20 | 39,029,188,884 | 46,508 | 14,297,005,978 | 0.37 | 10,763 s | 265 s | 40.6× | 8.2× | 3.7× |
| 21 | 314,666,222,712 | 176,484 | 153,476,447,629 | 0.49 | – | 2,030 s | – | – | 7.7× |

Growth is the time at n divided by the time at n - 1; speedup is the
conventional time divided by the decomposition time at the same n (below 1×
the conventional counter is the faster).
`cpu-comparison-times.tex` is a pgfplots figure of the two time columns on a
logarithmic scale, for a LaTeX document; `cpu-comparison-times-linear.tex` is
the same on a linear scale, where every count below n = 17 lies on the axis.

## Reading the table

**At small n the conventional counter is the faster.** Up to n = 9 it wins,
by 28 times at n = 6 and 3.4 times at n = 9, where a count takes
microseconds: the decomposition pays a fixed cost per row set (a table set
up, two searches started) and per count (the hand-off to the thread pool)
that the conventional counter has no counterpart for. By n = 10 that cost
is spread thin (the decomposition's time per leaf falls from 124 ns at n = 6
and 36 ns at n = 8 to under 25 ns at every larger n) and the two counters
cross; the
decomposition leads by a factor of two at n = 12 and 13, and from there its
lead grows.

**The conventional counter's time is the number of solutions.** Its cost
per solution rises slowly as the search deepens, from 105 ns at n = 10 and
180 ns at n = 13 to between 220 and 280 ns at n = 16 to 20 (the table
below), so its time grows very nearly as Q(n) does: between five and eight
times per step from n = 12 on, the factor creeping up with n.

**The decomposition's time is the number of one-colour leaves it visits**, at
13 to 21 ns each, and those grow more slowly than the solutions. Every
solution is a pair of one-colour placements, one on the rows R and one on
R̄, with complementary columns; the conventional search visits every pair,
while the decomposition visits every placement once and counts the pairs by
looking up a table. The leaves per solution column shows the effect: over two
steps the number of solutions grows 40× to 63×, the number of leaves 22× to
30×, so the leaves per solution halve, and so does the decomposition's time
relative to the conventional time.

| n | conventional, ns per solution | decomposition, ns per leaf | solutions, growth over two steps | leaves, growth over two steps | conventional time, growth over two steps | decomposition time, growth over two steps |
|---|---|---|---|---|---|---|
| 16 | 220 | 20.5 | 40.4× | 21.6× | 42× | 20.3× |
| 17 | 238 | 14.9 | 42.0× | 23.3× | 47× | 21.9× |
| 18 | 254 | 19.5 | 45.1× | 24.5× | 52× | 23.2× |
| 19 | 265 | 14.1 | 51.9× | 26.5× | 58× | 25.1× |
| 20 | 276 | 18.5 | 58.6× | 28.1× | 64× | 26.7× |
| 21 | – | 13.2 | 63.3× | 29.8× | – | 28.0× |

**The alternation.** For even n, reflecting the complement of a row set is a
further symmetry of the work list (orbit weight 4), so an even n has hardly
more representative row sets than the odd n before it (3,299 against 3,235;
46,508 against 46,252), and from n = 12 on the step from odd to even n costs
the decomposition only 3.0× to 3.7×, while the step into odd n costs 5.4× to
7.7×. The conventional counter has no such alternation, so the speedup jumps
at even n and barely moves at odd n: 1.1× and 1.1× at n = 10 and 11, then
2.0× and 1.9×, 3.7× and 3.7×, 7.6× and 7.9×, 17× and 18×, then 41× at
n = 20. Read over two steps, both counters grow steadily, and the ratio
between them about doubles.

**Beyond n = 20.** The n = 20 conventional count took 3.0 hours, 8.2×
the n = 19 time, and 276 ns per solution. The counter was not run past it; at
that cost per solution, a slight underestimate since the cost rises, n = 21
would need:

| n | conventional (s) | decomposition (s) | speedup |
|---|---|---|---|
| 20 | 10,763 (about 3 hours), measured | 265 | 41× |
| 21 | 86,776 (about 24 hours), extrapolated | 2,030 | 43× |

If the ratio kept doubling every two steps, the decomposition would be some
hundreds of times faster at n = 28 on one CPU thread. That is an
extrapolation over eight steps from the last measured ratio, and the table
sizes and the cache behaviour of the lookups change on the way, so it is an
indication of the trend, not a measurement.

**What this does and does not say.** Both runs are one thread of one CPU.
Both counters parallelize trivially, the conventional one over the first
rows' columns and the decomposition over row sets, so the ratio should carry
over to a multi-core CPU. The baseline is a good straightforward counter,
not the state of the art: a counter using the full symmetry group of the
square could search up to four times less, which would lower the ratio by at
most that factor and not change how it grows. Nothing here is about the GPU
path, whose tally-and-join form and its performance are the subject of
cuda-notes.md.

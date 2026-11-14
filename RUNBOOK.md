# Runbook: a fleet run of Q(n)

A run counts one board size on any number of boxes, each with one GPU,
driven from one operator machine over ssh. Nothing but this repository and a
file of signed, expiring, upload-only URLs goes onto a box: no credentials.
Every phase's records are pushed to a Cloud Storage bucket as soon as it is
done, so a box that dies loses at most the phase it was running, and a
replacement box takes over its remaining phases from what the bucket holds.

## Shape of a run

The binary's work list (`pnqueens N --items`: the feasible row sets, heaviest
table first, with a fingerprint of the list) is cut into S **phases**: phase
p is the row sets at positions p, p+S, p+2S, … so every phase samples heavy
and light row sets alike and costs about the same. **Worker** w of W runs the
phases p ≡ w (mod W); W is the number of boxes. A phase's records are its
per-row-set counts with their even and odd leaf counts,
`rowsets_N_sS_pP.csv`, headed by a provenance line, and a
manifest `rowsets_N_sS_pP.json` with the host, GPU, options, statistics and
partial sum. The phases of a run are compatible only when cut from the same
work list; its fingerprint is in the run manifest, in every record and in
every check.

Choose S so that a phase takes well under an hour (the most that a lost box
can cost); a few thousand row sets per phase keeps the GPU full. As measured
in September 2026 on a fleet of eleven RTX 5090 boxes (the GPU-hours are the
sums of the phases' wall times in the manifests):

| n | row sets (`--items`) | S | phase | RTX 5090 hours | wall on the fleet |
|---|---|---|---|---|---|
| 26 | 2,602,198 | 48 | 3.8 min | 3.1 | ~20 min |
| 27 | 10,030,008 | 240 | 6.9 min | 27.5 | ~2.5 h |
| 28 | 10,034,104 | 600 | 12.5 min | 124.7 | 12.2 h |
| 29 | 38,781,096 | 3000 | 32.7 min | 1,636.6 | 7 days 20 h, four boxes lost on the way |

Q(29) ran at the default configuration (prefix depth 6, 9 blocks per SM);
its verification sample at `--prefix 7 --blocks-per-sm 4` took 24.9 minutes
a phase, so a rerun at that size should use those options (`cuda-notes.md`,
2.12).

## One-time setup

**Bucket.** A Google Cloud Storage bucket (here `pnqueens-compute-records`)
with uniform bucket-level access, public access prevention enforced and
object versioning on, and a service account with Storage Object Admin on
that bucket only. Its JSON key lives on the operator machine, outside the
repository, at `../nqueens-signer.key` (mode 600; `*.key` is git-ignored).
`python3 tools/bucket_selftest.py` checks all of this: settings, a signed
upload, its read-back, its listing.

**Boxes.** `boxes.txt` lists them, one per line: `PORT HOST [LABEL]` (labels
default to box01, box02, … in file order and name the worker's log and its
objects). ssh must work without a prompt; extra ssh options such as an
identity file go in `PNQUEENS_SSH_OPTS`. A fresh box needs only the NVIDIA
driver and libnvrtc from the CUDA toolkit: `tools/ship.sh` installs Rust and
the pinned toolchain.

**The operator machine** needs Python 3, `openssl`, `curl`, `ssh` and a
build of this repository (`cargo build --release`) for the checks.

## Launch checklist

```
python3 tools/bucket_selftest.py                  # 0. bucket and key still fine
python3 tools/fleet.py ship                       # 1. build the committed tree on every box (refuses a dirty tree; fails unless every box reports HEAD and one binary hash)
python3 tools/fleet.py ladder                     # 2. n = 16 … 22 on every box: every line "ok"
python3 tools/fleet.py racecheck                  #    and the CUDA race detector on n = 10: "0 hazards" on every box
python3 tools/fleet.py push-urls 27 --stride 120  # 3. sign the run's upload URLs and ship them
python3 tools/fleet.py start 27 --stride 120      # 4. worker i of W on box i
python3 tools/fleet.py status 27 --stride 120     # 5. hourly: liveness per box, sync, missing phases, the binary's checks, ETA
```

The rehearsal is a known value: Q(24) … Q(27) must print `ok` from the
merged records before a new term is attempted. That was done on 2026-09-21
and 22 (VALIDATION.md says with which binaries); Q(28) was then run against the reference project's value, and Q(29),
which has no reference, relies on the checks of VALIDATION.md.

**A lost box.** `status` shows it DEAD and writes the phases nobody has
recorded to `runs/N_sS/missing.txt`. Add the replacement to `boxes.txt`, ship
to it (`tools/ship.sh PORT HOST`), give it the URLs (`fleet.py push-urls`
again ships the existing file), and hand it the phases:

```
python3 tools/fleet.py run-phases 27 --stride 120 --box box13 --phases $(tr '\n' , < runs/27_s120/missing.txt)
```

**The verification sample.** Rerun a few percent of the phases in another
configuration of the kernel: a different prefix depth changes the ticket
decomposition and a different residency the order of every table update,
and the counts must not change. The main run's manifests record the
effective `prefix_depth` and `blocks_per_sm` (the binary chooses the depth
by n when `--prefix` is not given: 6 at n = 27 to 29, with 8 blocks per SM
up to n = 28 and 9 at n = 29), so pick values that
differ from them, for example one deeper and half the residency:

```
python3 tools/fleet.py push-urls 27 --stride 120 --tag verify
python3 tools/fleet.py run-phases 27 --stride 120 --box box02 --phases 7,31,59,83,107 --tag verify --redo -- --prefix 7 --blocks-per-sm 4
```

The rerun's records are `rowsets_27_s120_p7_verify.*`; `status` counts
their row sets as recorded more than once, warns if a duplicate ran in the
same configuration as the original, and the binary refuses any
disagreement.

**Signed URLs last at most seven days.** For a longer run, `fleet.py
push-urls N --stride S --renew` signs and ships a fresh file; the worker
reads the file at every push, so nothing restarts.

## Records

In the bucket: `runs/N_sS/rowsets_N_sS_pP.csv` and `.json` per phase,
`runs/N_sS/manifest.json` for the run, `runs/N_sS/logs/<label>.log` per
worker. `tools/sync.py N --stride S` mirrors them into `runs/N_sS/` on the
operator machine; `tools/status.py` does that and then runs `pnqueens --sum`
over the CSVs, which is the authority on what the records say (VALIDATION.md).
On a box the same files are under `~/runs/N_sS/`, the primary copy until
pushed. The release of a finished count is the merged CSV of every row set
with the manifests, published with its SHA-256 (see VALIDATION.md, "Auditing
a release").

## Operating notes

- A worker records a phase only when the binary has finished it (the files
  are written under `.part` names and renamed), so rerunning `start` resumes
  and a killed worker loses only the phase in flight.
- A pushed record has a `.pushed` marker beside it. A worker queues every
  local record without one when it starts, retries after every phase and at
  the end, and exits with status 2 if any is still unpushed; `status` shows
  the count as `N UNPUSHED` on the box's line. So an outage or an expired
  URL file costs nothing but time: renew the URLs, restart the worker.
- Never stop a process with a `pkill -f` pattern that also appears in your
  own ssh command line: the remote shell matches itself and dies first. Use
  `fleet.py stop`, which uses the pid file.
- Do not mix records of different work maps: a worker whose binary has
  another fingerprint refuses to start against the run's manifest, and
  `--sum` refuses files with different fingerprints. The fingerprint changes
  only when the item ordering does (test `fingerprints_are_stable`).
- Every `fleet.py` command exits non-zero when any box fails its check
  (ship: the binary's source commit and one hash; ladder: seven `ok`; racecheck:
  zero hazards; status: the binary's checks of the records), and the binary
  itself exits non-zero on a MISMATCH, a mod-8 FAIL or a CONFLICT, so the
  checklist can be scripted and a step that fails cannot be overlooked.
  `pnqueens --version` prints what a binary is.
- Timing on shared boxes is contaminated by neighbours; counts are not.

## What the Q(29) run taught

The run of 2026-09-22 to 09-30 lost four of its eleven boxes and was watched
from a laptop that slept for hours at a time. No work was lost and no record
was ever in doubt, but the tail took longer than it had to. Its scripts and
logs are kept in `runs/ops29/` (git-ignored, with a README).

- **Lost boxes.** Do not wait for a replacement. When most boxes have
  finished their own lists, stop the laggards (`fleet.py stop`), refresh
  `missing.txt` with `status`, and spread the missing phases evenly over the
  boxes that answer, one `run-phases` per box. A box lost during that tail
  is handled the same way, a couple of phases at a time to each box as it
  finishes. An unreachable box is not necessarily dead: two boxes went
  silent for an hour behind a flaky gateway and kept recording, so judge a
  box by its records in the bucket, not by ssh.
- **The operator machine** matters at four moments only: renewing the URLs,
  redistributing phases, the verification sample, the release. While it is
  asleep the workers carry on and upload; finished boxes then sit idle until
  it wakes. Renew the URLs days early, by hand if need be.
- **Watcher scripts.** Keep them and their logs in the repository's
  `runs/` tree, not in a temporary directory (the system purged one after
  three days). Put every remote call under `timeout`: an ssh interrupted by
  sleep hung for seven hours. A poll in which no box answers means no
  network, not "nobody alive". Give `status` at least half an hour once a
  large run is complete: `--sum` then also counts S₂, about a quarter of an
  hour at n = 29.
- **Is a box free?** Ask `start_worker.sh`, which answers "worker started"
  or "a worker is running". `fleet.py run-phases` prints that answer but
  exits 0 either way, and `fleet.py cmd` runs in `~/pnqueens` while the
  workers' pid files are in `~/runs`, so a relative `runs/worker_*.pid`
  there finds nothing and every box looks free.
- **A tagged worker** (the verification sample) ends by reporting one
  record it could not push: the run manifest, for which the tagged URL file
  has no entry. It is harmless; the sample's own records are pushed.

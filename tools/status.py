#!/usr/bin/env python3
"""The state of a run from its records: which phases are recorded, which
are missing (written to DIR/<N>_s<S>/missing.txt for a replacement worker),
the time recorded so far and the projection, and the binary's checks of the
records (`pnqueens --sum`: coverage, weights, agreeing duplicates, total).

    status.py N --stride S [--dir runs] [--no-sync] [--binary B] [--key FILE]

Syncs the records from the bucket first unless --no-sync. Exits non-zero
when the binary's checks fail, or when it finds the records incomplete
although no phase is missing.
"""
import argparse, glob, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("n", type=int)
ap.add_argument("--stride", type=int, required=True)
ap.add_argument("--dir", default=os.path.join(HERE, "..", "runs"))
ap.add_argument("--no-sync", action="store_true")
ap.add_argument("--binary", default=os.path.join(HERE, "..", "target", "release", "pnqueens"))
ap.add_argument("--key", default=os.path.join(HERE, "..", "..", "nqueens-signer.key"))
a = ap.parse_args()

run = f"{a.n}_s{a.stride}"
rundir = os.path.join(a.dir, run)
if not a.no_sync:
    subprocess.run([sys.executable, os.path.join(HERE, "sync.py"), str(a.n), "--stride", str(a.stride), "--dir", a.dir, "--key", a.key], check=True)

# A phase is recorded when its manifest is there; its CSV must be too.
records = {}          # phase -> list of (tag, manifest)
for path in sorted(glob.glob(os.path.join(rundir, f"rowsets_{run}_p*.json"))):
    m = re.search(rf"rowsets_{run}_p(\d+)(?:_(\w+))?\.json$", path)
    if not m or not os.path.exists(path[:-5] + ".csv"):
        continue
    records.setdefault(int(m.group(1)), []).append((m.group(2) or "", json.load(open(path))))
present = sorted(records)
missing = [p for p in range(a.stride) if p not in records]
verified = [p for p, v in records.items() if len(v) > 1]
# A verification rerun counts only if it ran another configuration of the
# kernel (prefix depth or residency): otherwise it repeated the same work.
def config(m):
    stats = m.get("stats") or {}
    return (stats.get("prefix_depth"), stats.get("blocks_per_sm"))


same_config = [p for p in verified if len({config(m) for _, m in records[p]}) < len(records[p])]
walls = [m["wall_s"] for v in records.values() for _, m in v if _ == ""]
hosts = sorted({m["host"] for v in records.values() for _, m in v})
print(f"run {run}: {len(present)} of {a.stride} phases recorded ({len(verified)} more than once), {len(missing)} missing; "
      f"{len(hosts)} hosts: {', '.join(hosts)}")
if walls:
    avg = sum(walls) / len(walls)
    print(f"recorded wall {sum(walls) / 3600:.2f} h over {len(walls)} phases, {avg / 60:.1f} min per phase, "
          f"{len(missing) * avg / 3600:.2f} h of phases still to run")
if same_config:
    print(f"WARNING: phases {same_config} were recorded more than once in the same configuration (prefix depth and residency): "
          f"agreement between them verifies less than the runbook's alternate-configuration sample")
with open(os.path.join(rundir, "missing.txt"), "w") as f:
    f.write("\n".join(map(str, missing)) + ("\n" if missing else ""))
csvs = sorted(glob.glob(os.path.join(rundir, f"rowsets_{run}_p*.csv")))
csvs = [c for c in csvs if os.path.exists(c[:-4] + ".json")]
if csvs:
    print("--- pnqueens --sum")
    sys.stdout.flush()
    code = subprocess.run([a.binary, "--sum", *csvs]).returncode
    if code == 3 and missing:
        code = 0                              # incomplete because phases are still to run
    if code:
        sys.exit(f"status: the records FAILED the binary's checks (exit {code})")

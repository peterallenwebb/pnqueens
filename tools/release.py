#!/usr/bin/env python3
"""The release of a finished count: one CSV of every row set, sorted by row
set, with a provenance line naming the run it was merged from; the phase
manifests in a tarball; and SHA-256 sums.

    release.py N --stride S [--dir runs] [--out release] [--binary B]

Reads DIR/<N>_s<S>/ (synced by status.py), merges the phase records (a row
set recorded more than once, by a verification rerun, must agree, and is
written once), and finally runs `pnqueens --sum` on the merged file, which
must say ok. The merged file's provenance line carries the work map, the
number of phases merged, and the commits and binaries that produced them.
"""
import argparse, glob, hashlib, json, os, re, subprocess, sys, tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("n", type=int)
ap.add_argument("--stride", type=int, required=True)
ap.add_argument("--dir", default=os.path.join(HERE, "..", "runs"))
ap.add_argument("--out", default=os.path.join(HERE, "..", "release"))
ap.add_argument("--binary", default=os.path.join(HERE, "..", "target", "release", "pnqueens"))
a = ap.parse_args()

run = f"{a.n}_s{a.stride}"
rundir = os.path.join(a.dir, run)
csvs = sorted(glob.glob(os.path.join(rundir, f"rowsets_{run}_p*.csv")))
csvs = [c for c in csvs if os.path.exists(c[:-4] + ".json")]
if not csvs:
    sys.exit(f"no recorded phases in {rundir}")

rows = {}             # mask -> line without the n field
workmaps, commits, binaries, phases = set(), set(), set(), set()
for path in csvs:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                prov = dict(kv.split("=", 1) for kv in line[1:].split() if "=" in kv)
                workmaps.add(prov.get("workmap")); commits.add(prov.get("commit")); binaries.add(prov.get("binary")); phases.add(prov.get("phase"))
            elif line and not line.startswith("n,"):
                n_, mask, rest = line.split(",", 2)
                if int(n_) != a.n:
                    sys.exit(f"{path}: n = {n_}")
                if rows.setdefault(mask, rest) != rest:
                    sys.exit(f"CONFLICT: row set {mask} recorded as {rows[mask]} and as {rest}")
if len(workmaps) != 1:
    sys.exit(f"the records come from different work maps: {workmaps}")
os.makedirs(a.out, exist_ok=True)
merged = os.path.join(a.out, f"rowsets_{a.n}.csv")
with open(merged, "w") as f:
    f.write(f"# pnqueens n={a.n} workmap={workmaps.pop()} items={len(rows)} merged={len(phases)}_phases_of_stride_{a.stride} "
            f"commit={','.join(sorted(c for c in commits if c))} binary={','.join(sorted(b for b in binaries if b))}\n")
    f.write("n,rowset_hex,weight,count,even_leaves,odd_leaves\n")
    for mask in sorted(rows, key=lambda m: int(m, 16)):
        f.write(f"{a.n},{mask},{rows[mask]}\n")
manifests = os.path.join(a.out, f"manifests_{run}.tar.gz")
with tarfile.open(manifests, "w:gz") as tar:
    for path in sorted(glob.glob(os.path.join(rundir, "*.json"))):
        tar.add(path, arcname=os.path.basename(path))
sums = os.path.join(a.out, "SHA256SUMS")
with open(sums, "a") as f:
    for path in (merged, manifests):
        with open(path, "rb") as g:
            f.write(f"{hashlib.sha256(g.read()).hexdigest()}  {os.path.basename(path)}\n")
print(f"{merged}: {len(rows)} row sets from {len(phases)} phases ({len(csvs)} files); {manifests}; sums in {sums}")
sys.exit(subprocess.run([a.binary, "--sum", merged]).returncode)

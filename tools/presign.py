#!/usr/bin/env python3
"""Signed upload URLs for a run's records (RUNBOOK.md, "Records push").

    presign.py N --stride S [--hosts boxes.txt] [--tag TAG] [--days 7] [--bucket B] [--key FILE] [--out push]

Writes push/urls_<N>_s<S>[_<TAG>].env with one PUT URL per object of the run:
every phase's CSV and manifest (p<P>_csv=..., p<P>_json=...), the run
manifest (manifest=...) and one worker log per box label (log_<LABEL>=...).
The same file goes to every box as pnqueens/runs/push_urls.env, so a
replacement box can push any phase. A verification rerun uses --tag verify,
which names the objects rowsets_<N>_s<S>_p<P>_verify.* and the file
push/urls_<N>_s<S>_verify.env. URLs last at most seven days: presign again
and re-ship the file for a longer run (the worker rereads it at every push).
"""
import argparse, os, sys
import gcs

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("n", type=int)
ap.add_argument("--stride", type=int, required=True)
ap.add_argument("--hosts", default=os.path.join(HERE, "..", "boxes.txt"), help="PORT HOST [LABEL] per line; labels default to box01, box02, ...")
ap.add_argument("--tag", default="")
ap.add_argument("--days", type=float, default=7.0)
ap.add_argument("--bucket", default="pnqueens-compute-records")
ap.add_argument("--key", default=os.path.join(HERE, "..", "..", "nqueens-signer.key"))
ap.add_argument("--out", default=os.path.join(HERE, "..", "push"))
a = ap.parse_args()

key = gcs.load_key(a.key)
run = f"{a.n}_s{a.stride}"
tag = f"_{a.tag}" if a.tag else ""
labels = []
if os.path.exists(a.hosts):
    for line in open(a.hosts):
        parts = line.split()
        if parts and not parts[0].startswith("#"):
            labels.append(parts[2] if len(parts) > 2 else f"box{len(labels) + 1:02d}")
expires = int(a.days * 86400)
lines = []
for p in range(a.stride):
    for ext in ("csv", "json"):
        lines.append(f"p{p}_{ext}={gcs.sign_url(key, a.bucket, f'runs/{run}/rowsets_{run}_p{p}{tag}.{ext}', expires=expires)}")
if not a.tag:
    lines.append(f"manifest={gcs.sign_url(key, a.bucket, f'runs/{run}/manifest.json', expires=expires)}")
for label in labels:
    lines.append(f"log_{label}={gcs.sign_url(key, a.bucket, f'runs/{run}/logs/{label}{tag}.log', expires=expires)}")
os.makedirs(a.out, exist_ok=True)
path = os.path.join(a.out, f"urls_{run}{tag}.env")
with open(path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"{path}: {len(lines)} signed URLs for gs://{a.bucket}/runs/{run}/ ({a.stride} phases, {len(labels)} box logs), valid {a.days:g} days")

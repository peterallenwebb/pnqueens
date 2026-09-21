#!/usr/bin/env python3
"""Pulls a run's records from the bucket into a local directory.

    sync.py N --stride S [--dir runs] [--bucket B] [--key FILE]

Downloads every object under runs/<N>_s<S>/ that is missing locally or whose
MD5 differs from the local copy's, into DIR/<N>_s<S>/ (logs under
DIR/<N>_s<S>/logs/), and checks each download against the bucket's MD5
before putting it in place. Prints what it fetched; then `pnqueens --sum
DIR/<N>_s<S>/rowsets_*.csv` checks the records, and status.py reports the
missing phases.
"""
import argparse, os
from concurrent.futures import ThreadPoolExecutor
import gcs

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("n", type=int)
ap.add_argument("--stride", type=int, required=True)
ap.add_argument("--dir", default=os.path.join(HERE, "..", "runs"))
ap.add_argument("--bucket", default="pnqueens-compute-records")
ap.add_argument("--key", default=os.path.join(HERE, "..", "..", "nqueens-signer.key"))
a = ap.parse_args()

run = f"{a.n}_s{a.stride}"
prefix = f"runs/{run}/"
token = gcs.access_token(gcs.load_key(a.key))
objects = gcs.list_objects(token, a.bucket, prefix)
local = os.path.join(a.dir, run)
os.makedirs(os.path.join(local, "logs"), exist_ok=True)
todo = []
for name, size, md5 in objects:
    path = os.path.join(local, name[len(prefix):])
    if gcs.needs_fetch(path, md5):
        todo.append((name, path, md5))
with ThreadPoolExecutor(8) as pool:
    fetched = sum(pool.map(lambda t: gcs.download(token, a.bucket, t[0], t[1], t[2]), todo))
print(f"gs://{a.bucket}/{prefix}: {len(objects)} objects, {len(todo)} fetched ({fetched / 1e6:.1f} MB) into {local}")

#!/usr/bin/env python3
"""Runs a worker's phases of a run on this box and pushes their records.

    worker.py N --stride S [--worker w --workers W | --phases a,b,c | --phases-file F]
              [--label L] [--tag T] [--redo] [--gpu i] [--binary B] [--dir ~/runs]
              [--push-env FILE] [--retries 2] [-- binary options...]

The binary's work list is cut into S phases (phase p: positions ≡ p mod S;
see `pnqueens --stride`); worker w of W runs the phases p ≡ w (mod W), or the
phases given. A phase's records, rowsets_<N>_s<S>_p<P>[_<T>].csv and .json, go
to DIR/<N>_s<S>/, written under a .part name and renamed when the binary has
finished, so a phase is recorded exactly when its .json exists; phases already
recorded are skipped unless --redo (verification reruns, with --tag and the
alternate options after `--`). After every phase the records and the worker's
log are pushed to the bucket with the signed URLs of DIR/push_urls[_T].env
(tools/presign.py). A pushed record gets a `.pushed` marker beside it; a
worker starts by queueing every local record without one, whoever recorded
it, retries the queue after every phase and at the end, and exits with
status 2 if anything is still unpushed, so no upload is forgotten across a
restart or an expired URL. The box's copy stays the primary record.

Every phase is cut from the binary's work map, whose fingerprint the run's
manifest (DIR/<N>_s<S>/manifest.json, written by the first worker) records:
a binary with a different work map refuses to run.
"""
import argparse, json, os, re, socket, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("n", type=int)
ap.add_argument("--stride", type=int, required=True)
ap.add_argument("--worker", type=int, default=0)
ap.add_argument("--workers", type=int, default=1)
ap.add_argument("--phases", help="a,b,c (overrides --worker/--workers)")
ap.add_argument("--phases-file", help="one phase per line (e.g. status.py's missing.txt)")
ap.add_argument("--label", default=socket.gethostname(), help="this box's name in logs and object names")
ap.add_argument("--tag", default="", help="suffix of the record names, for verification reruns")
ap.add_argument("--redo", action="store_true", help="run phases even if already recorded")
ap.add_argument("--gpu", type=int, help="pin to this CUDA device (CUDA_VISIBLE_DEVICES)")
ap.add_argument("--binary", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "target", "release", "pnqueens"))
ap.add_argument("--dir", default=os.path.expanduser("~/runs"))
ap.add_argument("--push-env", help="signed URLs (default DIR/push_urls[_TAG].env when present)")
ap.add_argument("--retries", type=int, default=2)
ap.add_argument("--cpu", action="store_true", help="count on the CPU (tests)")
a, extra = ap.parse_known_args()
# Everything after `--` (or anything unrecognized) goes to the binary: the
# alternate configuration of a verification rerun.
a.extra = [x for x in extra if x != "--"]

# Our own process group, so that killing it (fleet.py stop) also stops the
# phase's binary.
try:
    os.setsid()
except OSError:
    pass

run = f"{a.n}_s{a.stride}"
tag = f"_{a.tag}" if a.tag else ""
rundir = os.path.join(a.dir, run)
os.makedirs(os.path.join(rundir, "logs"), exist_ok=True)
logpath = os.path.join(rundir, "logs", f"{a.label}{tag}.log")


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(logpath, "a") as f:
        f.write(line + "\n")


# ---- which phases
if a.phases_file:
    phases = [int(x) for x in open(a.phases_file).read().split()]
elif a.phases:
    phases = [int(x) for x in a.phases.split(",")]
else:
    phases = list(range(a.worker, a.stride, a.workers))
if any(p < 0 or p >= a.stride for p in phases):
    sys.exit("phase out of range")

# ---- the work map and the run manifest
env = dict(os.environ)
if a.gpu is not None:
    env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
items_line = subprocess.run([a.binary, str(a.n), "--items"], capture_output=True, text=True, check=True).stdout.strip()
info = dict(kv.split("=") for kv in items_line.split())
workmap, items_total = info["workmap"], int(info["items"])
manifest_path = os.path.join(rundir, "manifest.json")
if os.path.exists(manifest_path):
    manifest = json.load(open(manifest_path))
    if manifest["workmap"] != workmap:
        sys.exit(f"REFUSED: the run's work map is {manifest['workmap']}, this binary's is {workmap}: do not mix their phases")
else:
    manifest = {"n": a.n, "stride": a.stride, "workmap": workmap, "items": items_total,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "by": a.label}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)

# ---- records push
push_env = a.push_env or os.path.join(a.dir, f"push_urls{tag}.env")
pending = []          # (url key, path) not yet pushed; a record's push is remembered by a .pushed marker


def urls():
    if not os.path.exists(push_env):
        return {}
    return dict(line.strip().split("=", 1) for line in open(push_env) if "=" in line)


def queue(key, path):
    if (key, path) not in pending:
        pending.append((key, path))


def push_pending():
    """Tries every queued upload; a record that arrives gets its marker."""
    table = urls()
    failures = []
    for key, path in list(pending):
        url = table.get(key)
        if url is None:
            failures.append(f"{key}: no URL in {push_env}")
            continue
        r = subprocess.run(["curl", "-sS", "-f", "--max-time", "300", "-o", "/dev/null", "-X", "PUT", "-T", path, url], capture_output=True, text=True)
        if r.returncode == 0:
            pending.remove((key, path))
            if path != logpath:
                open(path + ".pushed", "w").close()
        else:
            failures.append(f"{key}: {(r.stderr or '').strip()[:80]}")
    if failures:
        log(f"push: {len(failures)} upload(s) failed, will retry (first: {failures[0]})")


def queue_unpushed_records():
    """Every record of this run and tag on this box without a .pushed marker."""
    for name in sorted(os.listdir(rundir)):
        m = re.match(rf"rowsets_{run}_p(\d+){re.escape(tag)}\.(csv|json)$", name)
        if m and not os.path.exists(os.path.join(rundir, name + ".pushed")):
            queue(f"p{m.group(1)}_{m.group(2)}", os.path.join(rundir, name))


push_configured = bool(urls())
if push_configured:
    queue("manifest", manifest_path)
    queue_unpushed_records()
    if pending[1:]:
        log(f"push: {len(pending) - 1} record(s) from before were never pushed; queued")
    push_pending()
else:
    log(f"push: not configured ({push_env} is absent); records stay on this box")

# ---- the phases
def recorded(p):
    return os.path.exists(os.path.join(rundir, f"rowsets_{run}_p{p}{tag}.json"))


todo = phases if a.redo else [p for p in phases if not recorded(p)]
log(f"N={a.n} stride {a.stride}: worker {a.worker}/{a.workers} ({a.label}) owns {len(phases)} phases, {len(todo)} to do; "
    f"work map {workmap}, {items_total} items; binary {a.binary}{' with ' + ' '.join(a.extra) if a.extra else ''}")
walls, failed = [], []
for i, p in enumerate(todo):
    base = os.path.join(rundir, f"rowsets_{run}_p{p}{tag}")
    cmd = [a.binary, str(a.n), *([] if a.cpu else ["--cuda"]), "--stride", str(a.stride), "--phase", str(p),
           "--out", base + ".csv.part", "--manifest", base + ".json.part", *a.extra]
    ok = False
    for attempt in range(1 + a.retries):
        for stale in (base + ".csv.part", base + ".json.part", base + ".csv.pushed", base + ".json.pushed"):
            if os.path.exists(stale):
                os.unlink(stale)
        start = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        wall = time.time() - start
        if r.returncode == 0 and os.path.exists(base + ".json.part"):
            os.replace(base + ".csv.part", base + ".csv")
            os.replace(base + ".json.part", base + ".json")
            ok = True
            break
        log(f"phase {p}: attempt {attempt + 1} failed after {wall:.0f}s: {(r.stderr or r.stdout).strip()[-300:]}")
    if not ok:
        failed.append(p)
        continue
    walls.append(wall)
    m = json.load(open(base + ".json"))
    left = len(todo) - i - 1
    avg = sum(walls) / len(walls)
    log(f"phase {p}: total {m['weighted_total']} over {m['row_sets']} row sets in {wall:.0f}s | "
        f"{i + 1}/{len(todo)} done, avg {avg:.0f}s, ETA {left * avg / 3600:.2f} h")
    queue(f"p{p}_csv", base + ".csv")
    queue(f"p{p}_json", base + ".json")
    queue(f"log_{a.label}", logpath)
    push_pending()
queue(f"log_{a.label}", logpath)
push_pending()
unpushed = [key for key, path in pending if path != logpath] if push_configured else []
log(f"worker {a.worker}/{a.workers} ({a.label}) finished: {len(todo) - len(failed)} phases run, "
    f"{len(failed)} failed{' ' + str(failed) if failed else ''}, {len(unpushed)} record(s) not pushed{' ' + str(unpushed) if unpushed else ''}")
push_pending()
sys.exit(1 if failed else 2 if unpushed else 0)

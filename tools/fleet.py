#!/usr/bin/env python3
"""Runs the operator's steps across the boxes of boxes.txt (PORT HOST [LABEL]
per line; labels default to box01, box02, ...), in parallel over ssh. ssh
options such as an identity file come from PNQUEENS_SSH_OPTS.

    fleet.py ship                                   tools/ship.sh on every box; the binary hashes must agree
    fleet.py ladder                                 n = 16..22 on every box: every count ok
    fleet.py push-urls N --stride S [--tag T]       presign (unless the file exists) and ship the URLs
    fleet.py start N --stride S [-- worker options] worker i of W on box i
    fleet.py run-phases N --stride S --box LABEL --phases a,b,c [--tag T --redo] [-- binary options]
    fleet.py stop N --stride S                      kill the workers
    fleet.py status N --stride S                    liveness per box, then status.py
    fleet.py cmd 'shell command'                    on every box, in the checkout
"""
import argparse, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15", *os.environ.get("PNQUEENS_SSH_OPTS", "").split()]


def boxes(path):
    out = []
    for line in open(path):
        parts = line.split()
        if parts and not parts[0].startswith("#"):
            out.append((parts[0], parts[1], parts[2] if len(parts) > 2 else f"box{len(out) + 1:02d}"))
    return out


def ssh(box, command, timeout=None):
    """Runs a command on a box: its label, exit code and output (stderr, where
    login banners go, only when the command failed)."""
    port, host, label = box
    r = subprocess.run(["ssh", "-p", port, *SSH_OPTS, host, command], capture_output=True, text=True, timeout=timeout)
    return label, r.returncode, (r.stdout + (r.stderr if r.returncode else "")).strip()


def scp(box, src, dst):
    port, host, label = box
    return subprocess.run(["scp", "-P", port, *SSH_OPTS, src, f"{host}:{dst}"], capture_output=True, text=True).returncode == 0


def everywhere(fleet, fn):
    with ThreadPoolExecutor(len(fleet)) as pool:
        for label, code, out in pool.map(fn, fleet):
            for line in out.splitlines() or ["(no output)"]:
                print(f"{label}: {line}")
            if code:
                print(f"{label}: EXIT {code}")
    sys.stdout.flush()


ap = argparse.ArgumentParser()
ap.add_argument("command")
ap.add_argument("n", nargs="?", help="board size (the shell command, for cmd)")
ap.add_argument("--stride", type=int)
ap.add_argument("--hosts", default=os.path.join(ROOT, "boxes.txt"))
ap.add_argument("--tag", default="")
ap.add_argument("--box")
ap.add_argument("--phases")
ap.add_argument("--redo", action="store_true")
ap.add_argument("--renew", action="store_true", help="push-urls: presign again even if the file exists")
ap.add_argument("rest", nargs="*")
a = ap.parse_args()
fleet = boxes(a.hosts)
need_run = a.command in ("push-urls", "start", "run-phases", "stop", "status")
if need_run and (a.n is None or a.stride is None):
    sys.exit(f"{a.command} needs N and --stride")
run = f"{a.n}_s{a.stride}" if need_run else None
if a.command == "cmd":
    a.rest = [a.n or "", *a.rest]

if a.command == "ship":
    def ship(box):
        port, host, label = box
        r = subprocess.run(["sh", os.path.join(HERE, "ship.sh"), port, host], capture_output=True, text=True)
        return label, r.returncode, (r.stdout + (r.stderr if r.returncode else "")).strip()
    everywhere(fleet, ship)
elif a.command == "ladder":
    cmd = "cd pnqueens && d=$(mktemp -d) && for n in 16 17 18 19 20 21 22; do ./target/release/pnqueens $n --cuda --out $d/$n.csv 2>&1 | grep -E '^Q\\(|error'; done; rm -rf $d"
    everywhere(fleet, lambda b: ssh(b, cmd, timeout=600))
elif a.command == "push-urls":
    tag = f"_{a.tag}" if a.tag else ""
    env_file = os.path.join(ROOT, "push", f"urls_{run}{tag}.env")
    if a.renew or not os.path.exists(env_file):
        subprocess.run([sys.executable, os.path.join(HERE, "presign.py"), str(a.n), "--stride", str(a.stride), "--hosts", a.hosts, "--tag", a.tag], check=True)
    def ship_urls(box):
        ok = scp(box, env_file, f"runs/push_urls{tag}.env")
        return box[2], 0 if ok else 1, "URLs shipped" if ok else "scp failed"
    everywhere(fleet, ship_urls)
elif a.command == "start":
    W = len(fleet)
    def start(box, i):
        extra = " ".join(a.rest)
        return ssh(box, f"cd pnqueens && sh tools/start_worker.sh {a.n} {a.stride} --worker {i} --workers {W} --label {box[2]} {extra}")
    with ThreadPoolExecutor(W) as pool:
        for label, code, out in pool.map(lambda ib: start(ib[1], ib[0]), enumerate(fleet)):
            print(f"{label}: {out}")
elif a.command == "run-phases":
    box = next((b for b in fleet if b[2] == a.box), None) or sys.exit(f"no box labelled {a.box}")
    tag = f"--tag {a.tag}" if a.tag else ""
    extra = (" -- " + " ".join(a.rest)) if a.rest else ""
    label, code, out = ssh(box, f"cd pnqueens && sh tools/start_worker.sh {a.n} {a.stride} --phases {a.phases} --label {box[2]} {tag} {'--redo' if a.redo else ''}{extra}")
    print(f"{label}: {out}")
elif a.command == "stop":
    everywhere(fleet, lambda b: ssh(b, f"kill $(cat runs/worker_{run}.pid) 2>/dev/null && echo stopped || echo 'not running'"))
elif a.command == "status":
    everywhere(fleet, lambda b: ssh(b, f"last=$(tail -n 1 runs/{run}/logs/{b[2]}.log 2>/dev/null); "
                                       f"if kill -0 $(cat runs/worker_{run}.pid 2>/dev/null) 2>/dev/null; then echo -n 'ALIVE  '; "
                                       f"elif echo \"$last\" | grep -q finished; then echo -n 'DONE   '; else echo -n 'DEAD   '; fi; echo \"${{last:-(no log)}}\""))
    subprocess.run([sys.executable, os.path.join(HERE, "status.py"), str(a.n), "--stride", str(a.stride)])
elif a.command == "cmd":
    everywhere(fleet, lambda b: ssh(b, "cd pnqueens && " + " ".join(a.rest)))
else:
    sys.exit(f"unknown command {a.command}")

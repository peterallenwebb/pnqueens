#!/usr/bin/env python3
"""Exercises worker.py and status.py end to end on the CPU at n = 12, with no
bucket and no ssh: two workers of a stride, a replacement worker for the
phases one of them did not finish, a verification rerun, the refusal of a
foreign work map, and the binary's --sum saying ok.

    python3 tools/test_tools.py [--binary B]
"""
import argparse, json, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--binary", default=os.path.join(HERE, "..", "target", "release", "pnqueens"))
a = ap.parse_args()
binary = os.path.abspath(a.binary)
tmp = tempfile.mkdtemp()
N, S = "12", "4"
run = f"{N}_s{S}"


def worker(*args, check=True):
    return subprocess.run([sys.executable, os.path.join(HERE, "worker.py"), N, "--stride", S, "--dir", tmp, "--binary", binary, "--cpu", *args],
                          capture_output=True, text=True, check=check)


def status():
    return subprocess.run([sys.executable, os.path.join(HERE, "status.py"), N, "--stride", S, "--dir", tmp, "--no-sync", "--binary", binary],
                          capture_output=True, text=True, check=True).stdout


try:
    # Worker 0 of 2 runs phases 0 and 2; worker 1 of 2 runs only phase 1 (as if it died before 3).
    worker("--worker", "0", "--workers", "2", "--label", "a")
    worker("--phases", "1", "--label", "b")
    out = status()
    assert "3 of 4 phases recorded" in out and "1 missing" in out and "PARTIAL" in out, out
    missing = open(os.path.join(tmp, run, "missing.txt")).read().split()
    assert missing == ["3"], missing
    # A replacement box takes the missing phases; the run is complete.
    worker("--phases-file", os.path.join(tmp, run, "missing.txt"), "--label", "c")
    out = status()
    assert "4 of 4 phases recorded" in out and "Q(12): 14200 ok" in out and "mod-8 check: PASS" in out, out
    # A verification rerun of phase 2 records it again; the duplicates agree.
    worker("--phases", "2", "--redo", "--tag", "verify", "--label", "a")
    out = status()
    assert "(1 more than once)" in out and "recorded more than once, all in agreement" in out and "Q(12): 14200 ok" in out, out
    # Resume: nothing left to do for worker 0.
    r = worker("--worker", "0", "--workers", "2", "--label", "a")
    assert "0 to do" in r.stdout, r.stdout
    # A foreign work map is refused.
    m = json.load(open(os.path.join(tmp, run, "manifest.json")))
    m["workmap"] = "0000000000000000"
    json.dump(m, open(os.path.join(tmp, run, "manifest.json"), "w"))
    r = worker("--phases", "0", "--redo", check=False)
    assert r.returncode != 0 and "REFUSED" in (r.stdout + r.stderr), r.stdout + r.stderr
    print("all tool tests passed")
finally:
    shutil.rmtree(tmp)

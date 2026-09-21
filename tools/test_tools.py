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


def ship_tests():
    """ship.sh against a fake ssh that runs the remote script locally, with a
    fake cargo: a failing build must fail the ship even though an old
    executable is there; a build whose executable reports another commit
    must fail; a good build must succeed and print the hash."""
    clone = os.path.join(tmp, "clone")
    subprocess.run(["git", "clone", "-q", os.path.join(HERE, ".."), clone], check=True)
    shutil.copy(os.path.join(HERE, "ship.sh"), os.path.join(clone, "tools", "ship.sh"))
    subprocess.run(["git", "-C", clone, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "test"], check=True)
    commit = subprocess.run(["git", "-C", clone, "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True).stdout.strip()
    box = os.path.join(tmp, "box")
    fakebin = os.path.join(tmp, "fakebin")
    os.makedirs(os.path.join(box, ".cargo", "bin"))
    os.makedirs(fakebin)
    # The fake ssh ignores its options and runs the script in the fake box's home.
    with open(os.path.join(fakebin, "ssh"), "w") as f:
        f.write('#!/bin/sh\nwhile [ "$1" != "${1#-}" ]; do shift 2; done\nshift\ncd "$HOME" && exec sh -c "$1"\n')
    os.chmod(os.path.join(fakebin, "ssh"), 0o755)
    env = dict(os.environ, PATH=fakebin + ":" + os.environ["PATH"], HOME=box)

    def cargo(script):
        with open(os.path.join(box, ".cargo", "bin", "cargo"), "w") as f:
            f.write("#!/bin/sh\n" + script)
        os.chmod(os.path.join(box, ".cargo", "bin", "cargo"), 0o755)

    def ship():
        return subprocess.run(["sh", os.path.join(clone, "tools", "ship.sh"), "22", "fakehost"], capture_output=True, text=True, env=env, cwd=clone)

    # 1. A stale executable and a failing compiler.
    os.makedirs(os.path.join(box, "pnqueens", "target", "release"))
    with open(os.path.join(box, "pnqueens", "target", "release", "pnqueens"), "w") as f:
        f.write("#!/bin/sh\necho pnqueens commit stale binary 0000\n")
    os.chmod(os.path.join(box, "pnqueens", "target", "release", "pnqueens"), 0o755)
    cargo('echo "error: could not compile" >&2; exit 101')
    r = ship()
    assert r.returncode != 0 and "BUILD FAILED" in r.stdout + r.stderr, (r.stdout, r.stderr)
    assert not os.path.exists(os.path.join(box, "pnqueens", "target", "release", "pnqueens")), "the stale executable survived"
    # 2. A build whose executable reports another commit.
    cargo('mkdir -p target/release && printf "#!/bin/sh\necho pnqueens commit deadbeefdead binary 1111\n" > target/release/pnqueens && chmod +x target/release/pnqueens')
    r = ship()
    assert r.returncode != 0 and "not commit " + commit in r.stdout + r.stderr, (r.stdout, r.stderr)
    # 3. A good build.
    cargo(f'mkdir -p target/release && printf "#!/bin/sh\necho pnqueens commit {commit} binary 2222\n" > target/release/pnqueens && chmod +x target/release/pnqueens')
    r = ship()
    assert r.returncode == 0 and r.stdout.strip().endswith(f"commit {commit} binary 2222"), (r.stdout, r.stderr)


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
    ship_tests()
    print("all tool tests passed")
finally:
    shutil.rmtree(tmp)

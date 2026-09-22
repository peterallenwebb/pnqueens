#!/bin/sh
# Starts a worker on this box, detached, with its pid in ~/runs/worker_<N>_s<S>.pid
# and its output in ~/runs/worker_<N>_s<S>.log (the same command resumes it).
# The worker makes itself a process group, so `kill -- -PID` stops it and
# the phase it is running (fleet.py stop).
#
#     sh tools/start_worker.sh N STRIDE [worker.py arguments...]
set -e
n=$1; stride=$2; shift 2
cd "$(dirname "$0")/.."
mkdir -p "$HOME/runs"
log="$HOME/runs/worker_${n}_s${stride}.log"; pid="$HOME/runs/worker_${n}_s${stride}.pid"
if [ -f "$pid" ] && kill -0 "$(cat "$pid")" 2>/dev/null; then echo "a worker is running (pid $(cat "$pid"))"; exit 1; fi
nohup python3 tools/worker.py "$n" --stride "$stride" "$@" >> "$log" 2>&1 < /dev/null &
echo $! > "$pid"
echo "worker started, pid $(cat "$pid"), log $log"

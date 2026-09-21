#!/bin/sh
# Ships the committed tree to a box and builds it there.
#
#     tools/ship.sh PORT HOST
#
# Refuses a dirty tree, so the commit recorded in the binary is the commit
# that was built. The checkout goes to ~/pnqueens on the box (its src, kernel
# and tools replaced; records live in ~/runs, outside it); Rust is installed
# if missing, the pinned toolchain by rustup. Prints the binary's SHA-256.
# ssh options can be added with PNQUEENS_SSH_OPTS (e.g. an identity file).
set -e
port=$1; host=$2
[ -n "$port" ] && [ -n "$host" ] || { echo "usage: tools/ship.sh PORT HOST" >&2; exit 2; }
cd "$(dirname "$0")/.."
if [ -n "$(git status --porcelain)" ]; then echo "refusing to ship a dirty tree" >&2; exit 1; fi
commit=$(git rev-parse --short=12 HEAD)
git archive --format=tar HEAD | ssh -p "$port" $PNQUEENS_SSH_OPTS "$host" "set -e
  mkdir -p pnqueens runs && cd pnqueens && rm -rf src kernel tools && tar -x
  [ -x \$HOME/.cargo/bin/cargo ] || curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null
  PNQUEENS_COMMIT=$commit \$HOME/.cargo/bin/cargo build --release --features cuda 2>&1 | grep -E 'error|warning: unused|Finished' || true
  [ -x target/release/pnqueens ] || { echo 'BUILD FAILED'; exit 1; }
  echo \"\$(hostname) commit $commit binary \$(sha256sum target/release/pnqueens | cut -c1-64)\""

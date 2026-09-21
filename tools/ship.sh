#!/bin/sh
# Ships the committed tree to a box and builds it there.
#
#     tools/ship.sh PORT HOST
#
# Refuses a dirty tree, so the commit recorded in the binary is the commit
# that was built. The checkout goes to ~/pnqueens on the box (its src, kernel
# and tools replaced; records live in ~/runs, outside it); Rust is installed
# if missing, the pinned toolchain by rustup. The old executable is removed
# before the build, the build fails on cargo's exit status, and the new
# executable must report the shipped commit: only then does the script
# succeed, printing "HOST commit C binary SHA256". ssh options can be added
# with PNQUEENS_SSH_OPTS (e.g. an identity file).
set -e
port=$1; host=$2
[ -n "$port" ] && [ -n "$host" ] || { echo "usage: tools/ship.sh PORT HOST" >&2; exit 2; }
cd "$(dirname "$0")/.."
if [ -n "$(git status --porcelain)" ]; then echo "refusing to ship a dirty tree" >&2; exit 1; fi
commit=$(git rev-parse --short=12 HEAD)
git archive --format=tar HEAD | ssh -p "$port" $PNQUEENS_SSH_OPTS "$host" "set -e
  mkdir -p pnqueens runs && cd pnqueens && rm -rf src kernel tools target/release/pnqueens && tar -x
  [ -x \$HOME/.cargo/bin/cargo ] || curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null
  if ! PNQUEENS_COMMIT=$commit \$HOME/.cargo/bin/cargo build --release --features cuda > build.log 2>&1; then
    tail -n 30 build.log; echo 'BUILD FAILED'; exit 1
  fi
  version=\$(./target/release/pnqueens --version) || { echo 'the built executable does not run'; exit 1; }
  case \"\$version\" in 'pnqueens commit $commit binary '*) ;; *) echo \"the built executable reports '\$version', not commit $commit\"; exit 1;; esac
  echo \"\$(hostname) commit $commit binary \${version##* }\""

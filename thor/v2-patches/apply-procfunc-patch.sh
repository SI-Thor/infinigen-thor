#!/usr/bin/env bash
# apply-procfunc-patch.sh - apply the procfunc 0.37.0 patches into the
# venv's site-packages: round 2's ValueProxy tracer patch and round 3's
# subtree-memo module (the engine-side splice). Idempotent: re-running is a
# no-op.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

SITE=".venv/lib/python3.11/site-packages"
SP="$SITE/procfunc"

if ! grep -q "class ValueProxy" "$SP/compute_graph/proxy.py" 2>/dev/null; then
  (cd "$SITE" && patch -p1 --forward --batch < "$HERE/patches/procfunc-0.37.0/procfunc-value-proxy.patch")
  echo "round-2 procfunc patch applied"
else
  echo "round-2 procfunc patch already applied"
fi

if [ ! -f "$SP/subtree_memo.py" ]; then
  (cd "$SITE" && patch -p1 --forward --batch < "$HERE/patches/procfunc-0.37.0/procfunc-subtree-memo.patch")
  echo "round-3 subtree-memo patch applied"
else
  echo "round-3 subtree-memo patch already applied"
fi

#!/usr/bin/env bash
# thor/p3-fine-runs.sh - phase 3 E1: is fine_terrain deterministic when the
# input scene is FIXED?  Runs --task fine_terrain (no populate, so the only
# thing that varies is the terrain generation + mesher itself) N times per
# device from outputs/p2/desert/coarse, then compares the saved blends
# exactly (terrain objects are canonicalized by make_unique at export, so
# any difference is geometric, not ordering).
#
# Env: N (default 3), DEVICES (default "cpu cuda"), TAG (default E1).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv-p2/bin/python"
CFG="desert.gin performance/dev.gin disable_assets/no_creatures.gin performance/fast_terrain_assets.gin"
IN="$REPO/outputs/p2/desert/coarse"
N="${N:-3}"
DEVICES="${DEVICES:-cpu cuda}"
TAG="${TAG:-E1}"
mkdir -p "$REPO/outputs/p3"

for dev in $DEVICES; do
  for r in $(seq 1 "$N"); do
    out="$REPO/outputs/p3/${TAG}-fine-$dev-r$r"
    rm -rf "$out"
    mkdir -p "$out"
    extra=()
    if [ "$dev" = cuda ]; then extra+=(-p "Terrain.device=cuda"); fi
    bash "$REPO/thor/run-timed.sh" "p3-${TAG}-fine-$dev-r$r" \
      "$PY" -m infinigen_examples.generate_nature --seed 0 -g $CFG \
      --task fine_terrain --input_folder "$IN" --output_folder "$out" \
      "${extra[@]}" || { echo "RUN FAILED fine-$dev-r$r"; exit 1; }
  done
done

for dev in $DEVICES; do
  blends=$(ls "$REPO"/outputs/p3/${TAG}-fine-$dev-r*/scene.blend 2>/dev/null)
  [ -z "$blends" ] && continue
  "$PY" "$REPO/thor/p3_compare.py" "SAME-DEVICE $dev (N=$N)" $blends || true
done
if [ "$DEVICES" = "cpu cuda" ]; then
  "$PY" "$REPO/thor/p3_compare.py" "CROSS-DEVICE cpu-r1 vs cuda-r1" \
    "$REPO/outputs/p3/${TAG}-fine-cpu-r1/scene.blend" \
    "$REPO/outputs/p3/${TAG}-fine-cuda-r1/scene.blend" || true
fi
echo "[p3-fine-runs] done"

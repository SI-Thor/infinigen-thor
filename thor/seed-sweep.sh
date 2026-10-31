#!/usr/bin/env bash
# thor/seed-sweep.sh [seeds-per-scene] [scene...] - an overnight sweep of Infinigen Nature scenes on the Thor.
#
# Why: phase 1 timed one desert scene once. A sweep across scene types and seeds gives the spread
# (p50/p95 per stage per scene type) that the night scheduler needs, and a stock of real rendered
# scenes for the critic's calibration set. It also keeps the GPU busy while the agents work on the core.
#
# For each scene type and seed it runs coarse -> populate+fine_terrain -> render through
# thor/run-timed.sh (wall time, peak RSS, tegrastats, GPU samples under outputs/logs/sweep-*),
# keeps the rendered frames, and deletes the large intermediate .blend files to save disk.
# A scene that fails is logged and skipped; the sweep continues.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
N="${1:-3}"; shift || true
SCENES=("$@"); [ ${#SCENES[@]} -eq 0 ] && SCENES=(desert forest mountain river coast plain canyon arctic)
OUT="$REPO/outputs/sweep"; mkdir -p "$OUT"
LOG="$OUT/sweep.tsv"; [ -f "$LOG" ] || printf "scene\tseed\tstage\tstatus\tseconds\n" > "$LOG"
PY="$REPO/.venv/bin/python"
for seed in $(seq 1 "$N"); do
  for scene in "${SCENES[@]}"; do
    base="$OUT/$scene-s$seed"; mkdir -p "$base"
    [ -f "$base/DONE" ] && continue
    ok=1
    for stage in coarse fine render; do
      case $stage in
        coarse) args=(--task coarse --output_folder "$base/coarse");;
        fine)   args=(--task populate fine_terrain --input_folder "$base/coarse" --output_folder "$base/fine");;
        render) args=(--task render --input_folder "$base/fine" --output_folder "$base/frames");;
      esac
      t0=$(date +%s)
      if bash thor/run-timed.sh "sweep-$scene-s$seed-$stage" "$PY" -m infinigen_examples.generate_nature \
           --seed "$seed" -g "$scene.gin" simple.gin "${args[@]}" >/dev/null 2>&1; then st=ok; else st=FAIL; ok=0; fi
      printf "%s\t%s\t%s\t%s\t%s\n" "$scene" "$seed" "$stage" "$st" "$(( $(date +%s) - t0 ))" >> "$LOG"
      [ $ok -eq 1 ] || break
    done
    find "$base" -name "*.blend*" -delete 2>/dev/null
    [ $ok -eq 1 ] && touch "$base/DONE"
  done
done
echo "sweep complete: $LOG"

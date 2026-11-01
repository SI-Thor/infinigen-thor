#!/usr/bin/env bash
# thor/sweep-worker.sh [from-seed] [to-seed] [scene...] - a parallel worker for the seed sweep (mkdir lock per scene-seed, so several workers share one range safely).
# Derived from - an overnight sweep of Infinigen Nature scenes on the Thor.
#
# Why: phase 1 timed one desert scene once. A sweep across scene types and seeds gives the spread
# (p50/p95 per stage per scene type) that the night scheduler needs, and a stock of real rendered
# scenes for the critic's calibration set. It also keeps the GPU busy while the agents work on the core.
#
# For each scene type and seed it runs coarse -> populate+fine_terrain -> render through
# thor/run-timed.sh (wall time, peak RSS, tegrastats, GPU samples under outputs/logs/sweep-*),
# keeps the rendered frames, and deletes the large intermediate .blend files to save disk.
# A scene that fails is logged and skipped; the sweep continues.
#
# 2026-10-31 (round 8): every stage now runs under the box's admission daemon, thor-admit, as
# **study-tier** work (builder-ide tools/thor-admit, branch thor-admit). The stage's label is
# infinigen.<scene>.<stage>, so the daemon learns a peak per biome and task and later stages can
# run --peak auto (it reserves max(p95, last run); until a label has three runs the daemon refuses
# auto, and the explicit bootstrap peaks below stand in). The daemon then admits a stage only when
# the box has room — several sweep workers (w5, w7, w8...) can be started at once without the
# 22:09 fall to 8 GB free, and live work pre-empts them. The mkdir lock per scene-seed stays: it
# keeps two workers off the same scene; the daemon is what keeps them off the same memory.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
FROM="${1:-4}"; TO="${2:-6}"; shift 2 || true
SCENES=("$@"); [ ${#SCENES[@]} -eq 0 ] && SCENES=(desert forest mountain river coast plain canyon arctic)
OUT="$REPO/outputs/sweep"; mkdir -p "$OUT"
LOG="$OUT/sweep.tsv"; [ -f "$LOG" ] || printf "scene\tseed\tstage\tstatus\tseconds\n" > "$LOG"
PY="$REPO/.venv/bin/python"

# The admission daemon's CLI. The default is the builder-ide checkout's release build; override
# with THOR_ADMIT_BIN. THOR_ADMIT_SOCKET selects a non-default daemon, as in the crate's CLI.
ADMIT="${THOR_ADMIT_BIN:-$HOME/lm-wt/thor-admit/tools/thor-admit/target/release/thor-admit}"
admit_socket=()
[ -n "${THOR_ADMIT_SOCKET:-}" ] && admit_socket=(--socket "$THOR_ADMIT_SOCKET")

# Bootstrap peaks: the explicit --peak a label's first runs use, before the daemon has the three
# runs of history that --peak auto needs. The defaults cover the worst the sweep's stages measured
# on this box (renders of 23-24 GB on 2026-10-31); override per stage without editing the script.
peak_for() {
  case "$1" in
    coarse) echo "${THOR_ADMIT_SWEEP_PEAK_COARSE:-16G}";;
    fine)   echo "${THOR_ADMIT_SWEEP_PEAK_FINE:-24G}";;
    render) echo "${THOR_ADMIT_SWEEP_PEAK_RENDER:-24G}";;
  esac
}

# Run one stage under a study-tier lease. Tries the learned peak first; when the daemon refuses
# auto (too few runs — the refusal names the label's history, and it happens before any work
# starts), the bootstrap peak stands in. Any other failure is the stage's own and is not retried:
# a failed command must not run twice.
admit_run() {
  local label="$1" peak="$2"; shift 2
  local errors="$OUT/.admit-$label.err"
  if "$ADMIT" run "${admit_socket[@]}" --tier study --peak auto --label "$label" -- "$@" 2>"$errors"; then
    return 0
  fi
  if grep -qE 'no history for the label|--peak auto needs 3' "$errors"; then
    "$ADMIT" run "${admit_socket[@]}" --tier study --peak "$peak" --label "$label" -- "$@" 2>>"$errors"
  else
    return 1
  fi
}

for seed in $(seq "$FROM" "$TO"); do
  for scene in "${SCENES[@]}"; do
    base="$OUT/$scene-s$seed"; mkdir -p "$base"
    [ -f "$base/DONE" ] && continue; mkdir "$base.lock" 2>/dev/null || continue
    ok=1
    for stage in coarse fine render; do
      case $stage in
        coarse) args=(--task coarse --output_folder "$base/coarse");;
        fine)   args=(--task populate fine_terrain --input_folder "$base/coarse" --output_folder "$base/fine");;
        render) args=(--task render --input_folder "$base/fine" --output_folder "$base/frames");;
      esac
      t0=$(date +%s)
      if admit_run "infinigen.$scene.$stage" "$(peak_for "$stage")" \
           bash thor/run-timed.sh "sweep-$scene-s$seed-$stage" "$PY" -m infinigen_examples.generate_nature \
             --seed "$seed" -g "$scene.gin" simple.gin "${args[@]}" >/dev/null 2>&1; then st=ok; else st=FAIL; ok=0; fi
      printf "%s\t%s\t%s\t%s\t%s\n" "$scene" "$seed" "$stage" "$st" "$(( $(date +%s) - t0 ))" >> "$LOG"
      [ $ok -eq 1 ] || break
    done
    find "$base" -name "*.blend*" -delete 2>/dev/null
    [ $ok -eq 1 ] && touch "$base/DONE"
  done
done
echo "sweep complete: $LOG"

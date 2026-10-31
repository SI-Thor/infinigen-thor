#!/usr/bin/env bash
# thor/concurrent-renders.sh N FINE_DIR PY - render the SAME fine scene N ways
# concurrently (Infinigen render task, Cycles CUDA), measure wall time and
# box headroom, report throughput vs the N=1 run.
#
# Why: phase 1 saw one GPU render at 98% util with ~3 GB RSS and 122 GB of
# unified memory idle. If N concurrent renders keep aggregate throughput
# near N x serial on this box, overnight dataset generation should run
# packed, not queued.
#
# Each worker gets its own COPY of the fine folder (the render task touches
# its input's assets on some scene types; isolation keeps that harmless).
# tegrastats samples the whole batch into outputs/logs/p2-conc-N.stats.
set -uo pipefail

N="$1"; FINE="$2"; PY="$3"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/outputs/p2/conc-$N"
LOGS="$REPO/outputs/logs"
mkdir -p "$OUT" "$LOGS"

/usr/bin/tegrastats --interval 1000 --logfile "$LOGS/p2-conc-$N.stats" & TSTATS=$!

PIDS=()
T0=$(date +%s)
for i in $(seq 1 "$N"); do
  IN="$OUT/input-$i"
  rm -rf "$IN"; cp -r "$FINE" "$IN"
  (
    bash "$REPO/thor/run-timed.sh" "p2-conc-$N-job$i" "$PY" \
      -m infinigen_examples.generate_nature --seed 0 -g desert.gin simple.gin \
      --task render --input_folder "$IN" --output_folder "$OUT/frames-$i" \
      > /dev/null 2>&1
  ) &
  PIDS+=($!)
done

RC=0
for p in "${PIDS[@]}"; do wait "$p" || RC=1; done
WALL=$(( $(date +%s) - T0 ))
kill "$TSTATS" 2>/dev/null

PEAK_RAM=$(grep -o "RAM [0-9]*/" "$LOGS/p2-conc-$N.stats" | tr -dc "0-9\n/" | cut -d/ -f1 | sort -n | tail -1)
echo "concurrency=$N wall=${WALL}s throughput=$(awk "BEGIN{printf \"%.2f\", $N/$WALL}") scenes/min peakSystemRAM=${PEAK_RAM:-?}MB rc=$RC"

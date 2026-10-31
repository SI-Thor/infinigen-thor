#!/usr/bin/env bash
# thor/run-timed.sh <name> <cmd...> - run a stage with wall time, peak RSS,
# and Jetson counters (tegrastats: RAM/power; nvidia-smi: GPU util/mem).
#
# Every "works" claim in this spike carries its proof; this wrapper produces
# the proof artifacts under outputs/logs/<name>.* :
#   .out      full stdout+stderr of the stage
#   .time     GNU time report (elapsed, peak RSS, CPU%)
#   .stats    tegrastats sample while running (1 Hz)
#   .gpu      nvidia-smi samples (GPU util / mem, 1 Hz)
#   .summary  one-line summary echoed to stdout at the end

set -uo pipefail

NAME="$1"; shift
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="$REPO/outputs/logs"
mkdir -p "$LOGS"

echo "[run-timed] $NAME: $*"
STATS_PID=""
GPU_PID=""

/usr/bin/tegrastats --interval 1000 --logfile "$LOGS/$NAME.stats" & STATS_PID=$!
(
  while true; do
    nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu \
      --format=csv,noheader 2>/dev/null
    sleep 1
  done
) > "$LOGS/$NAME.gpu" & GPU_PID=$!

START=$(date +%s)
/usr/bin/time -v -o "$LOGS/$NAME.time" "$@" > "$LOGS/$NAME.out" 2>&1
RC=$?
END=$(date +%s)

kill "$STATS_PID" "$GPU_PID" 2>/dev/null
wait 2>/dev/null

ELAPSED=$((END - START))
RSS_KB=$(awk -F': ' '/Maximum resident set size/ {print $2}' "$LOGS/$NAME.time")
CPU=$(awk -F': ' '/Percent of CPU/ {print $2}' "$LOGS/$NAME.time")
PEAK_RAM_MB=$(grep -o "RAM [0-9]*/" "$LOGS/$NAME.stats" | tr -dc "0-9\n/" | cut -d/ -f1 | sort -n | tail -1)
MAX_GPU_UTIL=$(awk -F', ' '{gsub("%","",$1); if ($1+0>m) m=$1+0} END {print m}' "$LOGS/$NAME.gpu")
MAX_GPU_MEM=$(awk -F', ' '{gsub(" MiB","",$2); if ($2+0>m) m=$2+0} END {print m}' "$LOGS/$NAME.gpu")

SUMMARY="$NAME rc=$RC elapsed=${ELAPSED}s peakRSS=$((RSS_KB/1024))MB cpu='${CPU:-}' peakSystemRAM='${PEAK_RAM_MB:-}'MB maxGPUutil='${MAX_GPU_UTIL:-}'% maxGPUmem='${MAX_GPU_MEM:-}'MiB"
echo "$SUMMARY" | tee "$LOGS/$NAME.summary"
exit $RC

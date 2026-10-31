#!/usr/bin/env bash
# thor/probe-thor.sh [venv] - capability gate for Infinigen on the Jetson Thor.
#
# Why: phase 1 failed twice on box capabilities that no install log surfaced
# until something broke an hour later (nvcc's default arch emits sm_75 cubins
# the sm_110 GPU refuses to load; the first bpy wheel had no denoiser). This
# probe measures each capability the pipeline depends on and exits non-zero if
# one is missing, so a bad environment is caught before a render run.
#
# Checks (PASS/FAIL each; any FAIL => exit 1):
#   cuda        CUDA toolkit + GPU visible, compute capability reported
#   umem        cudaMallocManaged works (unified memory; Jetson is inherent,
#               but the probe proves the runtime path, not the datasheet)
#   nvcc-arch   nvcc's DEFAULT arch (informational: phase 1 needed
#               -arch=native because nvcc 13 defaults to sm_75 here)
#   denoiser    bpy wheel has OpenImageDenoise and Cycles accepts it
#   cycles      Cycles lists a CUDA device in the probed venv
#   sm110       every CUDA-bearing .so under the venv's bpy and the tree's
#               terrain libs carries sm_110 SASS (cuobjdump --list-elf)
set -uo pipefail
shopt -s globstar nullglob

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${1:-$REPO/.venv-p2}"
NVCC=/usr/local/cuda/bin/nvcc
CUOBJDUMP=/usr/local/cuda/bin/cuobjdump
FAILS=0

probe() { printf '%-12s %s\n' "$1" "$2"; }
fail()  { probe "$1" "FAIL: $2"; FAILS=$((FAILS+1)); }
pass()  { probe "$1" "PASS: $2"; }

# --- cuda: GPU + toolkit -----------------------------------------------------
CC=""
if command -v nvidia-smi >/dev/null; then
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')
fi
if [[ -z "$CC" ]]; then
  CC=$($NVCC --list-gpu-arch 2>/dev/null | tail -n1 | sed 's/sm_//')  # fallback hint only
fi
if [[ -n "$CC" && -x "$NVCC" ]]; then
  pass "cuda" "GPU compute capability ${CC}, nvcc $($NVCC --version | tail -n1 | grep -o 'release [0-9.]*')"
else
  fail "cuda" "no GPU compute capability or nvcc missing"
fi

# --- umem: cudaMallocManaged --------------------------------------------------
TMP=$(mktemp -d)
cat > "$TMP/umem.cu" <<'EOF'
#include <cuda_runtime.h>
#include <cstdio>
int main() {
  int* p = nullptr;
  if (cudaMallocManaged(&p, 1024) != cudaSuccess) { puts("no"); return 1; }
  p[0] = 42;                         // CPU write
  cudaDeviceSynchronize();
  puts(p[0] == 42 ? "yes" : "no");   // read back after sync
  return 0;
}
EOF
if "$NVCC" -arch=native "$TMP/umem.cu" -o "$TMP/umem" 2>/dev/null && "$TMP/umem" 2>/dev/null | grep -q yes; then
  MEM=$(awk '/MemTotal/ {printf "%.0f", $2/1048576}' /proc/meminfo)
  pass "umem" "cudaMallocManaged read/write OK; ${MEM} GB unified RAM visible to Linux"
else
  fail "umem" "cudaMallocManaged probe failed"
fi

# --- nvcc-arch: the default (no -arch) target ---------------------------------
cat > "$TMP/empty.cu" <<'EOF'
__global__ void k() {}
int main() { return 0; }
EOF
"$NVCC" -cubin "$TMP/empty.cu" -o "$TMP/empty.cubin" 2>/dev/null
DEF_ARCH=$("$CUOBJDUMP" --list-elf "$TMP/empty.cubin" 2>/dev/null | grep -o 'sm_[0-9]*' | sort -u | paste -sd, -)
if [[ -n "$DEF_ARCH" ]]; then
  if [[ " $CC " == *"$(echo "$DEF_ARCH" | tr -d 'sm_,')"* ]]; then
    pass "nvcc-arch" "default arch ${DEF_ARCH} (matches the GPU)"
  else
    # Not a capability of the box: a footgun of the toolchain. Warn loudly but
    # do not fail the gate; builds must pass -arch=native / -arch=sm_110.
    probe "nvcc-arch" "WARN: default arch ${DEF_ARCH} != GPU sm_${CC/./}; always pass -arch=native (see terrain compile_terrain.sh)"
  fi
else
  fail "nvcc-arch" "could not compile a default-arch cubin with $NVCC"
fi
rm -rf "$TMP"

# --- denoiser + cycles devices (venv bpy) -------------------------------------
if [[ ! -x "$VENV/bin/python" ]]; then
  fail "denoiser" "no python at $VENV/bin/python (usage: probe-thor.sh [venv])"
  fail "cycles"   "no python at $VENV/bin/python"
  fail "sm110"    "no python at $VENV/bin/python"
else
  # Each line the probe prints decides exactly one check; a missing
  # build_options attribute means the feature was compiled out (the flags
  # only exist for enabled options).
  if OUT=$("$VENV/bin/python" - <<'EOF' 2>&1
import bpy

# The operative denoiser signal is the Cycles enum: bpy.app.build_options
# has no openimagedenoise field in 4.2, and the enum raises on assignment
# when OIDN is not compiled into the wheel (verified on the phase-1 wheel).
try:
    bpy.data.scenes["Scene"].cycles.denoiser = "OPENIMAGEDENOISE"
    print("oidn: OK - Cycles accepts the OPENIMAGEDENOISE denoiser")
except Exception as e:
    print(f"oidn: FAIL - {type(e).__name__}: {e}")

try:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"; prefs.get_devices()
    types = sorted({d.type for d in prefs.devices})
    if "CUDA" in types:
        print(f"cycles: OK - device types: {' '.join(types)}")
    else:
        print(f"cycles: FAIL - no CUDA device among: {' '.join(types)}")
except Exception as e:
    print(f"cycles: FAIL - {type(e).__name__}: {e}")
EOF
  ); then :; else fail "denoiser" "python probe crashed: $(echo "$OUT" | tail -n1)"; fi
  echo "$OUT" | grep -q "^oidn: OK"   && pass "denoiser" "$(echo "$OUT" | grep -m1 '^oidn:')"
  echo "$OUT" | grep -q "^oidn: FAIL" && fail "denoiser" "$(echo "$OUT" | sed -n 's/^oidn: FAIL - //p' | head -n1)"
  echo "$OUT" | grep -q "^cycles: OK"   && pass "cycles" "$(echo "$OUT" | grep -m1 '^cycles:')"
  echo "$OUT" | grep -q "^cycles: FAIL" && fail "cycles" "$(echo "$OUT" | sed -n 's/^cycles: FAIL - //p' | head -n1)"

  # --- sm110: every CUDA-bearing .so must carry sm_110 SASS --------------------
  BAD=0; TOTAL=0
  SOS=()
  while IFS= read -r -d '' so; do SOS+=("$so"); done < <(
    find "$VENV" -path "*site-packages/bpy*" -name "*.so" -print0 2>/dev/null
    find "$REPO/src/infinigen/terrain" \( -path "*/lib/*" -name "*.so" -o -name "marching_cubes*.so" \) -print0 2>/dev/null
  )
  for so in "${SOS[@]}"; do
    ELF=$("$CUOBJDUMP" --list-elf "$so" 2>/dev/null | grep -o 'sm_[0-9]*' | sort -u)
    [[ -z "$ELF" ]] && continue        # pure-CPU library: nothing to check
    TOTAL=$((TOTAL+1))
    if [[ "$ELF" != *"sm_110"* ]]; then
      echo "  $so carries only: $(echo "$ELF" | paste -sd, -)"
      BAD=$((BAD+1))
    fi
  done
  if (( TOTAL == 0 )); then
    fail "sm110" "no CUDA-bearing .so files found under $VENV or terrain libs"
  elif (( BAD > 0 )); then
    fail "sm110" "$BAD of $TOTAL CUDA-bearing libs lack sm_110 SASS"
  else
    pass "sm110" "all $TOTAL CUDA-bearing .so files carry sm_110 SASS"
  fi
fi

echo
if (( FAILS > 0 )); then
  echo "probe-thor: $FAILS check(s) FAILED"
  exit 1
fi
echo "probe-thor: all capability checks passed"

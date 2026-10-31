#!/usr/bin/env bash
# thor/probe-thor-receipts.sh - measure what this Thor can actually do, WITH
# RECEIPTS.
#
# NOTE: the team's pipeline gate is thor/probe-thor.sh (a PASS/FAIL capability
# gate over the venv).  This script is the MEASUREMENT probe: it answers the
# open questions in thor/THOR-ARCHITECTURE.md with a written receipt per
# probe, and covers five things the gate does not (OptiX in the driver,
# nvidia-smi soc -q, NVCC_APPEND_FLAGS, the storage substrate, host malloc
# device-addressability).  Both are worth having; they are different tools.
#
# Every "it works" claim in the Thor port should come from a run of this
# script, not from an argument about hardware. It answers the open questions
# from thor/THOR-ARCHITECTURE.md and thor/THOR-EXPLOITS.md:
#
#   1. does the driver ship libnvoptix.so?   -> cheap OptiX *denoiser* path (E4)
#   2. does nvidia-smi report this GPU the way Infinigen asks?  (E5)
#   3. is nvidia-smi soc -q available (Tegra-only)?              (E5)
#   4. which Cycles device types / denoisers does the wheel have?
#   5. what does nvcc target by DEFAULT?       -> the sm_75 trap (E6)
#   6. is NVCC_APPEND_FLAGS set?
#   7. is the build tree on fuseblk/USB?       -> the wedging shelf (E1)
#   8. is host malloc device-addressable?      -> unified memory (EXP A)
#   9. is the terrain CUDA .so real sm_110 SASS?
#
# Usage:  thor/probe-thor.sh [OUTDIR]
# Exit:   0 = all REQUIRED probes passed, 1 = at least one required probe failed.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="${1:-$REPO_ROOT/outputs/probe}"
mkdir -p "$OUTDIR"
RECEIPT="$OUTDIR/probe-receipt.txt"
LOG="$OUTDIR/probe.log"
: > "$RECEIPT"; : > "$LOG"

FAILED=0
declare -a RESULTS=()
say()  { printf '%s\n' "$*" | tee -a "$LOG"; }
rec()  { printf '%-10s %s\n' "$1" "$2" | tee -a "$RECEIPT"; }
ok()   { rec "OK"      "$1"; RESULTS+=("OK|$1"); }
miss() { rec "MISSING" "$1"; RESULTS+=("MISSING|$1"); }
info() { rec "INFO"    "$1"; RESULTS+=("INFO|$1"); }
fail() { rec "FAIL"    "$1"; RESULTS+=("FAIL|$1"); FAILED=$((FAILED+1)); }
skip() { rec "SKIP"    "$1"; RESULTS+=("SKIP|$1"); }
have() { command -v "$1" >/dev/null 2>&1; }

say "=== thor/probe-thor.sh  $(date -Is) ==="

# ---------------------------------------------------------------------------
say ""
say "--- 1. box baseline ---"
say "kernel: $(uname -srm)"
if [ -r /etc/os-release ]; then . /etc/os-release; say "os: ${PRETTY_NAME:-?}"; fi
say "cores: $(nproc)   MemTotal: $(awk '/MemTotal/{printf "%.1f GiB", $2/1048576}' /proc/meminfo 2>/dev/null)"
if have nvpmodel; then say "nvpmodel: $(nvpmodel -q 2>&1 | tr '\n' ' ')"; else miss "nvpmodel (not a Jetson?)"; fi
if have nvidia-smi; then
  say "nvidia-smi: $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>&1 | head -1)"
else
  fail "nvidia-smi not found - this is not a CUDA box"
fi
if have tegrastats; then
  info "tegrastats: $(timeout 3 /usr/bin/tegrastats --interval 1000 2>/dev/null | head -1)"
else
  skip "tegrastats (not a Jetson?)"
fi

# ---------------------------------------------------------------------------
say ""
say "--- 2. how is the GPU visible? (the Infinigen 2.0 detection call) ---"
# src/infinigen2/util/hardware_info.py runs exactly:
#   nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# and keeps the result ONLY if returncode == 0. On Tegra, nvidia-smi is a
# different implementation and GPU memory is shared with the CPU. If this
# query fails, _detect_gpus() returns [] and the box reports itself as
# GPU-less to the scheduler (THOR-EXPLOITS.md E5).
if have nvidia-smi; then
  OUT=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1); RC=$?
  say "  the Infinigen call -> rc=$RC out='$OUT'"
  if [ "$RC" -eq 0 ] && [ -n "$OUT" ]; then
    ok "nvidia-smi --query-gpu=name,memory.total works (Infinigen will detect the GPU)"
  else
    fail "Infinigen 2.0's GPU-detection query FAILS (rc=$RC) - the box will report NO GPU"
    info "  fix: use --query-gpu=name,compute_cap, and nvidia-smi soc -q for memory"
  fi
  OUT2=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>&1); RC2=$?
  if [ "$RC2" -eq 0 ]; then
    ok "nvidia-smi --query-gpu=name,compute_cap -> $(echo "$OUT2" | head -1)"
  else
    fail "nvidia-smi --query-gpu=name,compute_cap also failed (rc=$RC2)"
  fi
  # Tegra-only SoC view. NVIDIA's docs example for this prints
  # "MemTotal: 128.83 GiB" - i.e. a Thor.
  if nvidia-smi soc -q >/dev/null 2>&1; then
    ok "nvidia-smi soc -q available (Tegra SoC view: memory + per-CPU clock/util)"
    info "  $(nvidia-smi soc -q 2>/dev/null | tr '\n' ' ' | cut -c1-160)"
  else
    skip "nvidia-smi soc -q not available (not a Tegra, or an older driver)"
  fi
else
  skip "nvidia-smi absent - cannot test the Infinigen detection path"
fi

# ---------------------------------------------------------------------------
say ""
say "--- 3. OptiX runtime in the driver (decides the denoiser path) ---"
# The OptiX *renderer* is off the table on Thor: the T5000 has NO RT cores
# (NVIDIA developer forum, Sept 2025: "There is no RT cores in AGX T5000";
#  the module datasheet's "Ray Tracing" line was a datasheet bug).
# But the OptiX *denoiser* runs on compute and does not need RT cores, and it
# comes from the driver's libnvoptix.so - not from a build step. If the
# library is here, that is the cheap path to a working denoiser (E4).
OPTIX_LIB="$(find /usr/lib /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64 /opt -name 'libnvoptix.so*' 2>/dev/null | head -3)"
if [ -n "$OPTIX_LIB" ]; then
  ok "libnvoptix.so present: $(echo "$OPTIX_LIB" | tr '\n' ' ')"
  info "  -> WITH_CYCLES_DEVICE_OPTIX=ON may compile in the OptiX DENOISER."
  info "  -> keep compute_device_type=CUDA; the OptiX renderer has no RT cores."
else
  miss "libnvoptix.so NOT found - OptiX denoiser path closed; use the patch-infinigen2 fallback + a GPU denoiser"
fi
if [ -n "${OPTIX_ROOT:-}" ] && [ -d "$OPTIX_ROOT/include/optix.h" ]; then
  ok "OptiX SDK headers at \$OPTIX_ROOT=$OPTIX_ROOT"
else
  info "OptiX SDK headers: not on this box (the SDK DOES ship an Arm build - see THOR-ARCHITECTURE.md B2)"
fi

# ---------------------------------------------------------------------------
say ""
say "--- 4. storage: what is the build tree actually on? ---"
# The external cold shelf is NTFS-over-fuseblk on a USB bridge and wedged
# mid-run on 2026-10-31 (D-state page faults for 30+ min; unmounted but
# unreadable until the bridge was replugged). A ninja build tree on that
# substrate is the workload most likely to wedge it again - and the logs
# that would explain what happened live there too (THOR-EXPLOITS.md E1).
for p in "$REPO_ROOT" "$REPO_ROOT/.toolchain" "$REPO_ROOT/.ext-thor" "$REPO_ROOT/.venv"; do
  [ -e "$p" ] || continue
  FSTYPE=$(stat -f -c %T "$p" 2>/dev/null)
  say "  $p -> $FSTYPE"
  case "$FSTYPE" in
    fuseblk|fuse|ntfs*|msdos*|vfat*)
      fail "$p is on $FSTYPE (USB/fuse) - build trees and logs must be on NVMe" ;;
    *)  ok "$p is on $FSTYPE" ;;
  esac
done
if [ -L "$REPO_ROOT/.ext-thor" ]; then
  info "  .ext-thor -> $(readlink -f "$REPO_ROOT/.ext-thor")"
fi

# ---------------------------------------------------------------------------
say ""
say "--- 5. nvcc default target (the sm_75 trap) ---"
if have nvcc; then
  say "nvcc: $(nvcc --version | tail -2 | head -1)"
  say "NVCC_APPEND_FLAGS=${NVCC_APPEND_FLAGS:-<unset>}"
  if [ -n "${NVCC_APPEND_FLAGS:-}" ]; then
    ok "NVCC_APPEND_FLAGS is set (every nvcc call on this box gets it)"
  else
    miss "NVCC_APPEND_FLAGS unset - source thor/thor-env.sh (E6)"
  fi
  D=$(mktemp -d)
  cat > "$D/t.cu" <<'CUEOF'
#include <cstdio>
__global__ void k(float *x){ x[threadIdx.x] += 1.0f; }
int main(){ float h[4]={0,0,0,0},*d; cudaMalloc(&d,16);
  cudaMemcpy(d,h,16,cudaMemcpyHostToDevice); k<<<1,4>>>(d);
  cudaMemcpy(h,d,16,cudaMemcpyDeviceToHost); printf("%f\n",h[0]); return 0; }
CUEOF
  if nvcc -O2 "$D/t.cu" -o "$D/t" >"$D/nvcc.log" 2>&1; then
    ELF=$(cuobjdump --list-elf "$D/t" 2>/dev/null | grep -o 'sm_[0-9]*' | sort -u | tr '\n' ' ')
    if [ -n "$ELF" ]; then
      if echo "$ELF" | grep -q sm_110; then
        ok "nvcc default arch is sm_110 ($ELF)"
      else
        fail "nvcc default arch is $ELF, NOT sm_110 - every from-source extension needs -arch"
        info "  (cubins for $ELF cannot load on this GPU: 'no kernel image is available')"
      fi
      if cuobjdump --list-ptx "$D/t" 2>/dev/null | grep -q 'ptx'; then
        info "  PTX embedded: yes (portable across aarch64 GPUs)"
      else
        info "  PTX embedded: no (SASS-only - sm_110 specific)"
      fi
    else
      skip "could not read SASS arch from the default-arch binary"
    fi
    if [ -x "$D/t" ]; then
      R=$("$D/t" 2>&1); [ "$R" = "1.000000" ] && ok "default-arch kernel RUNS on this GPU" \
        || fail "default-arch kernel did not run correctly (got '$R')"
    fi
  else
    fail "nvcc failed to compile a trivial kernel with no -arch: $(tail -2 "$D/nvcc.log" | tr '\n' ' ')"
  fi
  rm -rf "$D"
else
  skip "nvcc not on PATH"
fi

# ---------------------------------------------------------------------------
say ""
say "--- 6. unified memory: is host malloc device-addressable? ---"
# Re-verify of builder-ide docs/lab/exp-unified-memory.md A1 on THIS kernel.
UVENV=""
for v in "$REPO_ROOT/.venv/bin/python" "$PWD/.venv/bin/python" "$REPO_ROOT/.toolchain/python-3.11/bin/python3.11"; do
  [ -x "$v" ] && { UVENV="$v"; break; }
done
[ -n "$UVENV" ] || UVENV="$(command -v python3 || true)"
if [ -n "$UVENV" ]; then
  "$UVENV" - "$LOG" <<'PYEOF' 2>&1 | tee -a "$LOG"
import ctypes, sys
lib = None
for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.13"):
    try:
        lib = ctypes.CDLL(name); break
    except OSError:
        continue
if lib is None:
    print("UNIFIED: SKIP (libcudart not loadable from this interpreter)"); sys.exit(0)
class Attr(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("device", ctypes.c_int),
                ("devicePointer", ctypes.c_void_p), ("hostPointer", ctypes.c_void_p)]
lib.cudaPointerGetAttributes.argtypes = [ctypes.POINTER(Attr), ctypes.c_void_p]
buf = ctypes.create_string_buffer(1024)
a = Attr()
lib.cudaPointerGetAttributes(ctypes.byref(a), ctypes.cast(buf, ctypes.c_void_p))
print("UNIFIED: devicePointer=0x%x hostPointer=0x%x type=%d device=%d" %
      (a.devicePointer or 0, a.hostPointer or 0, a.type, a.device))
if a.devicePointer and a.devicePointer == a.hostPointer:
    print("UNIFIED: OK - host malloc IS a valid device address (one pool)")
else:
    print("UNIFIED: FAIL - host and device pointers differ; not a unified pool here")
PYEOF
  if grep -q "UNIFIED: OK" "$LOG"; then ok "host malloc is device-addressable (unified memory confirmed)";
  elif grep -q "UNIFIED: FAIL" "$LOG"; then fail "host and device pointers differ - unified memory NOT in effect";
  else skip "unified-memory probe inconclusive"; fi
else
  skip "no python interpreter found for the unified-memory probe"
fi

# ---------------------------------------------------------------------------
say ""
say "--- 7. the bpy wheel: device types and denoisers ---"
BPY_PY=""
for v in "$REPO_ROOT/.venv/bin/python" "$PWD/.venv/bin/python" "$REPO_ROOT/.toolchain/python-3.11/bin/python3.11"; do
  if [ -x "$v" ] && "$v" -c "import bpy" >/dev/null 2>&1; then BPY_PY="$v"; break; fi
done
if [ -z "$BPY_PY" ]; then
  for w in "$REPO_ROOT"/.toolchain/wheels/bpy-*.whl "$PWD"/../wheels/bpy-*.whl "$HOME"/wheels/bpy-*.whl; do
    [ -f "$w" ] && { info "wheel present but not installed in a venv: $w"; break; }
  done
  skip "bpy not importable - run thor/install.sh first"
else
  "$BPY_PY" - <<'PYEOF' 2>&1 | tee -a "$LOG"
import bpy
print("BPY: version=%s build=%s" % (bpy.app.version_string, bpy.app.build_platform.decode()))
prefs = bpy.context.preferences.addons["cycles"].preferences
for dt in prefs.get_device_types(bpy.context):
    prefs.get_devices_for_type(dt[0])
print("BPY: cycles device types = %s" % sorted({d.type for d in prefs.devices}))
scene = bpy.context.scene
for name in ("OPTIX", "OPENIMAGEDENOISE"):
    try:
        scene.cycles.denoiser = name
        print("BPY: denoiser %s -> AVAILABLE (%s)" % (name, scene.cycles.denoiser))
    except Exception as e:
        print("BPY: denoiser %s -> UNAVAILABLE (%s)" % (name, type(e).__name__))
PYEOF
  if grep -q "denoiser .* -> AVAILABLE" "$LOG"; then
    ok "a Cycles denoiser is available in this wheel"
  else
    fail "NO denoiser in this wheel - Infinigen 2.0 render_cycles.py will raise RuntimeError('No usable denoiser')"
    info "  fix now, no rebuild: thor/patch-infinigen2-denoise-fallback.patch"
  fi
  if grep -q "device types = \['CUDA'" "$LOG"; then
    ok "Cycles exposes CUDA (expected on Thor: no RT cores -> no OPTIX device)"
  else
    info "Cycles device types: $(grep 'device types' "$LOG" | tail -1)"
  fi
fi

# ---------------------------------------------------------------------------
say ""
say "--- 8. terrain CUDA libraries: real sm_110 SASS? ---"
FOUND_TERRAIN=0
for root in "$REPO_ROOT/src/infinigen/terrain" "$PWD/src/infinigen/terrain"; do
  [ -d "$root/lib/cuda" ] || continue
  FOUND_TERRAIN=1
  n=$(find "$root/lib/cuda" -name '*.so' 2>/dev/null | wc -l)
  info "terrain cuda libs: $n .so under $root/lib/cuda"
  if [ "$n" -gt 0 ]; then
    SAMPLE=$(find "$root/lib/cuda" -name '*.so' | head -1)
    ELFS=$(cuobjdump --list-elf "$SAMPLE" 2>/dev/null | grep -o 'sm_[0-9]*' | sort -u | tr '\n' ' ')
    if echo "$ELFS" | grep -q sm_110; then
      ok "terrain CUDA kernels are sm_110 SASS ($ELFS)"
    elif [ -n "$ELFS" ]; then
      fail "terrain CUDA kernels are $ELFS, not sm_110 - rebuild with TERRAIN_CUDA_ARCH=sm_110"
    else
      skip "could not read SASS arch from $SAMPLE (cuobjdump unavailable?)"
    fi
  fi
  break
done
[ "$FOUND_TERRAIN" -eq 0 ] && skip "no terrain/lib/cuda tree found (terrain extras not installed)"

# ---------------------------------------------------------------------------
say ""
say "--- summary ---"
say "required probes failed: $FAILED"
printf '%s\n' "${RESULTS[@]}" > "$OUTDIR/probe-results.txt"
say "receipt: $RECEIPT"
say "results: $OUTDIR/probe-results.txt"
exit $(( FAILED > 0 ? 1 : 0 ))

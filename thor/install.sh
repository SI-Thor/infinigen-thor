#!/usr/bin/env bash
# thor/install.sh - install Infinigen (this repo, editable) into a venv against
# the locally-built bpy 4.2.0 aarch64 wheel, with the terrain C++/CUDA libs.
#
# Why: Infinigen pins bpy==4.2.0 and its legacy (1.0) pipelines used by
# Hello Room / Hello World (generate_indoors / generate_nature) need the
# terrain extension + submodules; the documented default install is
# MINIMAL=True which skips them. The venv uses the CPython 3.11.13 built by
# thor/build-bpy.sh, because bpy 4.2 only supports Python 3.11.
#
# The terrain .so libs are built by `make terrain` (compile_terrain.sh,
# patched to pass -arch=native to nvcc so the CUDA kernels target the Thor's
# sm_110 GPU). setup.py re-runs that during pip install; it is idempotent.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT_LINK="$REPO/.ext-thor"
PY="$REPO/.toolchain/python-3.11/bin/python3.11"
# VENV is overridable so phase-2 work can target .venv-p2 while the seed
# sweep keeps running against the original .venv (do not touch that one).
VENV="${VENV:-$REPO/.venv}"
WHEEL_DIR="$REPO/.toolchain/wheels"

stage() { printf '\n=== [%s] %s\n' "$(date +%H:%M:%S)" "$1"; }

stage "venv at $VENV (python 3.11 from thor toolchain)"
# Recreate if it points at a python outside .toolchain (e.g. the retired
# external-drive toolchain from before the fuseblk cold-shelf incident).
if [[ -x "$VENV/bin/python" ]] && grep -q "home = $REPO/.toolchain" "$VENV/pyvenv.cfg" 2>/dev/null; then
  echo "venv already wired to .toolchain"
else
  rm -rf "$VENV"
  "$PY" -m venv "$VENV"
fi
"$VENV/bin/python" --version

stage "install local bpy wheel first (satisfies the bpy==4.2.0 pin)"
WHEEL="$(ls -1t "$WHEEL_DIR"/bpy-4.2.0-*.whl 2>/dev/null | head -n1 || true)"
if [[ -z "$WHEEL" ]]; then
  echo "no bpy wheel in $WHEEL_DIR - run thor/build-bpy.sh first" >&2
  exit 1
fi
"$VENV/bin/pip" -q install --no-deps "$WHEEL"
"$VENV/bin/pip" show bpy > /dev/null && echo "bpy 4.2.0 installed in venv"

stage "submodules (terrain needs OcMesher + glm)"
git -C "$REPO" submodule update --init --recursive \
  src/infinigen/OcMesher src/infinigen/infinigen_gpl \
  src/infinigen/datagen/customgt/dependencies/glm

stage "pip install -e .[infinigen1,terrain] (builds terrain + marching_cubes)"
# INFINIGEN_INSTALL_TERRAIN is overridable: setup.py's `make terrain` step is
# NOT idempotent (it rewrites every terrain .so in the shared source tree),
# so a second venv created while renders are running against the first one
# (e.g. .venv-p2 during the phase-2 seed sweep on .venv) must skip it and
# reuse the .so already built in-tree; a half-written .so would dlopen-fail
# a running job. The [terrain] extra still installs all python deps.
INFINIGEN_MINIMAL_INSTALL=False INFINIGEN_INSTALL_TERRAIN="${INFINIGEN_INSTALL_TERRAIN:-True}" \
  "$VENV/bin/pip" install -e "$REPO[infinigen1,terrain]" \
  > "$EXT_LINK/logs/infinigen-install-${VENV##*/}.log" 2>&1 || {
    echo "install failed, tail of log:"; tail -30 "$EXT_LINK/logs/infinigen-install-${VENV##*/}.log"; exit 1; }

stage "smoke: import infinigen terrain + bpy in the same interpreter"
"$VENV/bin/python" - <<'EOF'
import bpy
from infinigen.terrain.utils.ctype_util import load_cdll
dll = load_cdll("terrain/lib/cpu/elements/mountains.so")
print("bpy:", bpy.app.version_string, "| terrain cpu mountains.so loads OK")
dll = load_cdll("terrain/lib/cuda/elements/mountains.so")
print("terrain cuda mountains.so loads OK")
EOF
stage "done"

#!/usr/bin/env bash
# thor/build-bpy.sh - Blender 4.2.0 as the `bpy` Python module for the Jetson AGX Thor
#
# Why this script exists: there is no prebuilt Blender 4.2 / bpy 4.2 for Linux
# ARM64 anywhere official (checked 2026-10-30):
#   - PyPI bpy==4.2.0 ships manylinux_2_28_x86_64, macos arm64/x86_64, win_amd64 only.
#   - download.blender.org/release/Blender4.2/ has blender-4.2.x-linux-x64.tar.xz only.
# Infinigen pins bpy==4.2.0, so the only path is building the Python module
# (WITH_PYTHON_MODULE) from source against CPython 3.11 (the Python Blender 4.2
# embeds) with Cycles CUDA kernels for the Thor's Blackwell GPU (compute
# capability 11.0 -> sm_110, CUDA 13.2 at /usr/local/cuda). OptiX has no
# aarch64 SDK, so CUDA only.
#
# Environment / box assumptions (see thor/SPIKE.md):
#   - Ubuntu 24.04 (noble) aarch64, 14 cores, passwordless sudo for apt.
#   /usr/local/cuda = CUDA 13.2 toolkit, driver 595.x, GPU "NVIDIA Thor" cc 11.0.
#   - Big trees live on the external USB SSD that mounts at a path WITH A SPACE
#   ("/media/caseydigennaro/TOSHIBA EXT"). make recipes (CPython) and some
#   CMake code do not quote paths, so everything is built through the
#   space-free repo-local symlink .ext-thor -> that drive's thor-spike/ dir.
#
# Stages are idempotent: each is skipped if its output already exists.
# Sources/build trees and the toolchain live under $EXT_LINK; the produced
# wheel lands in $EXT_LINK/wheels/ and is what thor/install.sh consumes.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT_DRIVE="${EXT_DRIVE:-/media/caseydigennaro/TOSHIBA EXT/thor-spike}"
EXT_LINK="$REPO/.ext-thor"                 # space-free path to $EXT_DRIVE
PY_PREFIX="$EXT_LINK/python-3.11"          # CPython 3.11 toolchain
BLENDER_SRC="$EXT_LINK/src/blender"        # blender v4.2.0 checkout
BUILD_DIR="$EXT_LINK/src/build-bpy"        # cmake build tree
WHEEL_DIR="$EXT_LINK/wheels"
LOG_DIR="$EXT_LINK/logs"
BLENDER_TAG="${BLENDER_TAG:-v4.2.0}"
JOBS="${JOBS:-$(nproc)}"
# Thor's Blackwell GPU is cc 11.0. CMake 3.28 accepts unknown numeric
# architectures and passes them straight to nvcc (verified on this box).
CUDA_ARCH="${CUDA_ARCH:-110}"

stage() { printf '\n=== [%s] %s (start %s)\n' "$(date +%H:%M:%S)" "$1" "$(date +%s)"; }
log()   { echo "[build-bpy] $*"; }

mkdir -p "$EXT_DRIVE" "$WHEEL_DIR" "$LOG_DIR"
ln -sfn "$EXT_DRIVE" "$EXT_LINK"

# ---------------------------------------------------------------------------
stage 0 "apt dependencies"
# System-library build of Blender 4.2 (no vcpkg) + Infinigen's documented deps.
# Not packaged in noble: OpenSubdiv is 'libosd-dev' (3.5), OIDN is absent
# (WITH_OPENIMAGEDENOISE=OFF), USD is absent (WITH_USD=OFF).
APT_PKGS=(
  build-essential git cmake ninja-build gettext wget pkg-config
  libx11-dev libxxf86vm-dev libxcursor-dev libxi-dev libxrandr-dev
  libxinerama-dev libxfixes-dev libxrender-dev libwayland-dev wayland-protocols
  libxkbcommon-dev libdbus-1-dev libegl-dev libgl-dev libgles2-mesa-dev mesa-common-dev
  libopenexr-dev libimath-dev libopenimageio-dev libopencolorio-dev libopenjp2-7-dev
  libjpeg-dev libpng-dev libtiff-dev libwebp-dev libtbb-dev libopenvdb-dev
  libosd-dev libboost-all-dev libpugixml-dev libyaml-cpp-dev libblosc-dev
  libsndfile1-dev libopenal-dev libpulse-dev libjack-jackd2-dev libfftw3-dev
  libavcodec-dev libavformat-dev libavutil-dev libavdevice-dev libswscale-dev libswresample-dev
  libzstd-dev liblzma-dev libbz2-dev zlib1g-dev libssl-dev libffi-dev libsqlite3-dev
  libreadline-dev libncurses-dev libgmp-dev libpotrace-dev
  libfreetype-dev libharfbuzz-dev libfribidi-dev libudev-dev libdrm-dev
  libepoxy-dev libglew-dev libglfw3-dev libglm-dev
)
sudo apt-get install -y -q "${APT_PKGS[@]}" > "$LOG_DIR/apt.log" 2>&1 || {
  log "apt install had failures, inspect $LOG_DIR/apt.log"; exit 1; }

# ---------------------------------------------------------------------------
stage 1 "CUDA sanity (nvcc -arch=native on the Thor GPU)"
if [[ -x /tmp/cudatest/t ]] && /tmp/cudatest/t | grep -q "cc=11.0"; then
  log "cuda sanity already proven"
else
  mkdir -p /tmp/cudatest && cd /tmp/cudatest
  cat > t.cu <<'EOF'
#include <cuda_runtime.h>
#include <cstdio>
__global__ void add_one(float *x) { x[threadIdx.x] += 1.0f; }
int main() {
  int dev=0; cudaDeviceProp p; cudaGetDeviceProperties(&p,dev);
  printf("device=%s cc=%d.%d\n", p.name, p.major, p.minor);
  float h[4]={0,0,0,0}; float *d;
  cudaMalloc(&d,16); cudaMemcpy(d,h,16,cudaMemcpyHostToDevice);
  add_one<<<1,4>>>(d);
  cudaMemcpy(h,d,16,cudaMemcpyDeviceToHost);
  printf("kernel result: %f %f %f %f\n", h[0],h[1],h[2],h[3]);
  return 0;
}
EOF
  /usr/local/cuda/bin/nvcc -O2 -arch=native t.cu -o t && ./t | tee "$LOG_DIR/cuda-sanity.log"
  cd - > /dev/null
fi

# ---------------------------------------------------------------------------
stage 2 "CPython 3.11.13 (shared) -> $PY_PREFIX"
# Blender 4.2 embeds exactly Python 3.11 (build_files/cmake/Modules/
# FindPythonLibsUnix.cmake: _PYTHON_VERSION_SUPPORTED 3.11); Ubuntu 24.04 ships
# 3.12 only, so we build 3.11.13 with --enable-shared. An rpath to $PY_PREFIX/lib
# is required or the interpreter cannot find libpython3.11.so.1.0.
if "$PY_PREFIX/bin/python3.11" -c 'import sys, sysconfig
assert sys.version_info[:2] == (3, 11)
assert sysconfig.get_config_var("Py_ENABLE_SHARED")' 2>/dev/null; then
  log "python 3.11 already OK"
else
  cd "$EXT_LINK/src"
  rm -rf Python-3.11.13 python-3.11
  [[ -f python-3.11.13.tgz ]] || curl -sLO https://www.python.org/ftp/python/3.11.13/Python-3.11.13.tgz
  mv -f Python-3.11.13.tgz python-3.11.13.tgz 2>/dev/null || true
  tar xf python-3.11.13.tgz
  cd Python-3.11.13
  LDFLAGS="-Wl,-rpath,$PY_PREFIX/lib" ./configure --prefix="$PY_PREFIX" \
    --enable-shared --with-ensurepip=install > "$LOG_DIR/cpython-configure.log" 2>&1
  make -j"$JOBS"  > "$LOG_DIR/cpython-make.log" 2>&1
  make install   > "$LOG_DIR/cpython-install.log" 2>&1
  "$PY_PREFIX/bin/python3.11" -c 'import sysconfig; print("python OK", sys.version)'
fi
PY="$PY_PREFIX/bin/python3.11"

# ---------------------------------------------------------------------------
stage 3 "Blender $BLENDER_TAG checkout"
if [[ -d "$BLENDER_SRC/.git" ]]; then
  log "blender checkout exists"
else
  mkdir -p "$EXT_LINK/src"
  git clone --branch "$BLENDER_TAG" --depth 1 --recurse-submodules \
    --shallow-submodules --jobs "$JOBS" \
    https://github.com/blender/blender.git "$BLENDER_SRC" \
    > "$LOG_DIR/clone-blender.log" 2>&1
fi
git -C "$BLENDER_SRC" log --oneline -1

# Local patches to the Blender tree (kept here, applied idempotently).
for p in "$REPO"/thor/patches/blender-*.patch; do
  if git -C "$BLENDER_SRC" apply --check --reverse "$p" 2>/dev/null; then
    log "patch already applied: $(basename "$p")"
  else
    git -C "$BLENDER_SRC" apply "$p"
    log "patch applied: $(basename "$p")"
  fi
done

# ---------------------------------------------------------------------------
stage 3b "OpenImageIO v2.5.11.0 (static)"
# Ubuntu 24.04 ships OIIO 2.4.17, but Blender 4.2 needs >= 2.5
# (ImageInput::valid_file(IOProxy*) does not exist in 2.4; first seen as a
# compile error in source/blender/imbuf/intern/oiio/openimageio_support.cc).
# Build the exact version Blender 4.2 pins (build_files/build_environment/
# cmake/versions.cmake) as static libs so the wheel carries no extra .so deps.
OIIO_PREFIX="$EXT_LINK/oiio"
OIIO_SRC="$EXT_LINK/src/OpenImageIO-2.5.11.0"
if [[ -f "$OIIO_PREFIX/lib/libOpenImageIO.a" ]]; then
  log "OIIO already installed"
else
  mkdir -p "$EXT_LINK/src"; cd "$EXT_LINK/src"
  [[ -d OpenImageIO-2.5.11.0 ]] || {
    curl -sL -o oiio-2.5.11.0.tar.gz https://github.com/AcademySoftwareFoundation/OpenImageIO/archive/refs/tags/v2.5.11.0.tar.gz
    tar xf oiio-2.5.11.0.tar.gz
  }
  cmake -S "$OIIO_SRC" -B "$OIIO_SRC/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$OIIO_PREFIX" \
    -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF -DBUILD_DOCS=OFF \
    -DUSE_PYTHON=OFF -DUSE_QT=OFF -DUSE_OpenCV=OFF -DUSE_FFMPEG=OFF \
    -DUSE_LIBRAW=OFF -DUSE_OPENCOLORIO=OFF -DUSE_OPENGL=OFF \
    > "$LOG_DIR/oiio-configure.log" 2>&1
  ninja -C "$OIIO_SRC/build" -j"$JOBS" > "$LOG_DIR/oiio-build.log" 2>&1
  cmake --install "$OIIO_SRC/build" > "$LOG_DIR/oiio-install.log" 2>&1
  log "OIIO installed at $OIIO_PREFIX"
fi

# ---------------------------------------------------------------------------
stage 4 "cmake configure bpy module (CUDA arch $CUDA_ARCH)"
cmake -S "$BLENDER_SRC" -B "$BUILD_DIR" -G Ninja \
  -C "$BLENDER_SRC/build_files/cmake/config/bpy_module.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$OIIO_PREFIX" \
  -DPYTHON_ROOT_DIR="$PY_PREFIX" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DOPENIMAGEIO_ROOT_DIR="$OIIO_PREFIX" \
  -UOPENIMAGEIO_INCLUDE_DIR -UOPENIMAGEIO_LIBRARY -UOPENIMAGEIO_LIBRARIES \
  -UOPENIMAGEIO_UTIL_LIBRARY \
  -DWITH_CYCLES_CUDA_BINARIES=ON \
  -DCYCLES_CUDA_BINARIES_ARCH=sm_"$CUDA_ARCH" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DWITH_CYCLES_DEVICE_OPTIX=OFF \
  -DWITH_CYCLES_DEVICE_HIP=OFF \
  -DWITH_CYCLES_DEVICE_ONEAPI=OFF \
  -DWITH_OPENIMAGEDENOISE=OFF \
  -DWITH_ALEMBIC=OFF \
  -DWITH_OPENCOLLADA=OFF \
  -DWITH_USD=OFF \
  -DWITH_BUILDINFO=OFF \
  2>&1 | tee "$LOG_DIR/blender-configure.log" | grep -Ei "error|warning: unable|could NOT|not found" || true
log "configure done"

# ---------------------------------------------------------------------------
stage 5 "build (ninja -j$JOBS) - expect 30-90 min"
cmake --build "$BUILD_DIR" -j "$JOBS" > "$LOG_DIR/blender-build.log" 2>&1
log "build finished"

# ---------------------------------------------------------------------------
stage 6 "bpy wheel"
cd "$BLENDER_SRC/build_files/utils"
"$PY" make_bpy_wheel.py "$BUILD_DIR/bin" --output-dir="$WHEEL_DIR" \
  > "$LOG_DIR/wheel.log" 2>&1
ls -la "$WHEEL_DIR"

# ---------------------------------------------------------------------------
stage 7 "smoke test: import bpy + render on CPU and CUDA"
SMOKE="$EXT_LINK/logs/bpy-smoke"
rm -rf "$SMOKE"; mkdir -p "$SMOKE"
"$PY" -m venv "$SMOKE/venv"
"$SMOKE/venv/bin/pip" -q install "$WHEEL_DIR"/bpy-*.whl
"$SMOKE/venv/bin/python" - <<'EOF' 2>&1 | tee "$LOG_DIR/bpy-smoke.log"
import bpy, time
print("bpy", bpy.app.version_string, "| build:", bpy.app.build_platform.decode())
for dev in bpy.context.preferences.addons["cycles"].preferences.devices:
    print("cycles device:", dev.name, dev.type)
for device in ("CPU", "CUDA"):
    scene = bpy.data.scenes["Scene"]
    scene.render.engine = "CYCLES"
    scene.cycles.device = device
    scene.render.resolution_x = scene.render.resolution_y = 128
    scene.cycles.samples = 32
    scene.render.filepath = f"/tmp/bpy-smoke-{device.lower()}.png"
    t0 = time.time(); bpy.ops.render.render(write_still=True)
    print(f"render[{device}] {time.time()-t0:.1f}s -> {scene.render.filepath}")
EOF
log "DONE. wheel in $WHEEL_DIR"

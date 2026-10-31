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
# capability 11.0 -> sm_110, CUDA 13.2 at /usr/local/cuda).
#
# WHY THERE IS NO OPTIX DEVICE (read before "fixing" it):
#   The OptiX *SDK* does ship an ARM build - NVIDIA's download page lists
#   "OptiX SDK 9.1 (Windows, Linux, Arm)" and the legacy page lists 8.1.0 as
#   "(Windows, Linux, Arm)". The SDK is not the problem.
#   The problem is the *hardware*: the Jetson AGX Thor T5000 has NO RT CORES.
#   NVIDIA staff on the developer forum, 1 Sept 2025: "There is no RT cores in
#   AGX T5000" - and the module datasheet's "Modern Graphics features: Ray
#   Tracing" line was a datasheet bug ("This is a mistake in module data sheet.
#   We will fix it."). Without RT cores the OptiX ray tracer has nothing to
#   accelerate with, so the renderer stays on CUDA. This is a hardware fact and
#   will not change with an SDK update.
#   (The OptiX *denoiser* is a different animal - it runs on compute and does
#   not need RT cores - but it needs libnvoptix.so from the driver, and Cycles
#   only compiles the OptiX denoiser in when WITH_CYCLES_DEVICE_OPTIX is on.
#   thor/probe-thor.sh checks whether the driver ships it.)
#
# DENOISER: WITH_OPENIMAGEDENOISE is now ON.
#   Infinigen 2.0's src/infinigen2/exporters/render_cycles.py defaults to
#   DenoiseMode.BEST and RAISES RuntimeError("No usable denoiser ...") when
#   neither OPTIX nor OPENIMAGEDENOISE is available (1.x only warns). A 2.0
#   render on the Thor as built dies instead of degrading.
#   Blender 4.2 builds OIDN 2.3.0 FROM SOURCE (versions.cmake pins
#   oidn-2.3.0.src.tar.gz), so there is no aarch64 package to find - the build
#   does it. But upstream's openimagedenoise.cmake turns the SYCL, CUDA and
#   HIP devices ON for every non-Windows-ARM platform, which cannot build on
#   aarch64. thor/patches/blender-v4.2.0-oidn-arm64-cpu-only.patch forces the
#   CPU device (OIDN 2.3.0 supports ARM64/NEON); set OIDN_DEVICE_CUDA_ARCH=110
#   to build the CUDA device for sm_110 instead.
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
# Sources and build trees live under $EXT_LINK per the brief. The CPython
# toolchain and the produced wheel live on the NVMe (.toolchain/, gitignored)
# because the external cold shelf is NTFS-on-fuseblk here and it wedged hard
# under load mid-run: processes D-stated on page faults from libpython for
# 30+ min and the drive stayed unreadable even after a lazy umount. Runtime-
# loadable artifacts must not page from that drive.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT_DRIVE="${EXT_DRIVE:-/media/caseydigennaro/TOSHIBA EXT/thor-spike}"
EXT_LINK="$REPO/.ext-thor"                 # space-free path to $EXT_DRIVE
PY_PREFIX="$REPO/.toolchain/python-3.11"   # CPython 3.11 toolchain (NVMe: see note below)
BLENDER_SRC="$EXT_LINK/src/blender"        # blender v4.2.0 checkout
BUILD_DIR="$EXT_LINK/src/build-bpy"        # cmake build tree
WHEEL_DIR="$REPO/.toolchain/wheels"
LOG_DIR="$EXT_LINK/logs"
BLENDER_TAG="${BLENDER_TAG:-v4.2.0}"
JOBS="${JOBS:-$(nproc)}"
# Thor's Blackwell GPU is cc 11.0. CMake 3.28 accepts unknown numeric
# architectures and passes them straight to nvcc (verified on this box).
CUDA_ARCH="${CUDA_ARCH:-110}"
# OIDN device: leave empty for the CPU device (NEON); set to 110 for the OIDN
# CUDA device on sm_110. The OIDN CUDA device needs the same -arch discipline
# as everything else on this box: nvcc's default target is sm_75 and those
# cubins cannot load here.
OIDN_DEVICE_CUDA_ARCH="${OIDN_DEVICE_CUDA_ARCH:-}"

stage() { printf '\n=== [%s] %s (start %s)\n' "$(date +%H:%M:%S)" "$1" "$(date +%s)"; }
log()   { echo "[build-bpy] $*"; }

mkdir -p "$EXT_DRIVE" "$WHEEL_DIR" "$LOG_DIR"
ln -sfn "$EXT_DRIVE" "$EXT_LINK"

# ---------------------------------------------------------------------------
stage 0 "apt dependencies"
# System-library build of Blender 4.2 (no vcpkg) + Infinigen's documented deps.
# Not packaged in noble: OpenSubdiv is 'libosd-dev' (3.5), USD is absent
# (WITH_USD=OFF). OIDN is NOT in this list on purpose - Blender builds OIDN
# 2.3.0 from source and noble's libtbb-dev covers its TBB dependency.
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
  -DCMAKE_CXX_STANDARD_LIBRARIES="-ltiff" \
  -DWITH_CYCLES_CUDA_BINARIES=ON \
  -DCYCLES_CUDA_BINARIES_ARCH=sm_"$CUDA_ARCH" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DWITH_CYCLES_DEVICE_OPTIX=OFF \
  -DWITH_CYCLES_DEVICE_HIP=OFF \
  -DWITH_CYCLES_DEVICE_ONEAPI=OFF \
  -DWITH_OPENIMAGEDENOISE=ON \
  -DOIDN_DEVICE_CUDA_ARCH="$OIDN_DEVICE_CUDA_ARCH" \
  -DWITH_ALEMBIC=OFF \
  -DWITH_OPENCOLLADA=OFF \
  -DWITH_USD=OFF \
  -DWITH_BUILDINFO=OFF \
  2>&1 | tee "$LOG_DIR/blender-configure.log" | grep -Ei "error|warning: unable|could NOT|not found" || true
log "configure done"

# ---------------------------------------------------------------------------
stage 5 "build (ninja -j$JOBS) - expect 30-90 min"
cmake --build "$BUILD_DIR" -j "$JOBS" --target install > "$LOG_DIR/blender-build.log" 2>&1
log "build finished"

# ---------------------------------------------------------------------------
stage 6 "bpy wheel"
"$PY" -m pip -q install wheel
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
import bpy, time, sys
print("bpy", bpy.app.version_string, "| build:", bpy.app.build_platform.decode())
for dev in bpy.context.preferences.addons["cycles"].preferences.devices:
    print("cycles device:", dev.name, dev.type)
# --- denoiser assertion -------------------------------------------------
# Infinigen 2.0 raises RuntimeError("No usable denoiser ...") rather than
# degrading, so a wheel that renders but cannot denoise is not usable by the
# engine. Assert it here, on the box, every time the wheel is rebuilt.
denoisers = {}
for name in ("OPTIX", "OPENIMAGEDENOISE"):
    try:
        scene.cycles.denoiser = name
        denoisers[name] = "AVAILABLE"
    except Exception as e:
        denoisers[name] = "UNAVAILABLE (%s)" % type(e).__name__
print("denoisers:", denoisers)
if not any(v == "AVAILABLE" for v in denoisers.values()):
    print("SMOKE FAIL: no denoiser - Infinigen 2.0 render_cycles.py will raise")
    sys.exit(2)
usable = [k for k, v in denoisers.items() if v == "AVAILABLE"]
scene.cycles.denoiser = usable[0]
scene.cycles.use_denoising = True
print("denoiser in use:", scene.cycles.denoiser)

for device in ("CPU", "GPU"):
    scene = bpy.data.scenes["Scene"]
    scene.cycles.use_denoising = False  # OIDN is not available in this build
    scene.render.engine = "CYCLES"
    if device == "GPU":
        # Prefer CUDA on the Thor (no RT cores -> no usable OptiX renderer).
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == "CUDA"
    scene.cycles.device = device
    scene.render.resolution_x = scene.render.resolution_y = 128
    scene.cycles.samples = 32
    scene.render.filepath = f"/tmp/bpy-smoke-{device.lower()}.png"
    t0 = time.time(); bpy.ops.render.render(write_still=True)
    print(f"render[{device}] {time.time()-t0:.1f}s -> {scene.render.filepath}")
EOF
log "DONE. wheel in $WHEEL_DIR"

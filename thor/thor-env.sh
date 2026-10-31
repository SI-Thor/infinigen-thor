#!/usr/bin/env bash
# thor-env.sh - source this on the Thor box. One file, every box-specific
# assumption in one place, so nothing is re-derived from a commit message.
#
#   source thor-env.sh
#
# Why each line exists:
#
# 1. NVCC_APPEND_FLAGS="-arch=sm_110"
#    nvcc on this box DEFAULTS TO sm_75. Any from-source CUDA extension that
#    does not pass -arch gets cubins that cannot load on this GPU:
#        "no kernel image is available for execution on the device"
#    and - worse - sm_87 compiles CLEANLY here and produces a binary that
#    fails silently at run time.
#    scripts/install/compile_terrain.sh passes -arch=native by hand; that
#    fixes one script. NVCC_APPEND_FLAGS (documented by NVIDIA in the CUDA
#    Compiler Driver manual, section "NVCC Environment Variables") fixes every
#    nvcc invocation on the box at once - including ones inside setup.py,
#    CMake projects and third-party dependencies we do not control.
#    Flags are appended AFTER the command line, so an explicit -arch in a
#    script still wins; this is a safety net, not an override.
#    Verify with: nvcc --verbose ... | grep -i sm_
#
# 2. Toolchain paths on NVMe.
#    The external cold shelf is NTFS-over-fuseblk on a USB bridge and it
#    wedged mid-run (2026-10-31): processes D-stated 30+ min on page faults
#    from libpython3.11.so, and after a lazy umount the drive stayed
#    unreadable - fuse remount, kernel ntfs3 and a fresh ntfs-3g daemon all
#    hang until the USB bridge is replugged. Runtime-loadable artifacts must
#    not page from that drive.
#
# 3. TERRAIN_CUDA_ARCH
#    The terrain CUDA kernels are compiled by scripts/install/compile_terrain.sh;
#    it reads this.
#
# 4. A conservative CUDA memory fraction
#    On a unified-memory box the GPU has no VRAM to over-commit against, but
#    Cycles still asks the allocator for a budget. Leaving it at the default
#    makes Cycles reserve as if a big discrete card were attached. Tune with
#    measurements, not with a guess.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export REPO_ROOT

# 1. every nvcc call on this box targets sm_110
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-} -arch=sm_110"
export TERRAIN_CUDA_ARCH="${TERRAIN_CUDA_ARCH:-sm_110}"

# 2. runtime toolchain lives on NVMe, not on the external shelf
export THOR_TOOLCHAIN="${THOR_TOOLCHAIN:-$REPO_ROOT/.toolchain}"
export PATH="$THOR_TOOLCHAIN/python-3.11/bin:$PATH"

# 4. measured, not guessed - adjust once probe-thor.sh has real numbers
export CYCLES_DEVICE_TYPE="${CYCLES_DEVICE_TYPE:-CUDA}"

# 5. Infinigen 2.0 reads this to pick the Cycles device. Do NOT let it choose
#    OPTIX: there are no RT cores on this GPU, so the OptiX renderer is a
#    fallback that will not help. If the wheel ever gains an OptiX denoiser,
#    forcing CUDA here still leaves denoising to the renderer's own setting.
export INFINIGEN_CYCLES_DEVICE_TYPE="${INFINIGEN_CYCLES_DEVICE_TYPE:-CUDA}"

cat <<'MSG'
[thor-env] NVCC_APPEND_FLAGS = $NVCC_APPEND_FLAGS
[thor-env] TERRAIN_CUDA_ARCH = $TERRAIN_CUDA_ARCH
[thor-env] INFINIGEN_CYCLES_DEVICE_TYPE = $INFINIGEN_CYCLES_DEVICE_TYPE
[thor-env] toolchain = $THOR_TOOLCHAIN
MSG

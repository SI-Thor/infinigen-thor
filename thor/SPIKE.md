# SPIKE: Infinigen on the Jetson AGX Thor — phase 1 results

Date: 2026-10-30/31 · Branch: `thor-spike` · Box: Jetson AGX Thor Developer
Kit, 14-core aarch64 (Jetson L4T 6.8 / Ubuntu 24.04), 122 GB unified memory,
Blackwell GPU "NVIDIA Thor" **compute capability 11.0 (sm_110)**, CUDA 13.2 at
`/usr/local/cuda`, driver 595.78, gcc 13.3, cmake 3.28. NVMe had 343 GB free.
No other project services were running; timings below are with the box
otherwise idle (verified with `tegrastats` + `nvidia-smi` sampling, 1 Hz, in
`thor/run-timed.sh`).

## Verdict

**Yes — Infinigen runs natively on the Thor.** Blender 4.2 as the `bpy` Python
module had to be built from source (no prebuilt exists for Linux ARM64), which
cost ~5 patches and ~70 wall-clock minutes, but from there everything the
brief asked for works: Hello Room, Hello World, terrain on CPU, terrain CUDA
kernels compiled for sm_110, and Cycles rendering on the GPU at **7.1× the
CPU's speed**. The port work is narrow and mechanical; the phase-2 fork is
worth it.

## 1. Blender 4.2 on aarch64 (`thor/build-bpy.sh`)

**No prebuilt path exists** (checked 2026-10-30):

- PyPI `bpy==4.2.0` ships `manylinux_2_28_x86_64`, `macosx_11_0_arm64`,
  `macosx_11_0_x86_64`, `win_amd64` — no Linux aarch64 wheel.
- `download.blender.org/release/Blender4.2/`: Linux is `linux-x64.tar.xz`
  only; ARM64 exists only as macOS `.dmg`. (Flathub ships an aarch64
  standalone app, but it is not a `bpy` module and sandboxing the Thor's
  CUDA inside flatpak is not viable.)

So: **build `bpy` 4.2.0 from source** (`WITH_PYTHON_MODULE=ON`), from tag
`v4.2.0`, system libraries from apt, Cycles CUDA precompiled for `sm_110`
only. Produced `bpy-4.2.0-cp311-cp311-manylinux_2_39_aarch64.whl` (70 MB);
it imports in a fresh venv and renders on CPU and CUDA
(`outputs/proof/bpy-smoke-{cpu,gpu}.png`, log `.ext-thor/logs/bpy-smoke.log`).
Build wall time ≈ 70 min including the failed first passes and the OIIO
build; a rerun from the patched tree is ~45–60 min.

What had to change (all captured in `thor/build-bpy.sh` +
`thor/patches/*.patch`):

| Problem | Fix |
|---|---|
| No Python 3.11 in Ubuntu 24.04 (Blender 4.2 embeds exactly 3.11) | build CPython 3.11.13 `--enable-shared` (+ rpath, else the interpreter can't find `libpython3.11.so.1.0`) into `.toolchain/` |
| `CYCLES_CUDA_BINARIES_ARCH` defaults to `sm_30…sm_89`, all rejected by CUDA 13 | set `sm_110` explicitly |
| Cycles `kernel/device/cuda/config.h` stops at `__CUDA_ARCH__<=899` | patch adds a `>=900` branch (8.x-class launch bounds); `kernel_sm_110.cubin` (27 MB) then compiles cleanly under nvcc 13.2 |
| noble's OpenImageIO 2.4 lacks `ImageInput::valid_file(IOProxy*)` used by imbuf | build OIIO **v2.5.11.0** (Blender's own pin) static; re-point cached CMake finds (`-UOPENIMAGEIO_*`) |
| **the box's ffmpeg is NVIDIA's `8.0.1-nvidia1`** (Jetson repo), not Ubuntu's 6.1 | 3 API-removal patches: `AV_INPUT_BUFFER_MIN_SIZE`, `AVFrame::key_frame` + opaque `AVInputFormat`, `AVFrame::pkt_pos/pkt_dts` |
| static OIIO's transitive `-ltiff` dropped by `--as-needed` ordering → `TIFF*` undefined in `bpy/__init__.so` | `CMAKE_CXX_STANDARD_LIBRARIES=-ltiff` (appends after objects) |
| OptiX: no aarch64 SDK | `WITH_CYCLES_DEVICE_OPTIX=OFF`; CUDA-only (fine on Jetson) |
| OpenImageDenoise: no noble package | `WITH_OPENIMAGEDENOISE=OFF`. Infinigen sets `denoise=False` in both base configs, so nothing in this spike needs it; denoising would fail loudly if enabled |
| also auto-disabled by CMake (optional): OSL, NanoVDB, MaterialX, OpenXR, Haru, OpenPGL path guiding | no impact on Infinigen's node-based materials |

Disabled outright to shrink the surface: USD (no noble package), Alembic,
OpenCOLLADA. apt packages installed are listed in the script (≈60, incl.
`libosd-dev` = OpenSubdiv 3.5).

**Operational wall (the big one):** the 1.9 TB external cold shelf is NTFS
mounted via **fuseblk over USB**, and it wedged hard mid-run — an infinigen
process sat in D state for 30+ min blocked on page faults from
`libpython3.11.so` (which lived on that drive through the toolchain rpath);
after `umount -l` the drive stayed unreadable (fuse remount, kernel `ntfs3`,
and a fresh `ntfs-3g` daemon all hang ⇒ the USB bridge itself is stuck until
a physical replug). Consequence, already applied: **runtime-loadable
artifacts (CPython prefix, venv, wheels) live on NVMe in `.toolchain/` and
`.venv/`; only source/build trees stay on the external drive.** The Blender
build tree and the original wheel are on that offline drive; the venv's
installed copy of bpy 4.2.0 (NVMe) is what everything now runs against, and
`thor/build-bpy.sh` regenerates the wheel on any healthy box.
Also: the drive's mount point contains a space, which breaks make/CMake
quoting — everything is addressed through the space-free `.ext-thor` symlink.

## 2. Infinigen install + terrain (`thor/install.sh`)

`pip install -e .[infinigen1,terrain]` against the local wheel. The full
dependency tree resolved on aarch64 with one from-source build:
`python-fcl`, `landlab`, `coacd`, `rtree`, `cvdpack`, `OpenEXR` etc. all have
wheels; `scikit-image==0.19.3` (the `<0.20` pin) has no cp311 aarch64 wheel
and compiles from sdist in ~10 min. Install ≈ 19 min total.

Terrain C++/CUDA for aarch64: `make terrain` works after one one-line change
— `compile_terrain.sh` now passes `-arch=native` to nvcc (override with
`TERRAIN_CUDA_ARCH=sm_XXX`), because nvcc 13's default target (sm_75)
produces cubins the sm_110 GPU refuses to load. All 48 libs build in ~2 min;
`cuobjdump` confirms `arch = sm_110` in the CUDA libs. Runtime proof: Hello
World runs terrain on CPU (documented default), and both the cpu and cuda
`.so`s load via ctypes in the same interpreter as bpy.

## 3. Hello Room (Infinigen-Indoors) — works

Documented dining-room single-room command (`fast_solve.gin singleroom.gin`,
terrain off), then the render task. Logs and 1 Hz box sampling under
`outputs/logs/hello-room-*`:

| Stage | Time | Peak RSS (proc) | Box state |
|---|---|---|---|
| coarse (scene.blend 397 MB) | **534 s** | 2.10 GB | CPU 225%, sys RAM ≤ 47.6 GB |
| render (GPU, 8192 spp, 1280×720, 15 GT passes + beauty) | **1307 s** | 3.03 GB | **GPU util 98%**, CPU 2%, sys RAM ≤ 56.4 GB |

Docs' expectation on a desktop CPU: ~8 min coarse — the Thor matches it.
Outputs: `outputs/indoors/frames/` (Image/AO/DiffCol/… passes), proof render
`outputs/proof/hello-room-render.png`.

## 4. Hello World (Infinigen-Nature) — works

Documented step-by-step desert scene (`desert.gin simple.gin`), seed 0:

| Stage | Time | Peak RSS | Box state |
|---|---|---|---|
| coarse | **214 s** | 1.81 GB | CPU 224% |
| populate + fine_terrain (terrain mesher, CPU) | **57 s** | 1.44 GB | CPU 642% |
| render (GPU) | **48 s** | 2.48 GB | GPU 98% |
| render again for flat GT | **10 s** | 2.14 GB | GPU 48% |

Total ≈ 5.5 min (docs: ~10 min and 16 GB on an M1/desktop). Terrain ran on
CPU per the docs' default; the CUDA terrain libs are built for sm_110 and
load, but were not exercised by a scene run in this spike.

## 5. Cycles CPU vs GPU (`thor/bench_cpu_gpu.py`)

Same frame of the Hello Room scene, same 256 samples, 1280×720, denoising
off, identical scene file loaded fresh per device:

| Device | Time | Speedup |
|---|---|---|
| CPU (14 cores) | **450.1 s** | 1× |
| GPU (CUDA, sm_110 cubins) | **63.4 s** | **7.1×** |

PNGs: `outputs/proof/bench-cpu-s256.png`, `bench-gpu-s256.png` (byte-size
within 0.001% — same converged image). For calibration, the full documented
Hello Room render (8192 samples, all passes) took 21.8 min on the GPU.
`nvidia-smi memory.used` reports `[N/A]` on Jetson (unified memory) — use
tegrastats RAM for GPU-side memory accounting.

## Commits on `thor-spike` (not pushed)

- `2a7317f3` gitignore spike artifacts
- `3668454a` terrain: nvcc `-arch=native` (sm_75 default can't load on sm_110)
- `67f3e673` + `b5dead9b` `thor/build-bpy.sh` + Blender patches (CUDA arch/OIIO/ffmpeg-8/link)
- `f5a5af1b` `thor/install.sh`
- `361fc7cb` runtime toolchain to NVMe after the fuseblk drive wedge
- `dcf6a948` `thor/bench_cpu_gpu.py`, run-timed fixes

`upstream` history untouched; `main` not modified.

## Walls hit (summary)

1. **No Blender/bpy 4.2 for Linux ARM64 anywhere** — solved by building from
   source; 5 small patches (arch list, Blackwell config.h, OIIO 2.5,
   ffmpeg 8 API removals ×3, tiff link).
2. **The external NTFS/fuseblk cold shelf wedged under load and D-stated a
   run for 30+ min**; it is still offline pending a physical replug
   (fuse, ntfs3 and ntfs-3g mounts all hang). Mitigated by moving all
   runtime artifacts to NVMe. The Blender source/build trees and the
   original wheel are parked on it.
3. Ubuntu 24.04 gaps (Python 3.11, OIIO 2.5, OIDN, USD) — solved or
   disabled without affecting Infinigen's core path.
4. No OptiX on aarch64, ever — CUDA-only is the Thor's ceiling for Cycles
   (and it is plenty: 7.1× over 14 CPU cores).

## Phase 2 recommendation

**Port (cheap, mechanical):** carry the five Blender patches (or move to a
Blender 4.2 LTS branch once it gains CUDA-13/Blackwell fixes upstream), the
`-arch=native` terrain change, and the `thor/` scripts into the fork;
add a wheel-build CI job on the Thor so `bpy` aarch64 stops being
hand-built. Rebuild the toolchain wheel into the repo's artifact store once
the external drive is back. Re-enable the pieces disabled for convenience
(OIDN from source if denoising matters; USD if simulator export matters).

**Rewrite / Thor-native (where the wins are):**
- **Keep scenes warm.** Infinigen's batch pipeline spends its time in
  Python single-thread (225% CPU means ~2 of 14 cores) — generation is not
  CPU-bound on the Thor, so the interactive goal (warm scene + in-place
  edits + GPU re-render) is the right shape: the 48 s nature render / 63 s
  256-spp re-render is fast enough for a preview loop.
- **Batch renders across the unified 122 GB:** with GPU render at 98% util
  and ≤3 GB RSS per job, several concurrent Cycles jobs fit easily;
  `manage_jobs` with `num_concurrent` tuned per scene size is the quick win
  for overnight datasets.
- **Move terrain to the GPU for real:** the CUDA kernels now build for
  sm_110 but ran CPU-only here; wiring `Terrain.device='cuda'` (the config
  exists) and validating output parity + timing is a small, high-value task.
- **Denoising:** build OIDN with CUDA support from source, or standardize on
  higher samples / compositor denoise; decide after visual QA.
- **Disk policy:** everything runtime on NVMe, external drive re-formatted
  ext4 (the NTFS/fuseblk combination is a repeat-offender failure mode) and
  used strictly for cold caches/artifacts.

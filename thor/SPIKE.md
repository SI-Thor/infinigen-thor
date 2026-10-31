# thor/SPIKE.md — the Thor box, and what this spike is

*Reconstructed 2026-10-31. `thor/build-bpy.sh` referenced this file from its first
commit (`67f3e673`, 2026-10-30) but it was never committed; the box assumptions lived
only in that commit message. Updated with the fuseblk incident of 2026-10-31.*

## 1. The box

| | |
|---|---|
| Module | **Jetson AGX Thor Developer Kit (T5000)** |
| GPU | 2,560-core Blackwell, 96 fifth-gen Tensor cores, 10 TPCs (MIG) |
| Compute capability | **11.0 → sm_110** (was `sm_101` before CUDA 13.0; that name is hard-rejected) |
| CPU | 14-core Arm Neoverse-V3AE, 64-bit. **SVE is 128-bit (= NEON); no L3.** |
| Memory | 128 GB LPDDR5X, 256-bit, **273 GB/s**, one pool shared by CPU and GPU |
| Power | MAXN (mode 0), 40–130 W module envelope |
| OS | Ubuntu 24.04 (noble), aarch64, 14 cores, passwordless sudo for apt |
| Kernel | 6.8.12-1021-tegra |
| Driver | 595.x (595.78 seen on the box) |
| CUDA | **13.2** toolkit, nvcc 13.2.86 |
| `nvcc` default arch | **sm_75** — the trap. Every from-source CUDA extension needs `-arch=native` (or `sm_110`) |
| Python | 3.12.3 system; **3.11.13 built from source** for bpy (Blender 4.2 embeds 3.11) |
| Storage | external USB SSD, **NTFS over fuseblk**, mounted at `/media/caseydigennaro/TOSHIBA EXT` — the path **contains a space** |
| NVMe | the repo's own disk; `.toolchain/` lives here |

### The space in the path

CPython's `LD_LIBRARY_PATH` handling and several `make` recipes / CMake lists assume a
space-free prefix. **First victim: CPython's unquoted `LD_LIBRARY_PATH`.** Everything is
therefore addressed through a space-free repo-local symlink:

```
.ext-thor  ->  /media/caseydigennaro/TOSHIBA EXT/thor-spike
```

`.ext-thor`, `outputs/`, `.venv/` and `.toolchain/` are gitignored.

### The fuseblk incident (2026-10-31) — read this before scheduling a build

Mid-run, the external cold shelf **wedged hard**:

- an infinigen run **D-stated for 30+ minutes** blocked on page faults from
  `libpython3.11.so` (it lived on that drive, via rpath);
- after a lazy umount the drive **stayed unreadable** — fuse remount, kernel `ntfs3`
  and a fresh `ntfs-3g` daemon all hang, so **the USB bridge itself is stuck until it
  is physically replugged**.

**What changed** (`361fc7cb`): runtime-loadable artifacts moved to NVMe —
`.toolchain/python-3.11` (the CPython prefix venvs interpret from) and
`.toolchain/wheels` (the bpy wheel). `install.sh` now recreates the venv when
`pyvenv.cfg` does not point at `.toolchain`.

**What did not change, and should:** `$EXT_LINK/src/build-bpy` (the ninja build tree),
`$EXT_LINK/src/blender` (the source checkout) and `$EXT_LINK/logs` (the proof
artifacts) are **still on the wedging shelf**. A ninja build is the worst possible
workload for NTFS-over-fuseblk-on-USB — 30–90 minutes of tens of thousands of small
writes and random header reads, every one a FUSE round trip — and the logs that would
explain a failure live there too. **Move `BUILD_DIR` and `LOG_DIR` to `.toolchain/`.**
See `THOR-EXPLOITS.md` E1.

**Also:** the wheel built earlier is on the offline drive; the installed `.venv`
site-packages copy of bpy 4.2.0 is intact on NVMe and is what everything now runs
against. If the shelf is replugged, the wheel can be copied back for provenance.

### What the box is NOT

**No RT cores.** NVIDIA staff on the developer forum (1 Sept 2025): *"There is no RT
cores in AGX T5000."* The module datasheet's "Modern Graphics features: Ray Tracing"
line was a datasheet bug — *"This is a mistake in module data sheet. We will fix it."*

This is a **hardware** fact, and it is the *real* reason the OptiX renderer is off in
`build-bpy.sh` — the stated reason there ("OptiX has no aarch64 SDK") is **false**:
NVIDIA ships the OptiX SDK for Arm. See `THOR-ARCHITECTURE.md` §4 B2.

## 2. What this spike is

**A bootstrap, not a port.** Six commits on `thor-spike`:

1. `2a7317f3` — gitignore the local artifacts
2. `3668454a` — `compile_terrain.sh`: `-arch=native` for the CUDA terrain kernels
3. `67f3e673` — `build-bpy.sh` / `install.sh` / `run-timed.sh` + the Cycles sm_110 patch
4. `b5dead9b` — finish the bpy build end to end; the three ffmpeg patches
5. `f5a5af1b` — install the venv against the locally-built bpy wheel
6. `361fc7cb` — move the runtime toolchain off the wedged drive onto NVMe

Against `main`: **9 files, ~490 insertions.** **No file under `src/infinigen/` or
`src/infinigen2/` is touched.** `main` is byte-identical to upstream
`princeton-vl/Infinigen` (same SHA `224598d4`).

## 3. What was hard, and what it cost

Everything below was discovered by building on the box.

- **No bpy for Linux ARM64 exists — at any version.** Checked exhaustively: PyPI
  `bpy==4.2.0` ships `manylinux_2_28_x86_64` + macOS + Windows; PyPI `bpy` latest
  (5.2.3) ships macOS arm64 + **Windows** arm64, still no Linux aarch64;
  `conda-forge` and the `blender` channel have no `bpy` package at all; Blender's
  download site is `linux-x64` only. Building the Python module from source is the
  only path.
- **CPython 3.11 must be built** (`--enable-shared`, with an rpath to
  `$PY_PREFIX/lib` or the interpreter cannot find `libpython3.11.so.1.0`).
  Ubuntu 24.04 ships 3.12; Blender 4.2 embeds exactly 3.11.
- **OpenImageIO 2.5.11.0, not noble's 2.4.** 2.4 lacks
  `ImageInput::valid_file(IOProxy*)`, which `imbuf` needs. Built static so the wheel
  carries no extra `.so` deps.
- **Static OIIO's transitive `-ltiff` was dropped by `--as-needed` ordering**,
  leaving `TIFF*` undefined in `bpy/__init__.so`. Fixed with
  `CMAKE_CXX_STANDARD_LIBRARIES=-ltiff`, which puts it back *after* the objects.
- **The ffmpeg on this box is NVIDIA's `8.0.1-nvidia1`** (the Jetson repo), not
  Ubuntu's 6.1. Three API-removal patches: `AV_INPUT_BUFFER_MIN_SIZE`;
  `AVFrame::key_frame` + opaque `AVInputFormat`; `AVFrame::pkt_pos/dts`.
- **Cycles 4.2's `kernel/device/cuda/config.h` stops at `__CUDA_ARCH__ <= 899`**;
  sm_110 needs a `>= 900` branch with 8.x-class launch bounds, or the cubins are
  PTX-JIT'd instead of native.
- **`CYCLES_CUDA_BINARIES_ARCH=sm_110` only** — the default arch list starts at
  sm_50, every member of which nvcc 13 rejects.
- **OIDN has no noble package and no aarch64 prebuilt tarball** — but Blender 4.2
  builds OIDN **2.3.0 from source**, so there was never a package to find. The real
  blocker was upstream's own cmake enabling the SYCL/CUDA/HIP devices for every
  non-Windows-ARM platform; fixed by
  `thor/blender-v4.2.0-oidn-arm64-cpu-only.patch`.

## 4. Proof artifacts

Every claim carries its proof, under `$EXT_LINK/logs` — **except** that after the
incident the *build* logs belong on NVMe too (`THOR-EXPLOITS.md` E1).

| file | what it holds |
|---|---|
| `apt.log` | apt install output |
| `cuda-sanity.log` | `device=NVIDIA Thor cc=11.0` + a kernel that returns 1.0 |
| `cpython-{configure,make,install}.log` | the CPython 3.11 build |
| `clone-blender.log`, `blender-configure.log`, `blender-build.log` | the bpy build |
| `oiio-{configure,build,install}.log` | OIIO 2.5.11.0 |
| `wheel.log` | wheel packaging |
| `bpy-smoke.log` | bpy import, device types, **denoiser availability**, CPU + CUDA renders |
| `infinigen-install.log` | the editable install |

`thor/run-timed.sh` adds, per stage: `.out` (stdout+stderr), `.time` (GNU time:
elapsed, peak RSS, CPU%), `.stats` (`tegrastats` 1 Hz), `.gpu` (`nvidia-smi` 1 Hz),
`.summary` (one line).

`thor/probe-thor.sh` produces a single receipt for the *box itself*: GPU visibility
through `nvidia-smi` (including the `nvidia-smi soc -q` Tegra path), `libnvoptix.so`,
the storage substrate, nvcc's default arch, unified memory, the wheel's device types
and denoisers, and the terrain SASS arch. **Run it before changing anything.**

## 5. The rules this box teaches

1. **`nvcc` defaults to sm_75 here.** Every from-source CUDA extension needs
   `-arch=native` (or `sm_110`). `sm_87` compiles *cleanly* and produces a binary
   that will not run. Check with `cuobjdump --list-elf`, not "it compiled".
   `export NVCC_APPEND_FLAGS="-arch=sm_110"` covers the whole box at once.
2. **Never call `.cpu()` on a large tensor into pageable memory.** Measured at
   0.34 GB/s — 11.9 s for 4 GB, and it never fails, it just takes eleven seconds.
   Use a pinned staging buffer, or don't copy at all.
3. **Host `malloc` is a valid device address on this box** (`devicePointer ==
   hostPointer`). Zero-copy costs the same bandwidth as a device buffer
   (~100 GB/s); the win is that the copy *and* the second allocation disappear.
   Wrap with `cupy.cuda.UnownedMemory` / `wp.array(ptr=..., copy=False)` —
   `wp.from_numpy` **copies**.
4. **The box is shared.** Another seat's build held load ≈ 4–5 throughout the lab
   runs. Always report the load alongside a number.
5. **Optimise the algorithm before porting it to the GPU.** Warp beat a *faithful*
   CPU port by 7.9–10.5×; a *restructured* CPU (exploiting the separability of the
   SDF's displacement field) matched the GPU at 0.91–1.53×.
6. **FP4 is not reachable here** through torch 2.15 + torchao 0.18 — 13.69 TFLOPS,
   1.75× the cost of just dequantizing to BF16 and multiplying. FP8 is real
   (1.57–1.75× FP16 at 8192³) but only if the *activations* are quantized too;
   weights-only FP8 measured 0.95×.
7. **The GPU memory is not reportable the desktop way.** `nvidia-smi soc -q` is the
   Tegra path (NVIDIA's own example prints `MemTotal: 128.83 GiB` — a Thor);
   residency accounting comes from `/proc/meminfo` and `tegrastats`.
8. **Do not build on a USB shelf.** It has already taken a 30-minute run with it.

## 6. See also

- `thor/THOR-EXPLOITS.md` — the doors that are already open, with the fixes
- `thor/THOR-ARCHITECTURE.md` — the reverse-engineering report and the blocker list
- `thor/probe-thor.sh` — measure this box; every open question, with receipts
- `builder-ide` → `docs/lab/exp-{unified-memory,warp-terrain,fp8-tensorcores}.md`
  — the three lab experiments behind §5

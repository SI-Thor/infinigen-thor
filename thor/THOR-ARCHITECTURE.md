# Thor-native Infinigen — the architecture, the hardware envelope, and the blockers

**Date:** 2026-10-31 · **Target:** `SI-Thor/infinigen-thor` @ `thor-spike` (`361fc7cb`)
**Method:** clone + branch/diff archaeology, upstream tag inspection (`blender/blender@v4.2.0`),
NVIDIA primary docs/forums, and the team's own `builder-ide/docs/lab/` experiments.

> The actionable version of this document is **`THOR-EXPLOITS.md`** — doors that are
> already unlocked, with the fix attached. This file is the background and the
> evidence.

---

## 1. What the repo actually is

**`main` is byte-identical to upstream.** `SI-Thor/infinigen-thor` main HEAD =
`princeton-vl/Infinigen` main HEAD = `224598d4a2ff95b4fc928e72c222598b7122c698`.

**Upstream moved.** `princeton-nlp/Infinigen` now **404s**; the live upstream is
`princeton-vl/Infinigen` (visible in the merge commit *"…from princeton-vl/vb/tests_on_pypi"*).
Diffing against the old URL reports *"empty repo"*, not *"moved"* — a false-absence trap.

**The Thor work is 6 commits on `thor-spike`** (all 2026-10-30/31), 9 files, ~490 insertions:

| commit | what it does |
|---|---|
| `2a7317f3` | gitignore the local artifacts |
| `3668454a` | `compile_terrain.sh`: `-arch=native` for the CUDA terrain kernels |
| `67f3e673` | `build-bpy.sh` / `install.sh` / `run-timed.sh` + the Cycles sm_110 patch |
| `b5dead9b` | finish the bpy build end to end; the three ffmpeg patches |
| `f5a5af1b` | install the venv against the locally-built bpy wheel |
| `361fc7cb` | **move the runtime toolchain off the wedged external drive onto NVMe** |

**Not one line of Infinigen's own Python has been changed.** Zero files under
`src/infinigen/` or `src/infinigen2/`. The repo name says "port"; the content says
"bootstrap". It gets a working `bpy` onto an aarch64 box — it does not yet move any
processing to the Thor.

`thor/SPIKE.md` is referenced by `build-bpy.sh:14` and **exists in no branch**
(reconstructed in this drop). The consumer is `SI-Thor/builder-ide`: *"the program:
recipe + engine (infinigen-thor)"*, with the Thor doing *"the long, patient, generative
work"* and the laptop as *"the seat"*.

---

## 2. The hardware envelope

**The single most important correction to the project's mental model:**

> **The Jetson AGX Thor T5000 has no RT cores.** NVIDIA staff on the developer forum,
> 1 Sept 2025: *"There is no RT cores in AGX T5000."* The module datasheet's
> *"Modern Graphics features: Ray Tracing"* line was a **datasheet bug** —
> *"This is a mistake in module data sheet. We will fix it."*

| capability | measured / documented | what it means for Infinigen |
|---|---|---|
| **RT cores** | **none** | **OptiX ray tracing is permanently off the table.** Render on CUDA. |
| **OptiX SDK** | **ships for Arm** — "OptiX SDK 9.1 (Windows, Linux, **Arm**)"; legacy page lists 8.1.0 as "(Windows, Linux, Arm)" | `build-bpy.sh`'s stated reason for disabling OptiX is **factually wrong** (see §4 B2). The SDK is available; the *hardware* is not. |
| OptiX **denoiser** | driver-provided, runs on **compute**, no RT cores needed | The cheap path to a denoiser. `probe-thor.sh` answers it in 5 seconds. |
| CUDA cores | 2,560 Blackwell, 1.57 GHz, cc **11.0 = sm_110** | Cycles CUDA rendering works. This is where the GPU win is. |
| Unified memory | 128 GB LPDDR5X @ 273 GB/s, **one pool** | **The real win is capacity** — scenes that OOM a 24 GB desktop card. |
| Tensor cores | 96, 5th-gen; **FP8 e4m3 1.57–1.75× FP16**; **FP4 = measured no** | Only for a *neural* stage. **Not** for Cycles. |
| CPU | 14-core Neoverse-V3AE, **128-bit SVE (= NEON), no L3** | Scene synthesis is CPU-bound and memory-bound. |
| NVENC / NVDEC | hardware video engine | Encode from the unified buffer; skip the PNG round-trip. |
| MIG | 10 TPCs | Real isolation for parallel jobs. |
| `nvidia-smi` | Tegra variant: **`nvidia-smi soc -q`** is *"only available on Tegra Linux"*; its own doc example prints `MemTotal: 128.83 GiB` (a Thor) | GPU memory is **not** reportable the desktop way → Infinigen's detection call breaks (see `THOR-EXPLOITS.md` E5). |

**sm_110 renumbering:** Thor was `sm_101` before CUDA 13.0 and the old name is
**hard-rejected**; PTX **9.0** is the floor. And the nasty one: **`sm_87` compiles
cleanly on the Thor toolkit and produces a binary that will not run** — a silent
failure. Check with `cuobjdump --list-elf`, never *"it compiled"*.

---

## 3. Where the compute is, stage by stage

| stage | runs on today | on the Thor | action |
|---|---|---|---|
| scene synthesis (2.0, bpy scripting) | CPU | 14 V3AE cores, no L3 | algorithmic work first — EXP B's finding |
| terrain surface kernels (1.x) | CUDA or CPU, `Terrain.device='cuda'` | **already parallel** | keep CUDA; `-arch` done ✓ |
| meshing / erosion / marching cubes | CPU (OcMesher, coacd) | little to gain | leave on CPU |
| **Cycles render (2.0)** | CUDA GPU | 2,560 CUDA cores; **122 GB unified ⇒ capacity win** | force CUDA; raise the sample budget once there is a denoiser |
| **denoise** | **NONE in the Thor build** | OIDN CPU (NEON) or OptiX denoiser or a GPU neural denoiser | **the blocker — see B1** |
| video encode | CPU ffmpeg | NVENC | switch to `h264_nvenc` |
| neural stage | — | FP8 + FP8-resident weights | only if the critic/generator lands on the box |

**1.x has a real CUDA path** (`terrain/source/cuda/**/*.cu` → prebuilt `.so`, loaded by
`load_cdll(f"terrain/lib/{device}/surfaces/{name}.so")`, `surface_kernel/core.py:33`,
selected by `Terrain.device='cuda'`, `datagen/job_funcs.py:307,364`).
**2.0 has no GPU path at all** — the only CUDA reference under `src/infinigen2/` is
`CUDA_VISIBLE_DEVICES` in `util/hardware_info.py:73`.
⇒ **open strategic fork: port 1.x forward, or bring 2.0's stages onto the GPU?**

---

## 4. Blockers, ranked by leverage

### B1 — The build has no denoiser, and Infinigen 2.0 **raises** instead of degrading

`build-bpy.sh` sets `WITH_OPENIMAGEDENOISE=OFF` + `WITH_CYCLES_DEVICE_OPTIX=OFF`,
justified by *"infinigen sets denoise=False in both its base configs"*. True of **1.x**
only (`core/init.py:215-221` warns and degrades). **2.0** defaults to `DenoiseMode.BEST`,
requests `IMAGE_DENOISED`/`IMAGE_DENOISED_HDR`, and raises
`RuntimeError("No usable denoiser ...")` ⇒ **a 2.0 Cycles render on the Thor as built
dies, not degrades.**

The build is not blocked by packaging: Blender 4.2 **builds OIDN from source**
(`versions.cmake` pins `oidn-2.3.0.**src**.tar.gz`), and OIDN 2.2.0+ supports
**ARM64/NEON**. The real blocker is upstream's own
`build_files/build_environment/cmake/openimagedenoise.cmake`, which enables
`-DOIDN_DEVICE_SYCL/CUDA/HIP=ON` under the guard `if(NOT BLENDER_PLATFORM_WINDOWS_ARM)` —
whose comment says *"x64 platforms support SyCL, ARM64 don't"* while excluding only
**Windows**-ARM, **not Linux aarch64**. So on Linux ARM64 it tries to build devices it
cannot: SYCL needs oneAPI's DPC++ (an x86_64 binary), and the OIDN CUDA device would
compile for nvcc's default **sm_75**.

→ `thor/blender-v4.2.0-oidn-arm64-cpu-only.patch` forces the CPU device (or
`-DOIDN_DEVICE_CUDA_ARCH=110`). Verified `git apply --check` against
`blender/blender@v4.2.0`. ISPC is **not** a blocker (`ispc.cmake` sets
`-DARM_ENABLED=${BLENDER_PLATFORM_ARM}` on Linux).

**But the cheaper answer is `THOR-EXPLOITS.md` E2: don't build OIDN, patch the raise.**

### B2 — The OptiX rationale in `build-bpy.sh` is **factually wrong** (the decision is right)

`build-bpy.sh:11-12`: *"OptiX has no aarch64 SDK, so CUDA only."* NVIDIA ships the OptiX
SDK for **Arm**. The decision to keep the OptiX **renderer** off is still correct — for
the opposite reason: **no RT cores**. Leaving the wrong reason in the file is dangerous,
because *"no SDK for ARM"* reads as a packaging gap that some future release will fix. It
will not be, because the hardware has no RT cores.

### B3 — The spike targets Infinigen **1.x**; 2.0 is the maintained engine and has no GPU path

`install.sh` installs `[infinigen1,terrain]`. But `main` is Infinigen **2.0**, and
`builder-ide`'s recipe vocabulary maps onto 2.0's model. Meanwhile 2.0 has no GPU path
(§3). **The strategic question is open and nothing in the repo records having been asked.**

### B4 — `thor/SPIKE.md` is referenced and does not exist

`build-bpy.sh:14` points at it; `git log --all -- thor/SPIKE.md` is empty. The box
assumptions lived only in commit `67f3e673`'s message. Reconstructed in this drop.

### B5 — The binaries are sm_110-only, and `-arch=native` needs a GPU at build time

`-arch=native` and `CYCLES_CUDA_BINARIES_ARCH=sm_110` produce cubins with no PTX
fallback ⇒ the wheel is **not portable to any other aarch64 GPU**, and `-arch=native`
**fails on a machine with no GPU present**. The filename
(`bpy-4.2.0-cp311-cp311-manylinux_2_39_aarch64.whl`) reads as generic aarch64 but is
Thor-specific.

### B6 — The build tree is still on the drive that just ate a 30-minute run

`361fc7cb` moved the *runtime* toolchain to NVMe after the external NTFS-over-fuseblk
shelf wedged (D-state page faults for 30+ min; unreadable until the bridge was
replugged). But `BUILD_DIR` and `LOG_DIR` — the ninja tree and the proof artifacts —
are **still on that shelf**. See `THOR-EXPLOITS.md` E1.

---

## 5. The team's own lab numbers (trust these)

- **EXP A unified memory:** host `malloc` **is** a valid device address
  (`devicePointer == hostPointer`). Zero-copy read **100.87 GB/s** vs copy+read
  **2.15× worse**. **`.cpu()` into pageable memory = 0.34 GB/s — 11.9 s for 4 GB, and
  it never fails, it just takes eleven seconds.** Wrap instead:
  `cupy.cuda.UnownedMemory` / `wp.array(ptr=..., copy=False)`; **`wp.from_numpy` copies.**
- **EXP B Warp:** works on aarch64/sm_110, **native sm_110 SASS, no PTX JIT**. Beats a
  *faithful* CPU port 7.9–10.5× (3D SDF) / 10.2–14.2× (2D fBm) — but a *restructured*
  CPU (separable displacement field) **matches** the GPU at 0.91–1.53×.
  **Optimise the algorithm before porting it.**
- **EXP C FP8:** FP8 e4m3 **1.57–1.75× FP16** at 8192³, halves weight memory.
  **FP4 = measured NO** (13.69 TFLOPS; 1.75× the cost of dequantising to BF16 and
  multiplying). Weights-only FP8 quantisation = **0.95×** — the fp32→fp8 activation
  cast eats the win.

---

## 6. Provenance

- `SI-Thor/infinigen-thor` — cloned; all five branches compared; diffstat and
  per-commit messages read.
- `princeton-vl/Infinigen` — cloned for the tip-SHA comparison. (`princeton-nlp/Infinigen`
  now 404s — upstream moved.)
- `blender/blender@v4.2.0` — read `build_files/build_environment/cmake/versions.cmake`,
  `openimagedenoise.cmake`, `ispc.cmake`, `build_files/cmake/Modules/FindOpenImageDenoise.cmake`,
  `intern/cycles/CMakeLists.txt`, `intern/cycles/integrator/denoiser_optix.cpp`.
- `SI-Thor/builder-ide` — `README.md` and `docs/lab/exp-{unified-memory,warp-terrain,fp8-tensorcores}.md`.
- NVIDIA — OptiX downloads page (ARM support); Jetson AGX Thor product page; developer
  forum *"Thor & RT cores"* (staff answer, Sept 2025); `nvidia-smi` documentation
  (Tegra-only `nvidia-smi soc -q`); CUDA Compiler Driver manual (§NVCC Environment
  Variables — `NVCC_PREPEND_FLAGS` / `NVCC_APPEND_FLAGS`).
- PyPI / Anaconda — `bpy` wheel and conda-forge availability, checked exhaustively.
- `qdox-labs/thor-hacks` — measured sm_110 facts (sm_101→sm_110, PTX 9.0 floor,
  sm_87 silent failure).

*No claim in this document was made from memory alone; every hardware statement cites
an NVIDIA primary source or a measurement this team already recorded.*

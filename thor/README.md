# Thor/Infinigen — reverse-engineering + exploit drop

**Date:** 2026-10-31 · **Target:** `SI-Thor/infinigen-thor` @ `thor-spike` (`361fc7cb`)

| file | what it is |
|---|---|
| **`THOR-MAKING-NOT-CONSUMING.md`** | **Start here for the research.** Perception and generation are two readings of one score field — and the naive closed-loop objective certifies physically impossible worlds. **All the mathematics is measured, with receipts** (`thor/experiments/receipts/`). |
| **`experiments/`** | Four scripts, **numpy only** — no torch, no GPU, no dataset. A claim that needs a GPU is not yet a claim. Each carries a control that can fail, and each of them did fail during development. |
| **`THOR-EXPLOITS.md`** | Nine doors that are already unlocked, ordered by time saved, each with its fix. |
| `THOR-ARCHITECTURE.md` | The background: what the port actually is, the Thor's real hardware envelope, the compute map, six blockers ranked by leverage. |
| `probe-thor.sh` | The team's pipeline **gate** (from `thor-spike`): PASS/FAIL per capability over the venv — cuda, umem, nvcc-arch, denoiser, cycles, sm_110 SASS. Exits non-zero on a missing capability. |
| `probe-thor-receipts.sh` | The **measurement** probe: a written receipt per probe, covering the five things the gate does not — `libnvoptix.so` in the driver, `nvidia-smi soc -q`, `NVCC_APPEND_FLAGS`, the storage substrate (fuseblk/USB), and host-malloc device-addressability. |
| `thor-env.sh` | `source` it: `NVCC_APPEND_FLAGS=-arch=sm_110` and the NVMe toolchain paths, in one place. |
| `patch-infinigen2-denoise-fallback.patch` | 3-line fix to Infinigen 2.0: warn and continue when no denoiser exists, instead of `raise RuntimeError`. **Unblocks the 2.0 pipeline tonight, with no bpy rebuild.** |
| `build-bpy.sh` | Updated `thor/build-bpy.sh`: corrected OptiX rationale, **denoiser enabled**, and a smoke test that *asserts* a denoiser exists. |
| `blender-v4.2.0-oidn-arm64-cpu-only.patch` | Patches Blender 4.2's `openimagedenoise.cmake` so OIDN builds CPU-only on aarch64. Verified `git apply --check` against `blender/blender@v4.2.0`. |
| `SPIKE.md` | The box assumptions `build-bpy.sh` has referenced since day one — plus the fuseblk incident that just cost a 30-minute run. |

## The three things that matter most

0. **The naive closed-loop objective is degenerate, and it is measured.**
   `‖A(G(y)) − y‖` alone pins the generator *somewhere* on the fiber
   `F(y) = {x : A(x) = y}` and says nothing about *which* point. A
   deterministic inverse trained on it converges to the **centre of the
   fiber** — a point that satisfies the measurement and exists in no scene.
   Measured: pixel distillation is **11x** off-manifold at a *perfect* round
   trip; consistency-only is **19x** off-manifold with an even *better* round
   trip. **A consistency-only critic passes every frame of both.** The fix is
   a second term — the prior's plausibility — and the two must be balanced in
   gradient scale (measured: 4.4e-5 apart). And only a **stochastic**
   generator can be plausible and consistent at once, which makes the
   fiber-width gate a **necessity**, not an optimisation.


1. **`main` is byte-identical to upstream** (`princeton-vl/Infinigen`, SHA `224598d4`).
   The port is 6 commits of build infrastructure; **not one line of Infinigen's Python
   has been changed.** Upstream moved from `princeton-nlp` — diffing against the old
   URL reports "empty repo", not "moved".
2. **The Thor has no RT cores** (NVIDIA staff, Sept 2025). OptiX rendering is off the
   table permanently — and `build-bpy.sh`'s stated reason (*"OptiX has no aarch64
   SDK"*) is **false**; NVIDIA ships the SDK for Arm. The hardware is the reason.
   Design against *capacity and CUDA throughput*, not the 2,070 TFLOPS FP4 headline
   (measured no on this stack).
3. **The build has no denoiser and Infinigen 2.0 hard-fails without one.**
   `render_cycles.py` raises rather than degrades; 1.x only warns. But the cheap fix is
   a **3-line patch**, not an OIDN build: `patch-infinigen2-denoise-fallback.patch`.
4. **GPU detection asks for the name and the memory in one query.**
   `infinigen2/util/hardware_info.py` calls `nvidia-smi --query-gpu=name,memory.total`.
   On a unified-memory device (`memory.total` → `[N/A]`) the entry is malformed,
   and if the driver rejects that field the **GPU's name is lost with it**.
   Measured on Thor/Orin/dGPU with a fake `nvidia-smi` — see
   `tests/infinigen2/util/test_hardware_info.py`.
   **Fixed in this branch**: the two queries are separate, `[N/A]` is dropped,
   and a GPU with real VRAM keeps its exact old `"NAME, MEMORY"` format.
5. **The build tree is still on the drive that just ate a 30-minute run.**
   `361fc7cb` moved the runtime toolchain to NVMe but left the ninja build tree and
   the logs on the wedging fuseblk/USB shelf. Two lines to fix.

## Verified negative result (so nobody spends a day on it)

**There is no prebuilt `bpy` for Linux aarch64, at any version.** PyPI 4.2.0 and 5.2.3,
conda-forge, the `blender` channel and download.blender.org all checked with receipts.
Building from source is genuinely the only path.

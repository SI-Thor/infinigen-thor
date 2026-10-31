# Thor/Infinigen — reverse-engineering + exploit drop

**Date:** 2026-10-31 · **Target:** `SI-Thor/infinigen-thor` @ `thor-spike` (`361fc7cb`)

| file | what it is |
|---|---|
| **`THOR-EXPLOITS.md`** | **Start here.** Nine doors that are already unlocked, ordered by time saved, each with its fix. |
| `THOR-ARCHITECTURE.md` | The background: what the port actually is, the Thor's real hardware envelope, the compute map, six blockers ranked by leverage. |
| `probe-thor.sh` | Runs on the Thor. Measures GPU visibility, `libnvoptix.so`, the storage substrate, nvcc's default arch, unified memory, the wheel's device types and denoisers, terrain SASS arch. Exits non-zero on a missing capability. |
| `thor-env.sh` | `source` it: `NVCC_APPEND_FLAGS=-arch=sm_110` and the NVMe toolchain paths, in one place. |
| `patch-infinigen2-denoise-fallback.patch` | 3-line fix to Infinigen 2.0: warn and continue when no denoiser exists, instead of `raise RuntimeError`. **Unblocks the 2.0 pipeline tonight, with no bpy rebuild.** |
| `build-bpy.sh` | Updated `thor/build-bpy.sh`: corrected OptiX rationale, **denoiser enabled**, and a smoke test that *asserts* a denoiser exists. |
| `blender-v4.2.0-oidn-arm64-cpu-only.patch` | Patches Blender 4.2's `openimagedenoise.cmake` so OIDN builds CPU-only on aarch64. Verified `git apply --check` against `blender/blender@v4.2.0`. |
| `SPIKE.md` | The box assumptions `build-bpy.sh` has referenced since day one — plus the fuseblk incident that just cost a 30-minute run. |

## The five things that matter most

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
4. **The scheduler is about to think the Thor has no GPU.**
   `infinigen2/util/hardware_info.py` calls `nvidia-smi --query-gpu=name,memory.total`
   and keeps the result only if the exit code is 0. On Tegra, `nvidia-smi` is a
   different implementation and GPU memory is shared with the CPU — and your own
   research already says so. `probe-thor.sh` tests this exact call.
5. **The build tree is still on the drive that just ate a 30-minute run.**
   `361fc7cb` moved the runtime toolchain to NVMe but left the ninja build tree and
   the logs on the wedging fuseblk/USB shelf. Two lines to fix.

## Verified negative result (so nobody spends a day on it)

**There is no prebuilt `bpy` for Linux aarch64, at any version.** PyPI 4.2.0 and 5.2.3,
conda-forge, the `blender` channel and download.blender.org all checked with receipts.
Building from source is genuinely the only path.

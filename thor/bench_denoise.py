#!/usr/bin/env python3
"""thor/bench_denoise.py - same frame, low samples, denoise OFF vs ON (OIDN),
against a high-sample reference; reports render time and noise (PSNR to the
reference) for each arm.

Usage: <venv>/python thor/bench_denoise.py scene.blend outdir [samples] [ref_samples]

Defaults: 128 samples per arm, 1024-sample reference, Cycles on the Thor GPU
(CUDA; OptiX cannot exist here - no RT cores). OIDN denoises on the CPU
threads after the CUDA path trace, which is exactly the deployment shape
Infinigen would use on this box. PNGs are read back with skimage for PSNR:
"Render Result" pixels are not reliably populated in background mode.
"""
import sys
import time

import bpy
import numpy as np
from skimage.io import imread

scene_path, outdir = sys.argv[1], sys.argv[2]
samples = int(sys.argv[3]) if len(sys.argv) > 3 else 128
ref_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 1024

# bpy.app.build_options has no openimagedenoise field in 4.2; the Cycles
# enum assignment is the operative check (raises without OIDN).
bpy.data.scenes["Scene"].cycles.denoiser = "OPENIMAGEDENOISE"

bpy.ops.wm.open_mainfile(filepath=scene_path)
scene = bpy.context.scene
assert scene.render.engine == "CYCLES", scene.render.engine

prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "CUDA"
prefs.get_devices()
for d in prefs.devices:
    print(f"# device: {d.name} type={d.type}", flush=True)
    d.use = d.type == "CUDA"
scene.cycles.device = "GPU"


def render(tag, use_denoise):
    scene.cycles.samples = ref_samples if tag == "ref" else samples
    scene.cycles.use_denoising = use_denoise
    if use_denoise:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = f"{outdir}/denoise-{tag}-s{scene.cycles.samples}.png"
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    dt = time.time() - t0
    print(
        f"RESULT arm={tag} samples={scene.cycles.samples} "
        f"denoise={int(use_denoise)} seconds={dt:.1f} out={scene.render.filepath}",
        flush=True,
    )
    return scene.render.filepath


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) / 255 - b.astype(np.float64) / 255) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


paths = {
    "ref": render("ref", False),
    "noidn": render("noidn", False),
    "oidn": render("oidn", True),
}
imgs = {k: imread(v)[:, :, :3] for k, v in paths.items()}

print(f"RESULT psnr_noisy_vs_ref={psnr(imgs['noidn'], imgs['ref']):.2f}dB", flush=True)
print(f"RESULT psnr_denoised_vs_ref={psnr(imgs['oidn'], imgs['ref']):.2f}dB", flush=True)

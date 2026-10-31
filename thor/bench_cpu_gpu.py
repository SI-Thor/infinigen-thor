#!/usr/bin/env python3
"""thor/bench_cpu_gpu.py - render the SAME frame of an existing scene on
Cycles CPU vs Cycles CUDA (the Thor's GPU), same samples/resolution.

Usage: .venv/bin/python thor/bench_cpu_gpu.py scene.blend outdir [samples]

Prints one timing line per device; PNGs are written next to it. OptiX does
not exist on aarch64, so CUDA is the GPU device.
"""
import sys
import time

import bpy

scene_path, outdir = sys.argv[1], sys.argv[2]
samples = int(sys.argv[3]) if len(sys.argv) > 3 else 256

bpy.ops.wm.open_mainfile(filepath=scene_path)
scene = bpy.context.scene
assert scene.render.engine == "CYCLES", scene.render.engine
scene.cycles.samples = samples
scene.cycles.use_denoising = False  # no OIDN in this build
scene.render.filepath = f"{outdir}/bench"

prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "CUDA"
prefs.get_devices()
for d in prefs.devices:
    print(f"# device: {d.name} type={d.type}", flush=True)
    d.use = d.type == "CUDA"

for device in ("CPU", "GPU"):
    scene.cycles.device = device
    scene.render.filepath = f"{outdir}/bench-{device.lower()}-s{samples}.png"
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    dt = time.time() - t0
    x, y = scene.render.resolution_x, scene.render.resolution_y
    print(
        f"RESULT device={device} samples={samples} res={x}x{y} "
        f"seconds={dt:.1f} out={scene.render.filepath}",
        flush=True,
    )

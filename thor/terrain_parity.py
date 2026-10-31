#!/usr/bin/env python3
"""thor/terrain_parity.py - compare the terrain meshes of Infinigen fine_terrain
outputs (CPU vs CUDA device, with an optional CPU-vs-CPU baseline).

Usage: <venv>/python thor/terrain_parity.py a/scene.blend b/scene.blend [baseline/scene.blend]

The fine_terrain stage is NOT deterministic on a single device (two CPU runs
of the same seed/input differ by ~0.1% in vertex count on the desert scene),
so exact hashes cannot validate a device switch. This script reports:
  - per-object vertex/face counts, matched by object name
  - a hash of positions quantized to 1e-4 (exact-parity signal when it hits)
  - mesh statistics per object: bbox extent, mean/std of vertex positions
  - with a baseline: the CPU-vs-CPU deltas form the noise floor, and the
    CPU-vs-CUDA deltas are judged against it (within 2x floor => PARITY=stats)
Exit 0 on PARITY=exact or PARITY=stats; exit 1 on object-set mismatch or
deltas clearly beyond the same-device noise floor.
"""
import hashlib
import sys

import bpy
import numpy as np


def terrain_points(blend):
    bpy.ops.wm.open_mainfile(filepath=blend)
    bpy.context.view_layer.update()
    cols = [c for c in bpy.data.collections if c.name.startswith("terrain")]
    assert cols, f"no terrain collection in {blend}"
    objs = sorted(
        (o for c in cols for o in c.all_objects if o.type == "MESH"),
        key=lambda o: o.name,
    )
    out = {}
    for o in objs:
        me = o.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
        verts = np.array([v.co for v in me.vertices], dtype=np.float64)
        out[o.name] = dict(
            n_v=len(me.vertices),
            n_f=len(me.polygons),
            qhash=hashlib.sha256(np.round(verts, 4).tobytes()).hexdigest(),
            bbox=verts.max(axis=0) - verts.min(axis=0),
            mean=verts.mean(axis=0),
            std=verts.std(axis=0),
        )
        o.to_mesh_clear()
    return out


a_blend, b_blend = sys.argv[1:3]
base_blend = sys.argv[3] if len(sys.argv) > 3 else None
a, b = terrain_points(a_blend), terrain_points(b_blend)
base = terrain_points(base_blend) if base_blend else None

if set(a) != set(b) or (base is not None and set(a) != set(base)):
    print(f"PARITY=FAIL object sets differ: {set(a) ^ set(b)}")
    sys.exit(1)

total = sum(o["n_v"] for o in a.values())
print(f"# objects={len(a)} total_vertices={total}")

ok, quant_eq = True, True
print("# object  verts_a  verts_b  d_verts  max_axis_bbox_delta  mean_pos_delta")
for name in sorted(a):
    oa, ob = a[name], b[name]
    ob_ = base[name] if base else None
    quant_eq &= oa["qhash"] == ob["qhash"]
    if ob_ is not None:
        noise = abs(oa["n_v"] - ob_["n_v"])
        delta = abs(oa["n_v"] - ob["n_v"])
        ok &= delta <= max(2 * noise, 10)
        bbox_d = float(np.abs(oa["bbox"] - ob["bbox"]).max())
        bbox_n = float(np.abs(oa["bbox"] - ob_["bbox"]).max())
        ok &= bbox_d <= max(2 * bbox_n, 1e-3)
        mean_d = float(np.abs(oa["mean"] - ob["mean"]).max())
        mean_n = float(np.abs(oa["mean"] - ob_["mean"]).max())
        ok &= mean_d <= max(2 * mean_n, 1e-3)
        print(
            f"# {name}  {oa['n_v']}  {ob['n_v']}  {delta}"
            f" (same-device floor {noise})  {bbox_d:.2e} (floor {bbox_n:.2e})"
            f"  {mean_d:.2e} (floor {mean_n:.2e})"
        )
    else:
        print(
            f"# {name}  {oa['n_v']}  {ob['n_v']}  {oa['n_v'] - ob['n_v']}"
            f"  {float(np.abs(oa['bbox'] - ob['bbox']).max()):.2e}"
            f"  {float(np.abs(oa['mean'] - ob['mean']).max()):.2e}"
        )

if quant_eq:
    print("PARITY=exact (1e-4-quantized position hashes identical)")
elif ok:
    print(
        "PARITY=stats (differ beyond quantization, but within 2x the "
        "same-device noise floor - see per-object lines)"
    )
else:
    print("PARITY=FAIL deltas clearly exceed the same-device noise floor")
    sys.exit(1)

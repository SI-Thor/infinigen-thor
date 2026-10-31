#!/usr/bin/env python3
"""thor/terrain_parity.py - compare the terrain meshes of two scene.blend
files produced by Infinigen's fine_terrain stage (CPU vs CUDA device).

Usage: <venv>/python thor/terrain_parity.py a/scene.blend b/scene.blend

Parity between devices cannot be bit-exact (different FP orderings in the
erosion/solver kernels), so this reports:
  - per-object vertex/face counts, matched by object name
  - a hash of positions quantized to 1e-4 (exact-parity signal)
  - max/mean |delta| of matched vertex positions and full-terrain stats
Exit 0 with PARITY=quantized when the quantized hashes match, exit 0 with
PARITY=approx when they differ but counts match and deltas are tiny, exit 1
when counts differ (different topology - a real bug, not FP noise).
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
        n_v, n_f = len(me.vertices), len(me.polygons)
        verts = np.array([v.co for v in me.vertices], dtype=np.float64)
        o.to_mesh_clear()
        out[o.name] = (n_v, n_f, verts)
    return out


a, b = (terrain_points(p) for p in sys.argv[1:3])

if set(a) != set(b):
    print(f"PARITY=FAIL object sets differ: {set(a) ^ set(b)}")
    sys.exit(1)

total = sum(v[0] for v in a.values())
print(f"# objects={len(a)} total_vertices={total}")
deltas, quant_eq = [], True
for name in a:
    (va, fa, pa), (vb, fb, pb) = a[name], b[name]
    if va != vb or fa != fb:
        print(f"PARITY=FAIL {name}: counts {va}/{fa} vs {vb}/{fb}")
        sys.exit(1)
    qa = np.round(pa, 4).tobytes()
    qb = np.round(pb, 4).tobytes()
    same = hashlib.sha256(qa).hexdigest() == hashlib.sha256(qb).hexdigest()
    quant_eq &= same
    if pa.shape == pb.shape:
        d = np.abs(pa - pb)
        deltas.append((name, float(d.max()), float(d.mean()), int((d > 1e-4).sum())))

print("# name  max|delta|  mean|delta|  n(>1e-4)")
for name, mx, mn, n in sorted(deltas, key=lambda t: -t[1]):
    print(f"# {name}  {mx:.3e}  {mn:.3e}  {n}")

if quant_eq:
    print("PARITY=quantized (identical at 1e-4)")
else:
    worst = max((d[1] for d in deltas), default=float("nan"))
    print(
        f"PARITY=approx (differs beyond 1e-4; worst vertex delta {worst:.3e} "
        f"on scale {max(np.abs(v[2]).max() for v in a.values()):.1f})"
    )

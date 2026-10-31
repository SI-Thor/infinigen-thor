#!/usr/bin/env python3
"""thor/p3_compare.py - exact comparison of terrain objects across scene.blends.

Phase-3 determinism check. Mesh.export_blender() runs make_unique() (np.lexsort
canonicalization), so the vertex/face order in the saved blend is canonical and
a direct array comparison is a SET comparison: any difference is a genuine
geometric difference, not an ordering artifact.

Usage: p3_compare.py LABEL blend1 blend2 [blend3 ...]
Prints per-object: vertex/face counts, EXACT float32 hash agreement across all
inputs, and for the largest differing object a bit of statistics.
Exit 1 if any terrain object differs at all across the inputs.
"""
import hashlib
import sys

import bpy
import numpy as np


def terrain_meshes(blend):
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
        verts = np.array([v.co for v in me.vertices], dtype=np.float32)
        faces = np.array([p.vertices for p in me.polygons], dtype=np.int64)
        out[o.name] = dict(
            n_v=len(verts),
            n_f=len(faces),
            vhash=hashlib.sha256(verts.tobytes()).hexdigest(),
            fhash=hashlib.sha256(faces.tobytes()).hexdigest(),
            verts=verts,
        )
        o.to_mesh_clear()
    return out


def main():
    label = sys.argv[1]
    blends = sys.argv[2:]
    assert len(blends) >= 2
    runs = [terrain_meshes(b) for b in blends]
    names = sorted(set.intersection(*(set(r) for r in runs)))
    print(f"[{label}] comparing {len(blends)} blends, {len(names)} shared objects")
    any_diff = False
    for name in names:
        objs = [r[name] for r in runs]
        counts = [o["n_v"] for o in objs]
        vh = [o["vhash"] for o in objs]
        fh = [o["fhash"] for o in objs]
        same = len(set(vh)) == 1 and len(set(fh)) == 1 and len(set(counts)) == 1
        tag = "IDENTICAL" if same else "DIFFERS"
        if not same:
            any_diff = True
        print(f"  {name:32s} {tag:10s} verts={counts}")
    if any_diff:
        # characterize the first differing object a bit deeper
        for name in names:
            objs = [r[name] for r in runs]
            if len({o["vhash"] for o in objs}) > 1:
                ref = objs[0]
                print(f"  detail vs first run for {name}:")
                for i, o in enumerate(objs[1:], 2):
                    if o["n_v"] == ref["n_v"]:
                        d = np.abs(o["verts"] - ref["verts"])
                        ndiff = int((d > 0).any(axis=1).sum())
                        print(
                            f"    run{i}: same count, {ndiff}/{ref['n_v']} vertices "
                            f"differ bitwise, max|d|={d.max():.3e}"
                        )
                    else:
                        print(f"    run{i}: count differs ({o['n_v']} vs {ref['n_v']})")
                break
        sys.exit(1)
    print(f"[{label}] ALL TERRAIN OBJECTS BIT-IDENTICAL (float32, canonical order)")


if __name__ == "__main__":
    main()

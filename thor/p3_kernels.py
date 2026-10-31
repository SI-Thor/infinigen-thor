#!/usr/bin/env python3
"""thor/p3_kernels.py - phase 3 E2/E3: with the REAL desert terrain elements,
(a) are the SDF kernels invariant to batch order / batch composition?
(b) does an in-process double fine export differ, and how (order vs geometry)?

Runs in .venv-p2. Usage:
  p3_kernels.py order    <coarse_dir> [cpu|cuda]   # E3 kernel invariance
  p3_kernels.py double   <coarse_dir> [cpu|cuda]   # E2 same-process export x2
"""
import pickle
import sys
from pathlib import Path

import numpy as np

import infinigen_examples.generate_nature  # noqa: F401  registers gin configurables
from infinigen.core import init
from infinigen.core.placement import camera as cam_util
from infinigen.core.util.organization import Transparency
import bpy

CONFIGS = [
    "base_nature.gin",
    "desert.gin",
    "performance/dev.gin",
    "disable_assets/no_creatures.gin",
    "performance/fast_terrain_assets.gin",
]


def setup(coarse_dir, device):
    if device == "cuda":
        overrides = ["Terrain.device=cuda"]
    else:
        overrides = []
    init.apply_gin_configs(
        configs=CONFIGS,
        overrides=overrides,
        config_folders="infinigen_examples/configs_nature",
        mandatory_folders=None,
        mutually_exclusive_folders=None,
    )
    bpy.ops.wm.open_mainfile(filepath=str(coarse_dir / "scene.blend"))
    bpy.context.scene.frame_set(1)
    with open(coarse_dir / "assets" / "info.pickle", "rb") as f:
        info = pickle.load(f)
    return info


def make_terrain(coarse_dir, seed, device):
    from infinigen.terrain import Terrain

    Terrain.instance = None
    return Terrain(
        seed,
        task=["fine_terrain"],
        asset_folder="",
        asset_version="",
        on_the_fly_asset_folder=Path(coarse_dir) / "assets",
        device=device,
        height_offset=info["height_offset"],
        whole_bbox=info["whole_bbox"],
    )


def eval_kernels(kernels, XYZ):
    sdfs = []
    for kernel in kernels:
        sdf = kernel(XYZ, sdf_only=1)["sdf"]
        sdfs.append(sdf)
    return np.stack(sdfs, -1).astype(np.float64).min(axis=-1)


def mode_order(coarse_dir, device):
    global info
    info = setup(coarse_dir, device)
    terrain = make_terrain(coarse_dir, 0, device)
    opaque = [
        e for e in terrain.elements_list if e.transparency == Transparency.Opaque
    ]
    print(f"opaque elements: {[e.__class__.name for e in opaque]}")

    rng = np.random.RandomState(42)
    # positions a la the fine mesher: inside the populated bounds, plus some
    # stragglers far out
    P = rng.uniform(-80, 80, size=(20000, 3))
    P[:, 2] = rng.uniform(-30, 60, size=len(P))

    s_ref = eval_kernels(opaque, P.copy())

    perm = rng.permutation(len(P))
    s_perm = eval_kernels(opaque, P[perm].copy())
    inv = np.argsort(perm)
    d_order = np.abs(s_perm[inv] - s_ref)
    n_bitdiff = int((s_perm[inv] != s_ref).sum())

    half = len(P) // 2
    s_split = np.concatenate(
        [eval_kernels(opaque, P[:half].copy()), eval_kernels(opaque, P[half:].copy())]
    )
    n_splitdiff = int((s_split != s_ref).sum())
    d_split = np.abs(s_split - s_ref)

    print(f"[E3/{device}] batch-ORDER invariance: {n_bitdiff}/{len(P)} points differ "
          f"bitwise, max|d|={d_order.max():.3e}")
    print(f"[E3/{device}] batch-SIZE invariance:  {n_splitdiff}/{len(P)} points differ "
          f"bitwise, max|d|={d_split.max():.3e}")
    ok = n_bitdiff == 0 and n_splitdiff == 0
    print(f"[E3/{device}] kernels are {'ORDER-INVARIANT' if ok else 'ORDER-SENSITIVE'}")
    sys.exit(0 if ok else 2)


def mesh_sig(mesh):
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces)
    import hashlib

    raw = hashlib.sha256(v.tobytes()).hexdigest()[:16]
    order = np.lexsort(v.T)
    canon = hashlib.sha256(v[order].tobytes()).hexdigest()[:16]
    canonsorted = np.array2string(v[order][:3, :], precision=8)
    return dict(n=len(v), m=len(f), raw=raw, canon=canon, v=v)


def mode_double(coarse_dir, device):
    global info
    info = setup(coarse_dir, device)
    terrain = make_terrain(coarse_dir, 0, device)
    terrain.sample_surface_templates()
    terrain.surfaces_into_sdf()
    camera_rigs = cam_util.get_camera_rigs()
    cameras = [c for rig in camera_rigs for c in rig.children]

    sigs = []
    for it in range(2):
        meshes, _ = terrain.export(mesher_backend="SphericalMesher", cameras=cameras)
        sigs.append({k: mesh_sig(m) for k, m in meshes.items()})
        print(f"[E2/{device}] export {it}: " + ", ".join(
            f"{k}: n={s['n']} raw={s['raw']}" for k, s in sigs[-1].items()))

    for name in sigs[0]:
        a, b = sigs[0][name], sigs[1][name]
        if a["n"] != b["n"] or a["canon"] != b["canon"]:
            if a["n"] == b["n"]:
                d = np.abs(a["v"] - b["v"])
                nd = int((d > 0).any(axis=1).sum())
                print(f"[E2/{device}] {name}: GEOMETRIC DIFF (count same, "
                      f"{nd}/{a['n']} verts differ, max|d|={d.max():.3e})")
            else:
                print(f"[E2/{device}] {name}: COUNT DIFF {a['n']} vs {b['n']}")
            sys.exit(2)
        elif a["raw"] != b["raw"]:
            print(f"[E2/{device}] {name}: order-only difference (canonical hash equal)")
        else:
            print(f"[E2/{device}] {name}: BIT-IDENTICAL")
    print(f"[E2/{device}] double export verdict above")


def mode_probe(coarse_dir, device):
    """One instrumented fine export per process; log hashes of every stage to
    $P3_LOG so two processes can be diffed stage by stage."""
    global info
    import hashlib

    info = setup(coarse_dir, device)
    log = open(__import__("os").environ["P3_LOG"], "w")

    def h(arr):
        return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]

    terrain = make_terrain(coarse_dir, 0, device)
    for e in terrain.elements_list:
        parts = []
        for attr in ("int_params", "float_params", "int_params2", "float_params2"):
            v = getattr(e, attr, None)
            parts.append(h(v) if v is not None and np.size(v) else "empty")
        log.write(f"params {e.__class__.name}: {' '.join(parts)}\n")

    terrain.sample_surface_templates()
    terrain.surfaces_into_sdf()
    for e in terrain.elements_list:
        for i, surf in enumerate(e.displacement):
            for dt, vals in sorted(surf.imp_values_of_type.items()):
                log.write(f"surfparams {e.__class__.name}#{i} {dt}: {h(vals)}\n")

    camera_rigs = cam_util.get_camera_rigs()
    cameras = [c for rig in camera_rigs for c in rig.children]
    from infinigen.terrain.utils import get_caminfo

    full_info, cam_pose, fov, H, W, _ = get_caminfo(cameras)
    log.write(f"caminfo: pose={h(cam_pose)} fov={fov} H={H} W={W}\n")

    # instrument the two mesher classes: hash every kernel batch in and out
    from infinigen.terrain.mesher.cube_spherical_mesher import CubeSphericalMesher
    from infinigen.terrain.mesher.frontview_spherical_mesher import (
        FrontviewSphericalMesher,
    )

    state = {"call": 0}

    def instrument(cls, label):
        orig = cls.__call__

        def wrapped(self, kernels):
            orig_kc = self.kernel_caller
            seq = []

            def kc(kernels_, XYZ):
                out = orig_kc(kernels_, XYZ)
                seq.append(f"{len(XYZ)}:{h(XYZ)}->{h(out)}")
                return out

            self.kernel_caller = kc
            mesh = orig(self, kernels)
            self.kernel_caller = orig_kc
            v = np.asarray(mesh.vertices, dtype=np.float64)
            order = np.lexsort(v.T)
            log.write(
                f"{label} batch0..{len(seq) - 1}: " + " | ".join(seq) + "\n"
            )
            log.write(
                f"{label} mesh: n={len(v)} m={len(mesh.faces)} "
                f"raw={h(v)} canon={h(v[order])}\n"
            )
            return mesh

        cls.__call__ = wrapped

    instrument(CubeSphericalMesher, "cube")
    instrument(FrontviewSphericalMesher, "front")

    meshes, _ = terrain.export(mesher_backend="SphericalMesher", cameras=cameras)
    log.write("[probe] done\n")
    log.close()


MODES = {
    "order": mode_order,
    "double": mode_double,
    "probe": mode_probe,
}

if __name__ == "__main__":
    mode, coarse_dir, device = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    MODES[mode](coarse_dir, device)

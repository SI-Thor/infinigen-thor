#!/usr/bin/env python3
"""thor/p3_mesher_only.py - phase 3 E5: is the C++ spherical mesher alone
nondeterministic, with a kernel that is a pure per-point function?

Drives CubeSphericalMesher (the background mesher of the fine stage, including
its finefront/upscale/stitching paths) and FrontviewSphericalMesher (the
in-view half) directly with a synthetic sphere SDF evaluated in numpy — zero
Infinigen pipeline, zero numeric noise: any nondeterminism here is the
mesher's own (thread scheduling), not the kernels'.

Usage: p3_mesher_only.py [n_runs]
Prints per-mesher per-run (n_verts, n_faces, sha256 of raw float64 vertex
bytes, sha256 after np.lexsort canonicalization). Exit 1 if any run differs
from the first in count OR canonical hash.
"""
import hashlib
import sys

import numpy as np

from infinigen.terrain.mesher.cube_spherical_mesher import CubeSphericalMesher
from infinigen.terrain.mesher.frontview_spherical_mesher import (
    FrontviewSphericalMesher,
)
from infinigen.terrain.utils import Vars


class SphereKernel:
    tag = 0

    def __call__(self, XYZ, sdf_only=False):
        center = np.array([5.0, -3.0, 1.0])
        r = (
            30
            + 2.0 * np.sin(0.11 * XYZ[:, 0]) * np.cos(0.13 * XYZ[:, 1])
            + 1.5 * np.sin(0.07 * XYZ[:, 2] + 0.3 * np.sin(0.05 * XYZ[:, 0]))
        )
        sdf = np.linalg.norm(XYZ - center, axis=1) - r
        return {Vars.SDF: sdf.astype(np.float32)}


def kernel_caller(kernels, XYZ):
    sdfs = [k(XYZ, sdf_only=1)[Vars.SDF] for k in kernels]
    return np.stack(sdfs, -1)


def cam_pose():
    cam = np.eye(3, 4)
    cam[:3, :3] = [
        [0.9, -0.1, 0.4],
        [0.2, 0.95, -0.2],
        [-0.35, 0.25, 0.9],
    ]
    cam[:, 3] = [3.0, -2.0, 25.0]
    return cam


def run_cube():
    m = CubeSphericalMesher(
        cam_pose(),
        r_min=1,
        r_max=800,
        base_90d_resolution=120,
        base_R=40,
        H_fov=1.0,
        W_fov=1.0,
        N0=24,
        N1=24,
        inview_upscale=4,
        test_downscale=5,
        complete_depth_test=True,
        verbose=False,
    )
    m.kernel_caller = kernel_caller
    return m([SphereKernel()])


def run_frontview():
    H = W = (120 - 24 * 2) * 2
    m = FrontviewSphericalMesher(
        cam_pose(),
        1.0,
        1.0,
        1,
        800,
        H,
        W,
        40 * 2,
        upscale=4,
        complete_depth_test=True,
        test_downscale=5,
    )
    m.kernel_caller = kernel_caller
    return m([SphereKernel()])


def describe(mesh):
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces)
    raw = hashlib.sha256(v.tobytes()).hexdigest()[:16]
    order = np.lexsort(v.T)
    canon = hashlib.sha256(v[order].tobytes()).hexdigest()[:16]
    return len(v), len(f), raw, canon


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    failed = False
    for name, fn in [("CubeSphericalMesher", run_cube), ("FrontviewSphericalMesher", run_frontview)]:
        results = []
        for i in range(n):
            results.append(describe(fn()))
        print(f"{name}:")
        for i, (nv, nf, raw, canon) in enumerate(results):
            print(f"  run{i}: verts={nv} faces={nf} raw={raw} canon={canon}")
        counts = {r[0] for r in results}
        canons = {r[3] for r in results}
        raws = {r[2] for r in results}
        if len(counts) > 1:
            print(f"  ==> VERTEX COUNTS DIFFER across runs: {sorted(counts)}")
            failed = True
        elif len(canons) > 1:
            print("  ==> counts equal but canonical (sorted) hashes differ")
            failed = True
        elif len(raws) > 1:
            print("  ==> counts and canonical hashes IDENTICAL; only raw vertex order varies")
        else:
            print("  ==> fully bit-identical across runs")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

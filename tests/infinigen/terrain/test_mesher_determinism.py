# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Casey Digennaro (thor fork)

"""Regression test: the CPU spherical meshers must be deterministic.

The frontier lists (`newfound`) and the visibility depth buckets used to be
appended under `#pragma omp critical`, so their order depended on thread
scheduling. Two runs of the same seed then produced different vertex ORDER
always, and - because the front-view visibility rasterizer is order-sensitive
at depth ties - different vertex SETS (the fine terrain wobbled by a few
hundred vertices of ~486k run to run on the Jetson Thor).

This drives CubeSphericalMesher (background mesher half of the fine stage,
exercising the coarse-frontier, finefront/upscale and stitching paths) with a
synthetic per-point sphere SDF - zero pipeline, zero numeric noise - and
asserts the raw float64 vertex bytes and face bytes are identical across
runs. Before the fix this failed with differing raw hashes; a differing
CANONICAL (sorted) hash would additionally indicate set-level breakage.
"""

import hashlib

import numpy as np
import pytest

from infinigen.terrain.mesher import cube_spherical_mesher as cube_mod
from infinigen.terrain.utils import Vars

pytest.importorskip(
    "infinigen.terrain.mesher._marching_cubes_lewiner",
    reason="terrain meshing not installed",
)


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


def _kernel_caller(kernels, XYZ):
    return np.stack([k(XYZ, sdf_only=1)[Vars.SDF] for k in kernels], -1)


def _run_mesher():
    cam = np.eye(3, 4)
    cam[:3, :3] = [[0.9, -0.1, 0.4], [0.2, 0.95, -0.2], [-0.35, 0.25, 0.9]]
    cam[:, 3] = [3.0, -2.0, 25.0]
    mesher = cube_mod.CubeSphericalMesher(
        cam,
        1,
        800,
        120,
        40,
        1.0,
        1.0,
        24,
        24,
        inview_upscale=4,
        test_downscale=5,
        complete_depth_test=True,
    )
    mesher.kernel_caller = _kernel_caller
    return mesher([SphereKernel()])


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_cube_spherical_mesher_deterministic():
    sigs = []
    for _ in range(3):
        mesh = _run_mesher()
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces)
        sigs.append(
            (
                len(verts),
                len(faces),
                hashlib.sha256(verts.tobytes()).hexdigest(),
                hashlib.sha256(faces.tobytes()).hexdigest(),
            )
        )
    assert len(set(sigs)) == 1, f"mesher output varies across runs: {sigs}"

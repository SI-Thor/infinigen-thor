# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Casey Digennaro (harvest): the EXACT-solver regression tests
# - Louise Fox (upstream): the EXACT-solver fix itself

"""Regression tests for the Boolean solver used by populate.apply_cutter.

Upstream's FAST solver (legacy BMesh CSG) can spin forever in C++ on
degenerate geometry - no exception, no log line, the worker just stops. That
is the one failure mode that can silently burn a whole night of generation,
so the solver is pinned to EXACT (folouiseAWS/infinigen d52fac4963af,
upstream PR #498) and the behaviour is pinned here:

  * apply_cutter must ask Blender for the EXACT solver,
  * the cut must still be performed, and
  * a degenerate cutter must still complete, within a bounded time.

The "bounded time" half runs in a subprocess: a regression that reintroduces
the hang would otherwise wedge the test suite instead of failing it.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import bpy
import pytest

from infinigen.core.constraints.example_solver import populate

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"

# A degenerate cutter must not be allowed to take the whole night.
BOOLEAN_TIMEOUT_SECONDS = 120

BOX_FACES = [
    (0, 1, 2),
    (0, 2, 3),
    (4, 6, 5),
    (4, 7, 6),
    (0, 4, 5),
    (0, 5, 1),
    (1, 5, 6),
    (1, 6, 2),
    (2, 6, 7),
    (2, 7, 3),
    (3, 7, 4),
    (3, 4, 0),
]


def make_box(name, size, center=(0.0, 0.0, 0.0)):
    """A plain axis-aligned box, as raw mesh data (no operators involved)."""
    hx = hy = hz = size / 2
    cx, cy, cz = center
    verts = [
        (cx - hx, cy - hy, cz - hz),
        (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz),
        (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz),
        (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz),
        (cx - hx, cy + hy, cz + hz),
    ]
    return make_mesh_object(name, verts, list(BOX_FACES))


def make_mesh_object(name, verts, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def make_degenerate_cutter(name):
    """A cutter box carrying the degenerate faces upstream reported.

    Three flavours of degenerate, all of which show up in practice when walls
    are solidified and intersect across floors:

      * a zero-area triangle (three identical vertices),
      * a sliver triangle (two vertices 1e-9 apart),
      * a duplicated vertex in an otherwise normal triangle.
    """
    h = 0.2
    # Straddle the parent's corner, so the difference really removes material
    # from the parent's surface (a cutter fully inside the parent would leave
    # the outer surface untouched and the test would prove nothing).
    cx = cy = cz = 0.45
    verts = [
        (cx - h, cy - h, cz - h),
        (cx + h, cy - h, cz - h),
        (cx + h, cy + h, cz - h),
        (cx - h, cy + h, cz - h),
        (cx - h, cy - h, cz + h),
        (cx + h, cy - h, cz + h),
        (cx + h, cy + h, cz + h),
        (cx - h, cy + h, cz + h),
    ]
    faces = list(BOX_FACES)

    # zero-area triangle
    verts += [(0.1, 0.1, 0.1)] * 3
    faces.append((8, 9, 10))
    # sliver triangle
    verts += [(0.0, 0.0, 0.0), (1e-9, 0.0, 0.0), (0.0, 1e-9, 0.0)]
    faces.append((11, 12, 13))
    # duplicated vertex
    verts += [(-0.1, -0.1, -0.1), (-0.1, -0.1, -0.1), (-0.1, -0.09, -0.1)]
    faces.append((14, 15, 16))

    return make_mesh_object(name, verts, faces)


class FakeRelation:
    def __init__(self, target_name):
        self.target_name = target_name


class FakeObjState:
    def __init__(self, obj, relations):
        self.obj = obj
        self.relations = relations


class FakeState:
    """Only the bits apply_cutter actually reads."""

    def __init__(self, objs):
        self.objs = objs


def build_scene():
    """A parent cube and a degenerate cutter that intersects it."""
    parent = make_box("parent", size=1.0)
    cutter = make_degenerate_cutter("cutter")
    state = FakeState(
        {
            "cutter": FakeObjState(cutter, [FakeRelation("parent")]),
            "parent": FakeObjState(parent, []),
        }
    )
    return state, parent, cutter


def spy_on_modify_mesh(monkeypatch):
    """Record the kwargs apply_cutter passes to butil.modify_mesh."""
    calls = []
    original = populate.butil.modify_mesh

    def spy(obj, type, **kwargs):
        calls.append({"type": type, **kwargs})
        return original(obj, type, **kwargs)

    monkeypatch.setattr(populate.butil, "modify_mesh", spy)
    return calls


def test_apply_cutter_uses_the_exact_solver(monkeypatch):
    """apply_cutter must request Blender's EXACT boolean solver."""
    state, parent, cutter = build_scene()
    calls = spy_on_modify_mesh(monkeypatch)

    populate.apply_cutter(state, "cutter", cutter)

    assert len(calls) == 1, f"expected one boolean modifier, got {len(calls)}"
    assert calls[0]["type"] == "BOOLEAN"
    assert calls[0]["operation"] == "DIFFERENCE"
    assert calls[0]["solver"] == "EXACT", (
        "apply_cutter must use the EXACT solver: the FAST solver can hang "
        "forever on degenerate geometry (folouiseAWS d52fac4963af, PR #498)"
    )


def test_apply_cutter_actually_cuts_the_parent(monkeypatch):
    """The boolean must still do its job, not just be configured correctly."""
    state, parent, cutter = build_scene()
    before = len(parent.data.vertices)

    populate.apply_cutter(state, "cutter", cutter)

    assert len(parent.modifiers) == 0, "the modifier should have been applied"
    assert len(parent.data.vertices) != before, (
        "the cutter should have changed the parent mesh"
    )


def test_degenerate_cutter_terminates():
    """The hang must stay a hang-free, bounded operation.

    Run out-of-process so that a regression which brings the infinite loop
    back fails this test instead of wedging the suite.
    """
    script = textwrap.dedent(
        """
        import sys

        sys.path.insert(0, {tests_dir!r})

        from infinigen.core.constraints.example_solver import populate
        from test_populate_cutter import build_scene

        state, parent, cutter = build_scene()

        calls = []
        original = populate.butil.modify_mesh

        def spy(obj, type, **kwargs):
            calls.append(kwargs)
            return original(obj, type, **kwargs)

        populate.butil.modify_mesh = spy
        populate.apply_cutter(state, "cutter", cutter)
        print("SOLVER=" + calls[0]["solver"])
        print("PARENT_VERTS=" + str(len(parent.data.vertices)))
        """
    ).format(tests_dir=str(Path(__file__).resolve().parent))

    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_DIR)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=BOOLEAN_TIMEOUT_SECONDS,
            env=env,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"apply_cutter did not finish within {BOOLEAN_TIMEOUT_SECONDS}s on "
            "degenerate cutter geometry - the silent-hang failure mode is back"
        )

    assert proc.returncode == 0, (
        f"apply_cutter failed on degenerate geometry:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "SOLVER=EXACT" in proc.stdout, (
        f"expected the EXACT solver, got:\n{proc.stdout}\n{proc.stderr}"
    )

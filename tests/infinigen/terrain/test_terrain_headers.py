# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Casey Digennaro (harvest): the terrain header regression tests
# - ebajemon-tw3, Diego Davidovich Gomes (upstream): the fixes themselves

"""Regression tests for the terrain C++/CUDA headers this fork harvests.

Three upstream fixes are pinned here, all of them unmerged upstream:

  * ebajemon-tw3 0eb315a0 - FastNoiseLite.h included only <cmath>, so the
    header did not stand on its own: compiling it without the includer's
    <cassert>/<cstddef> failed ('assert' was not declared in this scope).
  * DiegoGomesDG b6ee485 - the same header used NULL in what becomes device
    code; NULL is not a pointer there.
  * DiegoGomesDG 3aa7b1a - sdf_from_mesh asserted on an empty mesh, so an
    empty mesh aborted the worker instead of producing a distance.

The tests compile the headers the way the terrain build does, and run the
sdf_from_mesh kernel on a host compiler so the empty-mesh behaviour is
actually exercised.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TERRAIN_SOURCE = REPO_ROOT / "src" / "infinigen" / "terrain" / "source"

COMMON_FNL = TERRAIN_SOURCE / "common" / "utils" / "FastNoiseLite.h"
SOIL_MACHINE_FNL = (
    TERRAIN_SOURCE / "cpu" / "soil_machine" / "include" / "FastNoiseLite.h"
)
SDF_FROM_MESH = TERRAIN_SOURCE / "common" / "sdf_from_mesh" / "sdf_from_mesh.h"

# The header is used from both a host (.cpp) and a device (.cu) translation
# unit; DEVICE_FUNC / CONSTANT_ARRAY are supplied by the includer.
HOST_PROLOGUE = "#define DEVICE_FUNC\n#define CONSTANT_ARRAY const\n"
DEVICE_PROLOGUE = (
    "#define DEVICE_FUNC __device__\n#define CONSTANT_ARRAY __device__ __constant__\n"
)

gxx = shutil.which("g++")
nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"

# A kernel body that exercises sdf_from_mesh_common, plus the empty-mesh
# case that used to trip the assert.
SDF_TEST_PROGRAM = r"""
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
using namespace std;
#define DEVICE_FUNC
#define CONSTANT_ARRAY const
#include "common/utils/vectors.h"
#include "common/sdf_from_mesh/sdf_from_mesh.h"

int failures = 0;

int main() {
    // 1. An empty mesh: no faces at all. Before the fix this hit
    //    assert(min_dist != 1e9) and aborted the process.
    {
        float3_nonbuiltin verts[1];
        int faces[3] = {0, 0, 0};
        float sdf = -12345.0f;
        float3_nonbuiltin pos(0.0f, 0.0f, 0.0f);
        sdf_from_mesh_common(pos, &sdf, verts, faces, 0);
        if (sdf != 1e5f) {
            printf("FAIL empty mesh: sdf=%g, expected 1e5\n", sdf);
            failures++;
        } else {
            printf("OK empty mesh: sdf=%g\n", sdf);
        }
    }

    // 2. A mesh whose faces are all degenerate behaves like an empty one:
    //    every triangle is skipped, so the fallback must kick in.
    {
        float3_nonbuiltin verts[3] = {
            float3_nonbuiltin(0.0f, 0.0f, 0.0f),
            float3_nonbuiltin(0.0f, 0.0f, 0.0f),
            float3_nonbuiltin(0.0f, 0.0f, 0.0f),
        };
        int faces[3] = {0, 1, 2};
        float sdf = -12345.0f;
        float3_nonbuiltin pos(0.0f, 0.0f, 0.0f);
        sdf_from_mesh_common(pos, &sdf, verts, faces, 1);
        if (sdf != 1e5f) {
            printf("FAIL degenerate mesh: sdf=%g, expected 1e5\n", sdf);
            failures++;
        } else {
            printf("OK degenerate mesh: sdf=%g\n", sdf);
        }
    }

    // 3. A real triangle still produces the signed distance to its plane.
    {
        float3_nonbuiltin verts[3] = {
            float3_nonbuiltin(0.0f, 0.0f, 0.0f),
            float3_nonbuiltin(1.0f, 0.0f, 0.0f),
            float3_nonbuiltin(0.0f, 1.0f, 0.0f),
        };
        int faces[3] = {0, 1, 2};

        float above = 0.0f;
        sdf_from_mesh_common(
            float3_nonbuiltin(0.25f, 0.25f, 0.5f), &above, verts, faces, 1);
        if (fabsf(above - 0.5f) > 1e-4f) {
            printf("FAIL triangle above: sdf=%g, expected 0.5\n", above);
            failures++;
        } else {
            printf("OK triangle above: sdf=%g\n", above);
        }

        float below = 0.0f;
        sdf_from_mesh_common(
            float3_nonbuiltin(0.25f, 0.25f, -0.5f), &below, verts, faces, 1);
        if (fabsf(below + 0.5f) > 1e-4f) {
            printf("FAIL triangle below: sdf=%g, expected -0.5\n", below);
            failures++;
        } else {
            printf("OK triangle below: sdf=%g\n", below);
        }
    }

    return failures;
}
"""


def compile_and_run(tmp_path, source, include_dir, compiler, extra_flags=()):
    """Compile a snippet against the terrain headers and return the result."""
    src = tmp_path / "snippet.cpp"
    src.write_text(source)
    out = tmp_path / "snippet"
    cmd = [
        compiler,
        *extra_flags,
        "-std=c++17",
        "-I",
        str(include_dir),
        str(src),
        "-o",
        str(out),
    ]
    build = subprocess.run(cmd, capture_output=True, text=True)
    if build.returncode != 0:
        pytest.fail(
            f"compilation failed ({' '.join(cmd)}):\n{build.stdout}\n{build.stderr}"
        )
    run = subprocess.run([str(out)], capture_output=True, text=True)
    return run


@pytest.mark.skipif(gxx is None, reason="no host C++ compiler available")
@pytest.mark.parametrize(
    "header,label",
    [
        (COMMON_FNL, "terrain copy"),
        (SOIL_MACHINE_FNL, "soil_machine copy"),
    ],
)
def test_fastnoise_lite_header_stands_alone(tmp_path, header, label):
    """FastNoiseLite.h must compile without the includer supplying <cassert>.

    Before ebajemon-tw3 0eb315a0 this failed with
    "'assert' was not declared in this scope": the header only included
    <cmath> and relied on the includer having included <cassert> first.
    """
    source = HOST_PROLOGUE + f'#include "{header.relative_to(TERRAIN_SOURCE)}"\n'
    source += "int main() { return 0; }\n"
    src = tmp_path / "standalone.cpp"
    src.write_text(source)
    out = tmp_path / "standalone"
    build = subprocess.run(
        [
            gxx,
            "-std=c++17",
            "-fsyntax-only",
            "-I",
            str(TERRAIN_SOURCE),
            str(src),
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, (
        f"the {label} of FastNoiseLite.h does not stand alone:\n"
        f"{build.stdout}\n{build.stderr}"
    )


@pytest.mark.skipif(nvcc is None or not Path(nvcc).exists(), reason="no nvcc")
def test_fastnoise_lite_header_compiles_for_the_device(tmp_path):
    """The same header must also compile as device code."""
    source = (
        DEVICE_PROLOGUE
        + '#include "common/utils/FastNoiseLite.h"\n'
        + "int main() { return 0; }\n"
    )
    src = tmp_path / "standalone.cu"
    src.write_text(source)
    out = tmp_path / "standalone.o"
    build = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-c",
            "-o",
            str(out),
            "-I",
            str(TERRAIN_SOURCE),
            str(src),
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, (
        f"FastNoiseLite.h does not compile as device code:\n"
        f"{build.stdout}\n{build.stderr}"
    )


@pytest.mark.parametrize(
    "header,label",
    [
        (COMMON_FNL, "terrain copy"),
        (SOIL_MACHINE_FNL, "soil_machine copy"),
    ],
)
def test_fastnoise_lite_header_uses_no_bare_null(header, label):
    """NULL is not a pointer, and this header is compiled as device code.

    DiegoGomesDG b6ee485 replaced every NULL with nullptr in the terrain
    copy; this keeps them from coming back.
    """
    text = header.read_text()
    offenders = [
        (i + 1, line)
        for i, line in enumerate(text.splitlines())
        if re.search(r"(?<![:\w])NULL(?![:\w])", line)
    ]
    assert not offenders, (
        f"the {label} of FastNoiseLite.h uses NULL again, at: {offenders}"
    )


@pytest.mark.skipif(gxx is None, reason="no host C++ compiler available")
def test_sdf_from_mesh_handles_an_empty_mesh(tmp_path):
    """An empty (or all-degenerate) mesh must return a distance, not abort.

    Before DiegoGomesDG 3aa7b1a the kernel asserted
    `min_dist != 1e9` and took the worker down; in an NDEBUG build it
    returned a distance computed from an uninitialised sign.
    """
    run = compile_and_run(tmp_path, SDF_TEST_PROGRAM, TERRAIN_SOURCE, gxx)
    assert run.returncode == 0, f"sdf_from_mesh misbehaved:\n{run.stdout}\n{run.stderr}"
    assert "OK empty mesh" in run.stdout, run.stdout
    assert "OK degenerate mesh" in run.stdout, run.stdout
    assert "OK triangle above" in run.stdout, run.stdout
    assert "OK triangle below" in run.stdout, run.stdout

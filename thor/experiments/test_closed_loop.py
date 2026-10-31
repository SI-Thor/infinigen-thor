#!/usr/bin/env python3
"""
test_closed_loop.py - end-to-end proof that the harness works on a REAL
Infinigen output directory.

The harness was written against an ASSUMED layout ("rgb_*.png",
"*obj_index*.npy", at the top level).  The real layout, from
src/infinigen2/exporters/util/format.py, is:

    <output>/<camera>/0001.png                  the main image   ("%c/%f.png")
    <output>/<camera>/depth_0001.npy            ("%c/depth_%f.npy")
    <output>/<camera>/optical-flow_0001.npy     ("%c/optical-flow_%f.npy")
    <output>/<camera>/object_0001.npy           ("%c/object_%f.npy")

with %c = the camera name (a subdirectory) and %f = f"{frame:04d}".

A loader built on the assumption finds NOTHING, and reports "no frames found"
- which reads like a data problem and is actually a loader problem.  This file
builds a directory with the real layout and asserts the loader finds it.

No pytest fixtures are used, so this runs under pytest AND under plain
`python3 thor/experiments/test_closed_loop.py`.
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from closed_loop import (IdentityGenerator, IdentityPerception, NoisyGenerator,
                         load_frames, run_roundtrip, summarize)

H, W = 8, 6
N_FRAMES = 4


def build_infinigen_output(root, camera="cam_001", frames=N_FRAMES):
    """A directory laid out exactly the way Infinigen writes one."""
    d = Path(root) / camera
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for f in range(1, frames + 1):
        num = f"{f:04d}"
        depth = (rng.random((H, W)) * 5 + 1).astype(np.float32)
        flow = rng.normal(0, 0.5, (H, W, 2)).astype(np.float32)
        obj = rng.integers(0, 7, (H, W)).astype(np.uint32)
        rgb = rng.integers(0, 255, (H, W, 3)).astype(np.uint8)
        np.save(d / f"depth_{num}.npy", depth)
        np.save(d / f"optical-flow_{num}.npy", flow)
        np.save(d / f"object_{num}.npy", obj)
        np.save(d / f"surface-normal_{num}.npy",
                rng.normal(0, 1, (H, W, 3)).astype(np.float32))
        # the main image: "%c/%f.png" -> cam_001/0001.png
        try:
            from PIL import Image
            Image.fromarray(rgb).save(d / f"{num}.png")
        except Exception:
            pass
    return str(root)


def test_loader_finds_the_real_layout():
    with tempfile.TemporaryDirectory() as tmp:
        build_infinigen_output(tmp)
        frames = load_frames(tmp)
        assert len(frames) == N_FRAMES, f"found {len(frames)} frames, want {N_FRAMES}"
        f0 = frames[0]
        for modality in ("depth", "flow", "obj_index", "normal"):
            assert modality in f0, f"{modality} missing; got {sorted(f0)}"
        assert f0["depth"].shape == (H, W), f0["depth"].shape
        assert f0["flow"].shape == (H, W, 2), f0["flow"].shape
        # object index is a LABEL MAP, not one-hot
        assert f0["obj_index"].shape == (H, W), f0["obj_index"].shape
        assert f0["obj_index"].dtype == np.uint32, f0["obj_index"].dtype


def test_loader_ignores_a_non_infinigen_directory():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "notes"
        d.mkdir()
        (d / "readme.txt").write_text("not a render")
        (d / "rgb_0001.png").write_bytes(b"not really a png")
        assert load_frames(tmp) == [], "the loader invented frames"


def test_roundtrip_is_zero_for_identity_and_positive_for_noise():
    """The harness must be able to TELL a good generator from a bad one.
    Identity -> residual exactly 0.  Identity+eps -> residual > 0."""
    with tempfile.TemporaryDirectory() as tmp:
        build_infinigen_output(tmp)
        frames = load_frames(tmp)
        # A vacuous pass is a lie: with zero frames the loop below asserts
        # nothing.  The frames must be there for the control to mean anything.
        assert len(frames) == N_FRAMES, f"loaded {len(frames)} frames"
        P = IdentityPerception()

        per = run_roundtrip(frames, P, IdentityGenerator())
        agg, _ = summarize(per)
        for k, v in agg.items():
            assert abs(v) < 1e-12, f"{k}: identity gave {v}, want 0"

        per = run_roundtrip(frames, P, NoisyGenerator())
        agg2, _ = summarize(per)
        for k in agg:
            assert agg2[k] > agg[k], f"{k}: noise did not increase the residual"


def test_object_index_metric_handles_label_maps():
    """object index is a label map (H,W) uint32.  argmax(-1) on it is
    meaningless; comparing the labels is not."""
    from closed_loop import metric
    a = np.array([[1, 2], [3, 4]], dtype=np.uint32)
    b = np.array([[1, 2], [3, 9]], dtype=np.uint32)
    assert abs(metric("obj_index", a, b) - 0.25) < 1e-9, metric("obj_index", a, b)
    assert metric("obj_index", a, a) == 0.0


def test_the_cli_produces_a_receipt_and_a_csv():
    """End to end through the command line, with fake backends."""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "render")
        out = os.path.join(tmp, "run")
        build_infinigen_output(src)
        here = os.path.dirname(os.path.abspath(__file__))
        r = subprocess.run(
            [sys.executable, os.path.join(here, "closed_loop.py"),
             "--source", src, "--out", out, "--controls-only"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, f"exit {r.returncode}\n{r.stdout}\n{r.stderr}"
        receipt = os.path.join(out, "closed-loop-receipt.json")
        assert os.path.exists(receipt), f"no receipt; stdout was:\n{r.stdout}"
        with open(receipt) as f:
            data = json.load(f)
        assert data["controls"]["positive"]["must_be_zero"] is True
        assert data["controls"]["negative"]["must_exceed"] is True
        assert "n_frames" in data


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'ALL PASS' if not failures else str(failures) + ' FAILED'}")
    sys.exit(1 if failures else 0)

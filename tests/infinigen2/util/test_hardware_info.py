# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""GPU detection must not lose a GPU to an unsupported sibling field.

The defect this guards against: `_detect_gpus()` used to ask for the GPU name
and its memory in ONE `nvidia-smi --query-gpu=name,memory.total` call.  On a
unified-memory device (Jetson AGX Thor, Jetson AGX Orin, GB10) there is no
dedicated VRAM, so `memory.total` comes back as `[N/A]`.  Where the driver
rejects the field outright the whole query exits non-zero, and the combined
call throws the GPU's NAME away with it - the box then reports no GPU at all,
and the run's GPU accounting is silently lost.

The name always works.  The memory is the field that does not.  So the two are
asked for separately, and an N/A costs the memory figure and nothing else.

These tests use a FAKE `nvidia-smi` on PATH.  They need no GPU, no Jetson and
no driver, which is the point: the behaviour under test is about which query
is issued and how its output is parsed, and that must be checkable anywhere.

No pytest fixtures are used, so this file runs under pytest AND under a plain
`python3 tests/infinigen2/util/test_hardware_info.py`.
"""

import os
import stat
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

try:  # the normal path, when the package's dependencies are installed
    from infinigen2.util.hardware_info import _detect_gpus, get_hardware_info  # noqa: E402
except Exception:  # pragma: no cover - fallback for a bare checkout
    # infinigen2/__init__.py imports bpy, which is not present everywhere
    # (this test deliberately has no GPU/bpy requirement).  Load the module
    # by path instead of dragging the whole package in.
    import importlib.util

    _HI = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "infinigen2"
        / "util"
        / "hardware_info.py"
    )
    _spec = importlib.util.spec_from_file_location("infinigen2_util_hw_info", _HI)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _detect_gpus = _mod._detect_gpus
    get_hardware_info = _mod.get_hardware_info

# A fake nvidia-smi that behaves like the real one on each class of machine.
# It dispatches on its own argv, because the whole point of the fix is that
# the name query and the memory query are SEPARATE calls with different
# answers.
FAKE_SMI = r'''#!/usr/bin/env python3
"""A fake `nvidia-smi` that answers the way the real one does.

It parses the `--query-gpu=<fields>` argument and prints ONE CSV line per
device with exactly the requested fields, in order - which is what the real
tool does, and which is what makes a COMBINED query return "name, memory"
rather than just the memory.  A fake that answered combined queries with only
one field would make every test here meaningless.
"""
import os
import sys

fields = []
for arg in sys.argv[1:]:
    if arg.startswith("--query-gpu="):
        fields = arg.split("=", 1)[1].split(",")

mode = os.environ.get("FAKE_SMI_MODE", "dgpu")
names = {
    "dgpu": "NVIDIA A100-SXM4-80GB",
    "thor": "NVIDIA Thor",
    "orin": "Orin (nvgpu)",
}
memories = {
    "dgpu": "81920 MiB",
    # Unified memory: there is no dedicated VRAM, so the field is [N/A].
    "thor": "[N/A]",
    "orin": "[N/A]",
}

if mode == "memory_field_rejected":
    if "memory.total" in fields:
        sys.stderr.write(
            "Field 'memory.total' is not a valid field to query.\n"
        )
        sys.exit(1)
    sys.stdout.write(names["thor"] + "\n")
    sys.exit(0)

if mode == "broken":
    sys.stderr.write("nvidia-smi: driver/library version mismatch\n")
    sys.exit(9)

if mode not in names:
    sys.exit(0)

values = []
for field in fields:
    if field == "name":
        values.append(names[mode])
    elif field == "memory.total":
        values.append(memories[mode])
    else:
        values.append("[N/A]")
sys.stdout.write(", ".join(values) + "\n")
sys.exit(0)
'''


class _fake_smi:
    """Context manager: put a fake nvidia-smi on PATH (or take it away)."""

    def __init__(self, mode):
        self.mode = mode
        self._saved = None

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._bin = Path(self._tmp.name) / "bin"
        self._bin.mkdir(parents=True, exist_ok=True)
        if self.mode != "absent":
            exe = self._bin / "nvidia-smi"
            exe.write_text(FAKE_SMI)
            exe.chmod(exe.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        self._saved = os.environ.get("PATH", "")
        # An empty bin dir first, so the REAL nvidia-smi cannot be found.
        os.environ["PATH"] = f"{self._bin}{os.pathsep}{self._saved}"
        os.environ["FAKE_SMI_MODE"] = self.mode
        return self

    def __exit__(self, *exc):
        if self._saved is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = self._saved
        os.environ.pop("FAKE_SMI_MODE", None)
        self._tmp.cleanup()
        return False


def _detect(mode):
    with _fake_smi(mode):
        return _detect_gpus()


def test_dedicated_gpu_format_is_unchanged():
    """Regression guard: a GPU with real VRAM must still be reported in
    exactly the format the rest of the tree already parses -
    "NAME, MEMORY"."""
    gpus = _detect("dgpu")
    assert gpus == ["NVIDIA A100-SXM4-80GB, 81920 MiB"], gpus


def test_thor_is_detected_without_a_memory_figure():
    """Jetson AGX Thor: unified memory, so memory.total is [N/A].  The GPU
    must still be reported, and the entry must not carry a literal "[N/A]"
    that makes it look like a parsing failure."""
    gpus = _detect("thor")
    assert len(gpus) == 1, gpus
    assert "Thor" in gpus[0], gpus
    assert "[N/A]" not in gpus[0], gpus


def test_orin_is_detected_without_a_memory_figure():
    gpus = _detect("orin")
    assert len(gpus) == 1, gpus
    assert "Orin" in gpus[0], gpus
    assert "[N/A]" not in gpus[0], gpus


def test_gpu_survives_a_rejected_memory_field():
    """THE DEFECT.  If the driver refuses `memory.total`, the GPU must still
    be detected.  Before the fix this returned [] - no GPU at all."""
    gpus = _detect("memory_field_rejected")
    assert len(gpus) == 1, f"the GPU was lost: {gpus}"
    assert "Thor" in gpus[0], gpus


def test_missing_nvidia_smi_is_not_an_error():
    """No nvidia-smi means no GPU.  That is a fact, not a crash."""
    gpus = _detect("absent")
    assert gpus == [], gpus


def test_a_failing_nvidia_smi_is_not_an_error():
    gpus = _detect("broken")
    assert gpus == [], gpus


def test_get_hardware_info_never_raises():
    """get_hardware_info() is called during every generate() run and its
    result is written to metadata.json.  It must never take a run down."""
    for mode in ("dgpu", "thor", "orin", "memory_field_rejected", "broken", "absent"):
        with _fake_smi(mode):
            info = get_hardware_info()
        assert "cpu" in info, info
        assert "hostname" in info, info


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
    print(f"\n{'ALL PASS' if not failures else str(failures) + ' FAILED'}")
    sys.exit(1 if failures else 0)

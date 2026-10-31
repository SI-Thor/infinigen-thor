# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import getpass
import os
import platform
import socket
import subprocess


def _run(cmd: list[str], timeout: int = 5):
    """subprocess.run that returns None instead of raising on a missing or
    hung binary.  Every caller below treats "the tool is not there" and "the
    tool said no" the same way, so they should not have to care which."""
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _nvidia_gpus() -> list[str]:
    """Detect NVIDIA GPUs.

    The NAME and the MEMORY are fetched with SEPARATE queries, on purpose.  A
    single combined query loses the GPU entirely if any one of its fields is
    unsupported, and on unified-memory devices (Jetson AGX Thor, Jetson AGX
    Orin, GB10) `memory.total` is exactly that field: there is no dedicated
    VRAM to report, so it comes back as [N/A].  The name always works; the
    memory is the field that does not.

    Asking separately means an N/A - or a driver that rejects the field
    outright - costs the memory figure and nothing else, instead of taking the
    GPU's name down with it.

    Output format is unchanged for GPUs that do have dedicated VRAM:
    "NVIDIA A100-SXM4-80GB, 81920 MiB".  For a unified-memory device it is
    just the name, because gluing "[N/A]" onto it makes the entry look like a
    parsing failure rather than what it is - a device with no dedicated VRAM.
    """
    names_result = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if names_result is None or names_result.returncode != 0:
        return []
    names = [
        line.strip()
        for line in names_result.stdout.strip().splitlines()
        if line.strip()
    ]
    if not names:
        return []

    memories = [""] * len(names)
    mem_result = _run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"])
    if mem_result is not None and mem_result.returncode == 0:
        values = [
            line.strip()
            for line in mem_result.stdout.strip().splitlines()
            if line.strip()
        ]
        for i, value in enumerate(values[: len(names)]):
            # [N/A] / N/A / Not Supported all mean "no dedicated VRAM to
            # report" (unified memory).  Recording the literal string is worse
            # than recording nothing: it makes the entry look broken.
            lowered = value.lower()
            if lowered in ("n/a", "not supported", "[n/a]") or value.startswith("["):
                continue
            memories[i] = value

    return [
        f"{name}, {memories[i]}" if memories[i] else name
        for i, name in enumerate(names)
    ]


def _detect_gpus() -> list[str]:
    gpus = []

    gpus += _nvidia_gpus()

    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--csv"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines()[1:]:
                if line.strip():
                    gpus.append(line.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if line.startswith("Chipset Model:"):
                        gpus.append(line.split(":", 1)[1].strip())
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return gpus


def get_hardware_info() -> dict:
    info = {}

    info["username"] = getpass.getuser()
    info["hostname"] = socket.gethostname()
    info["platform"] = platform.platform()
    info["cpu"] = platform.processor() or platform.machine()

    gpus = _detect_gpus()
    if gpus:
        info["gpus_all"] = gpus

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        info["CUDA_VISIBLE_DEVICES"] = cuda_visible

    rocr_visible = os.environ.get("ROCR_VISIBLE_DEVICES")
    if rocr_visible is not None:
        info["ROCR_VISIBLE_DEVICES"] = rocr_visible

    slurm_vars = {k: v for k, v in os.environ.items() if k.startswith("SLURM_")}
    if slurm_vars:
        info["slurm"] = slurm_vars

    return info

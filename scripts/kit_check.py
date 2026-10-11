"""GPU-kit setup check (run by 1_SETUP.bat): is the GPU usable from this environment?

Writes results/kit_setup.json and exits non-zero with a plain-language message if not.
"""

from __future__ import annotations

import platform
import sys

import torch

from pharmachecked.config import REPO_ROOT
from pharmachecked.utils.env import capture_env
from pharmachecked.utils.io import read_json, write_json


def main() -> int:
    """Check CUDA and record the environment; returns a process exit code."""
    env = capture_env(REPO_ROOT)
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
        if torch.cuda.is_available()
        else None,
        "commit": env["git_commit"],
        "snapshot_modified": env["git_dirty"],
    }
    write_json(REPO_ROOT / "results" / "kit_setup.json", info)
    for key, value in info.items():
        sys.stdout.write(f"  {key}: {value}\n")
    kit_file = REPO_ROOT / "KIT.json"
    allow_cpu = kit_file.exists() and bool(read_json(kit_file).get("allow_cpu"))
    if not info["cuda_available"] and allow_cpu:
        sys.stdout.write("\nSelf-test kit: no GPU needed. Check passed.\n")
        return 0
    if not info["cuda_available"]:
        sys.stdout.write(
            "\nThe graphics card is not usable yet. Install the newest NVIDIA driver for the "
            "RTX 5060 (nvidia.com > Drivers, or the NVIDIA app), restart, and double-click "
            "1_SETUP.bat again.\n"
        )
        return 1
    if info["snapshot_modified"]:
        sys.stdout.write("\nFiles in the kit were changed. Unzip a fresh copy of the kit.\n")
        return 1
    sys.stdout.write("\nGPU check passed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

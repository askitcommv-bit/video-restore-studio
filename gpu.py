"""
NVIDIA GPU queries through nvidia-smi, which sees every process on the card (the driver's view).

relayvsr_worker.py has its own copy of free_gb(): it runs in RelayVSR's separate Python
environment and can't import the app's modules.
"""

import subprocess
from typing import Optional

import processes


def _query(fields: str) -> Optional[str]:
    """First line of an nvidia-smi --query-gpu answer, or None without a usable NVIDIA driver."""
    try:
        out = processes.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = out.stdout.strip().splitlines() if out.returncode == 0 else []
    return lines[0].strip() if lines else None


def name() -> Optional[str]:
    """Name of the first NVIDIA GPU, or None."""
    return _query("name") or None


def free_gb() -> Optional[float]:
    """Free VRAM on the first GPU across all processes, in GB, or None if it can't be read."""
    line = _query("memory.used,memory.total")
    try:
        used, total = (float(v) for v in line.split(","))
    except (AttributeError, ValueError):
        return None
    return (total - used) / 1024

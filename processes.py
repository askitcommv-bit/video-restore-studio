"""
Starting helper programs (ffmpeg, ffprobe, nvidia-smi, uv) from the app.

A GUI process has no console, so Windows would open a black console window for every console
program it starts. run() and popen() are subprocess.run / subprocess.Popen with that window
suppressed; everything else is passed through unchanged. Use plain subprocess only to open a
console on purpose (CREATE_NEW_CONSOLE, e.g. launching ComfyUI).
"""

import subprocess
import sys

NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def _flags(kwargs: dict) -> dict:
    kwargs["creationflags"] = kwargs.get("creationflags", 0) | NO_WINDOW
    return kwargs


def run(cmd, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, **_flags(kwargs))


def popen(cmd, **kwargs) -> subprocess.Popen:
    return subprocess.Popen(cmd, **_flags(kwargs))

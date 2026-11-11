"""
Process start-up shared by every entry point (desktop app, CLI, browser UI). Call init() once,
before doing any work: it is where the app touches its environment, so importing modules never does.
"""

import os
import sys

import config


def init() -> None:
    # Windows consoles default to a legacy code page; progress text uses ·, →, ✓
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass

    config.PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    # Tools the setup downloaded come first on PATH, for this process and everything it starts,
    # so "ffmpeg"/"ffprobe"/"uv" resolve without a system-wide install
    for tool_dir in (config.FFMPEG_BIN_DIR, config.UV_EXE.parent):
        if tool_dir.is_dir() and str(tool_dir) not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = str(tool_dir) + os.pathsep + os.environ.get("PATH", "")

    import setup_manager
    setup_manager.sync_worker()

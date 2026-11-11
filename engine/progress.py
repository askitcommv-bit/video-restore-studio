"""Progress reporting shared by the engines."""

from typing import Callable, Optional

# progress(current, total, text): current/total is a percentage of the whole job (total = 100)
ProgressCallback = Optional[Callable[[int, int, str], None]]


def report(progress: ProgressCallback, current: int, total: int, text: str) -> None:
    if progress:
        progress(current, total, text)


def format_eta(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes // 60}h {minutes % 60:02d}m" if minutes >= 60 else f"{max(minutes, 1)}m"

"""
Resumable file downloads with progress and cancellation; used for every download the app makes
(FFmpeg, uv, the RelayVSR code and weights, the ComfyUI engines' models).

The file is streamed to <dest>.part and renamed when complete, so an interrupted download resumes
with an HTTP Range request instead of starting over.
"""

import threading
import time
from pathlib import Path
from typing import Callable, Optional

import requests

from errors import DownloadForbidden

Progress = Callable[[str, int, int], None]  # text, done, total (total 0 = busy / unknown)


class Cancelled(Exception):
    """Raised inside a download (or any long step) when its cancel event is set."""


def check_cancel(cancel: Optional[threading.Event]) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def _no_progress(text: str, done: int, total: int) -> None:
    pass


def download(url: str, dest: Path, label: str = "", progress: Progress = _no_progress,
             cancel: Optional[threading.Event] = None, expected_size: int = 0,
             token: Optional[str] = None, repo_url: Optional[str] = None) -> Path:
    """Downloads url to dest unless dest is already complete (expected_size, or any size if 0).

    token is sent as a Bearer token (Hugging Face). A 401/403 raises DownloadForbidden, naming
    repo_url (where to accept a license) when given."""
    if dest.is_file() and (dest.stat().st_size == expected_size if expected_size else dest.stat().st_size > 0):
        return dest
    label = label or f"Downloading {dest.name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    legacy_part = dest.with_name(dest.name + ".downloading")  # partial files of the old downloader
    if legacy_part.exists() and not part.exists():
        legacy_part.replace(part)

    base_headers = {"User-Agent": "VideoRestoreStudio"}
    if token:
        base_headers["Authorization"] = f"Bearer {token}"
    for attempt in range(5):
        have = part.stat().st_size if part.exists() else 0
        headers = dict(base_headers, **({"Range": f"bytes={have}-"} if have else {}))
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 60)) as r:
                if r.status_code in (401, 403):
                    raise DownloadForbidden(url, repo_url, (r.headers.get("X-Error-Message") or "").strip())
                if r.status_code == 416:  # nothing left to fetch
                    break
                r.raise_for_status()
                if have and r.status_code != 206:  # server ignored the range: start over
                    have = 0
                length = r.headers.get("Content-Length")
                total = have + int(length) if length else expected_size
                t0, got0, last = time.monotonic(), have, 0.0
                with open(part, "ab" if have else "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        check_cancel(cancel)
                        f.write(chunk)
                        have += len(chunk)
                        now = time.monotonic()
                        if now - last > 0.25:
                            last = now
                            speed = (have - got0) / max(now - t0, 1e-3) / 1e6
                            progress(f"{label} · {speed:.0f} MB/s", have, total)
            break
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
            if attempt == 4:
                raise
            progress(f"{label} · connection lost, retrying…", 0, 0)
            time.sleep(3 * (attempt + 1))
    if expected_size and part.stat().st_size != expected_size:
        raise IOError(f"{dest.name}: got {part.stat().st_size} bytes, expected {expected_size}")
    part.replace(dest)
    return dest

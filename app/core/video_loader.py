"""
Opening a video for the app: its details (ffprobe) and the defect scan, read off the UI thread.

One VideoLoader serves every caller (opening a file, previewing a queued job, re-running the scan).
Results are tagged with the path; anything from a load that a newer one replaced is dropped.
"""

from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal

import setup_manager
import video_preprocessor
from errors import FFmpegMissing


class _LoadWorker(QThread):
    step = Signal(str, int, int)
    info_ready = Signal(dict)
    scanned = Signal(dict)
    failed = Signal(object)

    def __init__(self, path: str, read_info: bool):
        super().__init__()
        self.path = path
        self.read_info = read_info

    def run(self):
        if not setup_manager.ffmpeg_ready():
            self.failed.emit(FFmpegMissing())
            return
        if self.read_info:
            try:
                self.step.emit("Reading video details", 0, 0)
                self.info_ready.emit(video_preprocessor.get_video_info(self.path))
            except Exception as e:
                self.failed.emit(e)
                return
        try:
            res = video_preprocessor.analyze_video_defects(
                self.path, progress=lambda cur, tot, text: self.step.emit(text, cur, tot))
        except Exception as e:  # the scan only suggests settings: report it, don't fail the load
            res = {"error": str(e)}
        self.scanned.emit(res)


class VideoLoader(QObject):
    step = Signal(str, int, int)  # text, current, total (total 0 = busy)
    info_ready = Signal(str, dict)  # path, ffprobe details
    scanned = Signal(str, dict)  # path, defect scan ({"error": ...} if it couldn't scan)
    failed = Signal(str, object)  # path, exception (errors.FFmpegMissing when FFmpeg isn't installed)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current: Optional[str] = None
        self._threads: set = set()  # a QThread must outlive its run(); dropped once it has finished

    def load(self, path: str) -> None:
        """Reads the details, then scans."""
        self._start(path, read_info=True)

    def scan(self, path: str) -> None:
        """Scans again (the details are already known)."""
        self._start(path, read_info=False)

    def _start(self, path: str, read_info: bool) -> None:
        self.current = path
        worker = _LoadWorker(path, read_info)
        is_current = lambda: self.current == path
        worker.step.connect(lambda text, cur, tot: is_current() and self.step.emit(text, cur, tot))
        worker.info_ready.connect(lambda info: is_current() and self.info_ready.emit(path, info))
        worker.scanned.connect(lambda res: is_current() and self.scanned.emit(path, res))
        worker.failed.connect(lambda exc: is_current() and self.failed.emit(path, exc))
        worker.finished.connect(lambda w=worker: self._threads.discard(w))
        self._threads.add(worker)
        worker.start()

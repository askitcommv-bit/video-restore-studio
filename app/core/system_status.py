"""
Polls ComfyUI's availability and the GPU's free memory for the header, off the UI thread
(the HTTP probe takes ~250 ms online and ~2 s offline).
"""

import threading
from typing import Optional

from PySide6.QtCore import QObject, QTimer, Signal

import gpu
from engine import ComfyUIClient


class SystemStatusMonitor(QObject):
    # ComfyUI /system_stats (None = offline), free VRAM in GB (None = unknown)
    status = Signal(object, object)

    def __init__(self, client: ComfyUIClient, interval_ms: int = 4000, parent=None):
        super().__init__(parent)
        self.client = client
        self.comfy_online: Optional[bool] = None
        self._inflight = False
        self.status.connect(self._remember)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.poll)
        self._timer.start(interval_ms)

    def poll(self) -> None:
        if self._inflight:
            return
        self._inflight = True
        threading.Thread(target=lambda: self.status.emit(self.client.try_system_stats(), gpu.free_gb()),
                         daemon=True).start()

    def _remember(self, stats, _free_gb) -> None:
        self._inflight = False
        self.comfy_online = stats is not None

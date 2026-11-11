"""The window's header: logo and name, setup reminder, ComfyUI / GPU status, Outputs, ☰ menu."""

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMenu, QPushButton, QToolButton, QWidget

from app.ui import styles


class HeaderBar(QWidget):
    setup_clicked = Signal()
    outputs_clicked = Signal()

    def __init__(self, menu: QMenu, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 0, 0, 0)
        lay.setSpacing(12)

        logo = QLabel()
        dpr = self.devicePixelRatioF()
        pm = QPixmap(str(styles._ASSETS_DIR / "logo.png"))
        pm = pm.scaled(round(22 * dpr), round(22 * dpr), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pm.setDevicePixelRatio(dpr)
        logo.setPixmap(pm)
        lay.addWidget(logo)
        lay.addSpacing(-4)
        title = QLabel("Video Restore")
        title.setObjectName("title")
        lay.addWidget(title)
        lay.addStretch()

        self.btn_setup = QPushButton("⚠ Finish setup")
        self.btn_setup.setToolTip("Download what's missing: FFmpeg, the RelayVSR engine or its models")
        self.btn_setup.clicked.connect(self.setup_clicked)
        lay.addWidget(self.btn_setup)

        self.lbl_comfy = QLabel("ComfyUI …")
        self.lbl_comfy.setObjectName("faint")
        self.lbl_comfy.setToolTip("Only needed for the LTX, SoL-Refiner and Real-ESRGAN engines; "
                                 "RelayVSR runs without it")
        lay.addWidget(self.lbl_comfy)
        self.lbl_vram = QLabel("")
        self.lbl_vram.setObjectName("faint")
        self.lbl_vram.setToolTip("Free GPU memory (all apps). RelayVSR keeps part of its model in system RAM "
                                 "when this is short of ~14 GB.")
        lay.addWidget(self.lbl_vram)

        btn_outputs = QPushButton("Outputs")
        btn_outputs.setToolTip("Open the output folder")
        btn_outputs.clicked.connect(self.outputs_clicked)
        lay.addWidget(btn_outputs)

        btn_more = QToolButton()
        btn_more.setText("☰")
        btn_more.setObjectName("icon")
        btn_more.setToolTip("More")
        btn_more.setPopupMode(QToolButton.InstantPopup)
        btn_more.setMenu(menu)
        lay.addWidget(btn_more)

        self._comfy_state: Optional[tuple] = None

    def set_setup_needed(self, needed: bool) -> None:
        self.btn_setup.setVisible(needed)

    def set_status(self, comfy_online: bool, free_gb: Optional[float], comfy_needed: bool,
                   comfy_installed: bool) -> None:
        """ComfyUI runs the optional LTX / SoL / Real-ESRGAN engines only: its state is shown when it is
        installed (or running), and is a warning only while the selected engine needs it."""
        self.lbl_comfy.setVisible(comfy_installed or comfy_online)
        state = (comfy_online, comfy_needed)
        if state != self._comfy_state:  # restyle only on changes
            self._comfy_state = state
            self.lbl_comfy.setText("● ComfyUI" if comfy_online else "○ ComfyUI offline")
            styles.set_role(self.lbl_comfy, "status-warn" if comfy_needed and not comfy_online else "faint")
        # nvidia-smi sees every process; ComfyUI's own figure is per-process on Windows and misleading
        self.lbl_vram.setText(f"GPU {free_gb:.1f} GB free" if free_gb is not None else "")

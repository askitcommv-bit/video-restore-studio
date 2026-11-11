"""
Setup dialog: what each engine needs, downloaded with progress, cancellable and resumable
(setup_manager does the work). Required parts (RelayVSR) are always installed; the ComfyUI engines'
model packs are optional rows with a checkbox, shown when ComfyUI is installed.
"""

import threading
from typing import List

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QDialog, QFrame, QGridLayout, QHBoxLayout, QInputDialog, QLabel,
                               QProgressBar, QPushButton, QVBoxLayout)

import comfy_models
import config
import gpu
import setup_manager
from app.ui import styles
from app.ui.widgets import ElidedLabel
from errors import DownloadForbidden, friendly_message


class SetupWorker(QThread):
    progress = Signal(str, object, object)  # text, done, total (objects: byte counts exceed 32-bit int)
    done = Signal(object)  # None on success, "cancelled", or the exception

    def __init__(self, keys: List[str]):
        super().__init__()
        self.keys = keys
        self.cancel = threading.Event()

    def run(self):
        try:
            setup_manager.install(lambda text, d, t: self.progress.emit(text, d, t), self.cancel, self.keys)
            self.done.emit(None)
        except setup_manager.Cancelled:
            self.done.emit("cancelled")
        except Exception as e:  # network, disk, uv, gated repo: shown in the dialog
            self.done.emit(e)


def _size(n: float) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


class SetupDialog(QDialog):
    """Modal setup window. Emits `changed` after an install attempt finished."""
    changed = Signal()

    def __init__(self, parent=None, preselect: tuple = ()):
        super().__init__(parent)
        self.setWindowTitle("Set up Video Restore Studio")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.worker = None
        self._gpu = gpu.name()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(10)

        title = QLabel("Set up Video Restore Studio")
        title.setObjectName("heading")
        self.lbl_intro = QLabel("")
        self.lbl_intro.setObjectName("muted")
        self.lbl_intro.setWordWrap(True)
        lay.addWidget(title)
        lay.addWidget(self.lbl_intro)

        box = QFrame()
        box.setObjectName("panel")
        grid = QGridLayout(box)
        grid.setContentsMargins(14, 12, 14, 12)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)
        self.rows = {}
        row = 0
        comps = setup_manager.components()
        for optional in (False, True):
            group = [c for c in comps if c.optional == optional]
            if not group:
                continue
            heading = QLabel("NEEDED FOR RELAYVSR UPSCALING" if not optional else "OPTIONAL — COMFYUI ENGINES")
            heading.setObjectName("section")
            grid.addWidget(heading, row, 0, 1, 3)
            row += 1
            for c in group:
                name = QCheckBox(c.name) if optional else QLabel(c.name)
                name.setStyleSheet("font-weight: 600;")
                if optional:
                    name.setChecked(c.key in preselect)
                    name.toggled.connect(self.refresh)
                detail = QLabel(c.detail)
                detail.setObjectName("faint")
                col = QVBoxLayout()
                col.setSpacing(0)
                col.addWidget(name)
                col.addWidget(detail)
                size = QLabel(_size(c.download_gb * 1e9))
                size.setObjectName("muted")
                status = QLabel("")
                status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                status.setMinimumWidth(130)
                grid.addLayout(col, row, 0)
                grid.addWidget(size, row, 1, Qt.AlignRight)
                grid.addWidget(status, row, 2)
                self.rows[c.key] = (name if optional else None, status)
                row += 1
        grid.setColumnStretch(0, 1)
        lay.addWidget(box)

        self.lbl_gpu = QLabel("")
        self.lbl_disk = QLabel("")
        for lbl in (self.lbl_gpu, self.lbl_disk):
            lbl.setWordWrap(True)
            lay.addWidget(lbl)

        self.lbl_step = ElidedLabel("")
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(4)
        self.lbl_count = QLabel("")
        self.lbl_count.setObjectName("faint")
        self.lbl_error = QLabel("")
        self.lbl_error.setWordWrap(True)
        self.lbl_error.setTextInteractionFlags(Qt.TextSelectableByMouse)
        for w in (self.lbl_step, self.bar, self.lbl_count, self.lbl_error):
            lay.addWidget(w)

        buttons = QHBoxLayout()
        self.btn_token = QPushButton("Hugging Face token…")
        self.btn_token.setObjectName("ghost")
        self.btn_token.setToolTip("Needed for the gated LTX-2.5 models: accept the license at "
                                  "huggingface.co/Lightricks/LTX-2.5, then create a Read token")
        self.btn_token.clicked.connect(self._edit_token)
        self.btn_token.setVisible(any(c.optional for c in comps))
        buttons.addWidget(self.btn_token)
        buttons.addStretch()
        self.btn_secondary = QPushButton("Not now")
        self.btn_primary = QPushButton("Install")
        self.btn_primary.setObjectName("primary")
        self.btn_secondary.clicked.connect(self._on_secondary)
        self.btn_primary.clicked.connect(self._on_primary)
        buttons.addWidget(self.btn_secondary)
        buttons.addWidget(self.btn_primary)
        lay.addSpacing(4)
        lay.addLayout(buttons)

        self._show_progress(False)
        self.refresh()

    # -- state --------------------------------------------------------------
    def _selected(self) -> List[str]:
        """Keys to install: missing required parts, plus the ticked optional ones."""
        keys = []
        for c in setup_manager.components():
            box, _ = self.rows[c.key]
            if c.ready or (c.needs_gpu and not self._gpu):
                continue
            if not c.optional or (box is not None and box.isChecked()):
                keys.append(c.key)
        return keys

    def refresh(self, *_):
        comps = {c.key: c for c in setup_manager.components()}
        for key, c in comps.items():
            box, status = self.rows[key]
            if c.ready:
                status.setText("✓ Installed")
                styles.set_role(status, "status-ok")
            elif c.needs_gpu and not self._gpu:
                status.setText("Needs an NVIDIA GPU")
                styles.set_role(status, "status-warn")
            else:
                status.setText("To download")
                styles.set_role(status, "muted")
            if box is not None:
                box.setEnabled(not c.ready)

        selected = [comps[k] for k in self._selected()]
        need_disk = sum(c.disk_gb for c in selected)
        free = setup_manager.free_gb()
        if selected:
            self.lbl_intro.setText(
                f"A one-time download of about {sum(c.download_gb for c in selected):.1f} GB. "
                f"Everything goes into {config.PROJECT_DIR}.")
        elif setup_manager.needs_setup():
            self.lbl_intro.setText("Nothing selected to install.")
        else:
            self.lbl_intro.setText("Everything RelayVSR needs is installed.")

        if self._gpu:
            self.lbl_gpu.setText(f"GPU: {self._gpu}")
            styles.set_role(self.lbl_gpu, "faint")
        else:
            self.lbl_gpu.setText("No NVIDIA GPU found. RelayVSR needs one (with a recent driver); "
                                 "only FFmpeg can be installed on this PC.")
            styles.set_role(self.lbl_gpu, "status-warn")
        short = bool(selected) and free < need_disk
        self.lbl_disk.setText(f"Free space on that drive: {free:.0f} GB"
                              + (f" — about {need_disk:.0f} GB is needed. Free up space first." if short else ""))
        styles.set_role(self.lbl_disk, "status-warn" if short else "faint")

        self.btn_primary.setVisible(bool(selected))
        self.btn_primary.setEnabled(not short)
        self.btn_secondary.setText("Not now" if selected else "Close")
        self.btn_secondary.setEnabled(True)

    def _show_progress(self, on: bool):
        for w in (self.lbl_step, self.bar, self.lbl_count):
            w.setVisible(on)
        self.adjustSize()  # grow/shrink with the progress lines instead of squeezing the rows

    def _set_message(self, text: str, error: bool):
        self.lbl_error.setText(text)
        styles.set_role(self.lbl_error, "status-error" if error else "muted")

    # -- actions ------------------------------------------------------------
    def _edit_token(self):
        tok, ok = QInputDialog.getText(
            self, "Hugging Face access token",
            "Paste a Read token (starts with hf_). Create one at huggingface.co/settings/tokens\n"
            "after accepting the license at huggingface.co/Lightricks/LTX-2.5.",
            text=comfy_models.get_hf_token() or "")
        if ok and tok.strip():
            comfy_models.set_hf_token(tok.strip())
            self._set_message("Token saved.", error=False)

    def _on_primary(self):
        self._set_message("", error=False)
        self._show_progress(True)
        self.lbl_step.setText("Starting…")
        self.bar.setRange(0, 0)
        self.lbl_count.setText("")
        self.btn_primary.setVisible(False)
        self.btn_secondary.setText("Cancel")
        for box, _ in self.rows.values():
            if box is not None:
                box.setEnabled(False)
        self.worker = SetupWorker(self._selected())
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_done)
        self.worker.start()

    def _on_secondary(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel.set()
            self.btn_secondary.setEnabled(False)
            self.lbl_step.setText("Cancelling…")
        else:
            self.accept()

    def _on_progress(self, text: str, done, total):
        self.lbl_step.setText(text)
        if total:
            self.bar.setRange(0, 1000)
            self.bar.setValue(int(done * 1000 / total))
            self.lbl_count.setText(f"{_size(done)} of {_size(total)}")
        else:
            self.bar.setRange(0, 0)
            self.lbl_count.setText("")

    def _on_done(self, result):
        # self.worker is kept: this arrives while its thread is still finishing, and dropping a
        # running QThread aborts the app
        self._show_progress(False)
        self.refresh()
        self.changed.emit()
        if result == "cancelled":
            self._set_message("Cancelled. Install continues from where it stopped.", error=False)
        elif isinstance(result, DownloadForbidden):
            self._set_message(f"{friendly_message(result)}\n\nThen Install again; finished parts are kept.", error=True)
            self.btn_primary.setText("Retry")
        elif result is not None:
            self._set_message(f"Setup stopped: {friendly_message(result)}\n\nInstall again to retry; "
                              "finished parts are kept.", error=True)
            self.btn_primary.setText("Retry")
        elif not setup_manager.needs_setup():
            self.lbl_intro.setText("All set. You can upscale now.")

    def reject(self):  # Esc / window close while installing: cancel first
        if self.worker is not None and self.worker.isRunning():
            self._on_secondary()
            return
        super().reject()

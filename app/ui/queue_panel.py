"""
Queue panel: one compact row per job (name, status, progress, actions) under a header with
Add videos / Clear finished / Start-Pause. Long names are elided, never scrolled sideways.
"""

from pathlib import Path
from typing import Optional, Callable
import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QProgressBar, QFileDialog
)
from PySide6.QtCore import Qt, Signal

import config
import presets
import user_settings
from app.core.queue_manager import QueueManager, QueueJob
from app.ui.styles import set_role
from app.ui.widgets import ElidedLabel


def choose_batch_output_folder(parent, count: int) -> Optional[Path]:
    """Asks where a multi-file batch should be written (starts at the remembered output folder).
    The choice becomes the remembered folder; None means the user cancelled."""
    folder = QFileDialog.getExistingDirectory(
        parent, f"Choose Output Folder for {count} Videos", str(user_settings.get_last_output_folder())
    )
    if not folder:
        return None
    user_settings.set_last_output_folder(folder)
    return Path(folder)


# status -> (label, style role)
_STATUS = {
    "pending": ("Waiting", "status-idle"),
    "processing": ("Running", "status-info"),
    "completed": ("Done", "status-ok"),
    "failed": ("Failed", "status-error"),
    "canceled": ("Stopped", "status-warn"),
}


class QueueItemWidget(QFrame):
    """One job row: name · status · actions, then a thin progress bar and the current step."""
    remove_requested = Signal(str)
    stop_requested = Signal(str)
    resume_requested = Signal(str)
    preview_requested = Signal(object)  # QueueJob

    def __init__(self, job: QueueJob, parent=None):
        super().__init__(parent)
        self.job = job
        self.setObjectName("queueRow")
        self.setStyleSheet("QFrame#queueRow { border-bottom: 1px solid #222832; }")
        self._init_ui()
        self.update_job(job)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 8, 4, 8)
        layout.setSpacing(4)

        row = QHBoxLayout()
        row.setSpacing(10)
        self.lbl_title = ElidedLabel(self.job.filename, elide=Qt.ElideMiddle)
        row.addWidget(self.lbl_title, stretch=1)

        self.lbl_status_badge = QLabel()
        row.addWidget(self.lbl_status_badge)

        def action(text, tip, slot, role="ghost"):
            b = QPushButton(text)
            b.setObjectName(role)
            b.setToolTip(tip)
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(slot)
            row.addWidget(b)
            return b

        self.btn_preview = action("View", "Compare the result with its source in the player", self._on_preview_clicked)
        self.btn_open = action("Folder", "Show the output in Explorer", self._on_open_folder)
        self.btn_resume = action("Resume", "Continue from the last finished part",
                                 lambda: self.resume_requested.emit(self.job.job_id))
        # ✕ removes a job; while it is processing the same button stops it (keeping the finished part)
        self.btn_remove = action("✕", "", lambda: (self.stop_requested if self.job.status == "processing"
                                                   else self.remove_requested).emit(self.job.job_id), role="danger")
        layout.addLayout(row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        self.lbl_step = ElidedLabel("")
        self.lbl_step.setObjectName("faint")
        layout.addWidget(self.lbl_step)

    def update_job(self, job: QueueJob):
        self.job = job
        status = job.status
        self.shown_status = status  # the job object is shared, so the panel compares against this
        label, role = _STATUS.get(status, (status.title(), "status-idle"))
        if status == "processing":
            label = f"{job.progress}%"
        self.lbl_status_badge.setText(label)
        set_role(self.lbl_status_badge, role)

        self.setToolTip(f"{job.preset_name}\n→ {job.output_path}")
        self.progress_bar.setValue(job.progress)
        self.progress_bar.setVisible(status in ("processing", "canceled", "failed") and job.progress > 0
                                     or status == "processing")
        set_role(self.progress_bar, "stopped" if status != "processing" else "")

        step = {
            "pending": f"→ {Path(job.output_path).name}",
            "completed": f"Saved {Path(job.output_path).name}",
        }.get(status, job.current_step_text or "")
        self.lbl_step.setText(step)
        set_role(self.lbl_step, "faint")

        self.btn_preview.setVisible(status == "completed")
        self.btn_open.setVisible(status in ("completed", "canceled", "failed"))
        self.btn_resume.setVisible(status in ("canceled", "failed"))
        self.btn_remove.setText("Stop" if status == "processing" else "✕")
        self.btn_remove.setToolTip(
            "Stop — keeps the finished part so you can resume later" if status == "processing" else "Remove from queue"
        )

    def _on_open_folder(self):
        out = Path(self.job.output_path)
        if out.exists():
            os.system(f'explorer /select,"{out}"')
        elif out.parent.exists():
            os.startfile(str(out.parent))

    def _on_preview_clicked(self):
        if Path(self.job.output_path).exists():
            self.preview_requested.emit(self.job)


class QueuePanel(QFrame):
    """Batch queue below the player."""
    preview_requested = Signal(object)  # QueueJob

    def __init__(self, queue_manager: QueueManager, params_provider: Optional[Callable] = None,
                 status_callback: Optional[Callable[[str], None]] = None, parent=None):
        super().__init__(parent)
        self.qm = queue_manager
        self.params_provider = params_provider
        self.status_callback = status_callback or (lambda _msg: None)
        self.setObjectName("panel")
        self.item_widgets: dict[str, QueueItemWidget] = {}

        self._init_ui()
        self._connect_signals()
        self._refresh_all_items()

    def _init_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(16, 12, 16, 8)
        self.main_layout.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(8)
        title = QLabel("Queue")
        title.setObjectName("heading")
        header.addWidget(title)
        self.pill_queue_status = QLabel("")
        self.pill_queue_status.setObjectName("muted")
        header.addWidget(self.pill_queue_status)
        header.addStretch()

        self.btn_add_files = QPushButton("Add videos…")
        self.btn_add_files.setToolTip("Queue videos from disk with the current settings")
        self.btn_add_files.clicked.connect(self._on_add_files_clicked)
        header.addWidget(self.btn_add_files)

        self.btn_clear = QPushButton("Clear finished")
        self.btn_clear.setObjectName("ghost")
        self.btn_clear.clicked.connect(self.qm.clear_completed)
        header.addWidget(self.btn_clear)

        self.btn_start_queue = QPushButton("Start queue")
        self.btn_start_queue.setObjectName("primary")
        self.btn_start_queue.clicked.connect(self._on_toggle_queue_run)
        header.addWidget(self.btn_start_queue)
        self.main_layout.addLayout(header)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_content = QWidget()
        self.list_layout = QVBoxLayout(self.scroll_content)
        self.list_layout.setContentsMargins(0, 0, 4, 0)
        self.list_layout.setSpacing(0)
        self.list_layout.addStretch()
        self.scroll_area.setWidget(self.scroll_content)
        self.main_layout.addWidget(self.scroll_area, stretch=1)

        self.lbl_empty = QLabel("Nothing queued. Use Add to queue, or drop several videos on the window.")
        self.lbl_empty.setObjectName("faint")
        self.lbl_empty.setAlignment(Qt.AlignCenter)
        self.lbl_empty.setWordWrap(True)
        self.main_layout.addWidget(self.lbl_empty, stretch=1)

    def _connect_signals(self):
        self.qm.job_added.connect(self._on_job_added)
        self.qm.job_updated.connect(self._on_job_updated)
        self.qm.job_removed.connect(self._on_job_removed)
        self.qm.queue_status_changed.connect(lambda running, *_: self._update_status_pill(running))
        self.qm.queue_cleared.connect(self._refresh_all_items)

    def _refresh_all_items(self):
        for wid in self.item_widgets.values():
            wid.deleteLater()
        self.item_widgets.clear()
        for job in self.qm.jobs:
            self._create_job_widget(job)
        self._update_empty_visibility()
        self._update_status_pill(self.qm.is_running)

    def _create_job_widget(self, job: QueueJob):
        widget = QueueItemWidget(job)
        widget.remove_requested.connect(self.qm.remove_job)
        widget.stop_requested.connect(lambda _job_id: self.qm.stop_queue())
        widget.resume_requested.connect(self.qm.resume_job)
        widget.preview_requested.connect(self.preview_requested.emit)
        self.list_layout.insertWidget(max(0, self.list_layout.count() - 1), widget)
        self.item_widgets[job.job_id] = widget
        self._update_empty_visibility()

    def _on_job_added(self, job: QueueJob):
        if job.job_id not in self.item_widgets:
            self._create_job_widget(job)
        self._update_status_pill(self.qm.is_running)

    def _on_job_updated(self, job: QueueJob):
        widget = self.item_widgets.get(job.job_id)
        if widget is None:
            return
        was_running = widget.shown_status == "processing"
        widget.update_job(job)
        if job.status == "processing" and not was_running:
            self.scroll_area.ensureWidgetVisible(widget, 0, 40)  # bring the running job into view

    def _on_job_removed(self, job_id: str):
        if job_id in self.item_widgets:
            self.item_widgets.pop(job_id).deleteLater()
        self._update_empty_visibility()
        self._update_status_pill(self.qm.is_running)

    def _update_empty_visibility(self):
        has_items = len(self.qm.jobs) > 0
        self.lbl_empty.setVisible(not has_items)
        self.scroll_area.setVisible(has_items)

    def _update_status_pill(self, is_running: bool):
        counts = {}
        for j in self.qm.jobs:
            counts[j.status] = counts.get(j.status, 0) + 1
        parts = [f"{counts[k]} {word}" for k, word in
                 (("processing", "running"), ("pending", "waiting"), ("canceled", "stopped"),
                  ("failed", "failed"), ("completed", "done")) if counts.get(k)]
        self.pill_queue_status.setText("  ·  ".join(parts))
        self.btn_clear.setVisible(bool(counts.get("completed")))
        paused = is_running and self.qm.is_paused
        self.btn_start_queue.setText("Resume queue" if paused else "Pause after this job" if is_running else "Start queue")
        set_role(self.btn_start_queue, "" if is_running and not paused else "primary")
        self.btn_start_queue.setEnabled(is_running or bool(counts.get("pending")))

    def _on_toggle_queue_run(self):
        if self.qm.is_running and not self.qm.is_paused:
            self.qm.pause_queue()
        else:
            self.qm.start_queue()

    def _on_add_files_clicked(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add Videos to the Queue", str(user_settings.get_last_input_folder()),
            "Video Files (*.mp4 *.mov *.avi *.mkv *.webm)"
        )
        if not files:
            return
        user_settings.set_last_input_folder(Path(files[0]).parent)
        out_dir = None
        if len(files) > 1:
            out_dir = choose_batch_output_folder(self, len(files))
            if out_dir is None:
                return

        cur_params = None
        if self.params_provider:
            try:
                cur_params = self.params_provider()
            except Exception:
                cur_params = None

        if cur_params:
            p_id = cur_params.get("preset", presets.DEFAULT_PRESET)
            p_name = cur_params.get("preset_name", presets.get_preset(p_id).get("name", "Batch Restore"))
            base_params = dict(cur_params)
        else:
            p_id = presets.DEFAULT_PRESET
            p_data = presets.get_preset(p_id)
            p_name = p_data.get("name", "Batch Restore")
            base_params = {
                "engine_type": p_data.get("engine_type", "diffusion"),
                "superres_model": p_data.get("superres_model", "RealESRGAN_x2plus.pth"),
                "guide_strength": p_data.get("guide_strength", 0.72),
                "cfg": p_data.get("cfg", 2.5),
                "sol_sigma": p_data.get("sol_sigma", 0.50),
                "enable_teacache": p_data.get("enable_teacache", False),
                "teacache_thresh": p_data.get("teacache_thresh", 0.06),
                "mode": p_data.get("mode", "restore"),
                "preset": p_id,
                "positive_prompt": p_data.get("positive"),
                "negative_prompt": p_data.get("negative"),
                "output_size": p_data.get("output_size", "FullHD"),
                "tile_size": p_data.get("tile_size", "FullHD"),
                "seed": 42,
                "clean_noise": True,
                "stabilize_jitter": True,
                "deinterlace": False,
            }

        added = self.qm.add_batch_files(
            file_paths=files,
            base_params=base_params,
            preset_id=p_id,
            preset_name=p_name,
            output_dir=out_dir,
        )
        self.status_callback(f"Queued {len(added)} video(s) — press Start queue to process them.")

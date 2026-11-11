"""
Main window: wires the header, the settings sidebar, the player and the queue to the app's services
(video loader, job submission, setup, system status). Decisions live in app.core; this file is the
glue and the dialogs.
"""

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog, QMainWindow, QMenu, QMessageBox, QSplitter, QStatusBar, QVBoxLayout, QWidget
)

import comparison_generator
import config
import setup_manager
import user_settings
from app.core import job_submission
from app.core.queue_manager import QueueJob, QueueManager
from app.core.system_status import SystemStatusMonitor
from app.core.video_loader import VideoLoader
from app.ui.header_bar import HeaderBar
from app.ui.queue_panel import QueuePanel, choose_batch_output_folder
from app.ui.settings_panel import SettingsPanel
from app.ui.setup_dialog import SetupDialog
from app.ui.styles import DARK_THEME_QSS
from app.ui.video_viewport import VIDEO_EXTS, VideoPlayerWidget
from app.ui.widgets import ProgressDialog
from engine import ComfyUIClient, VideoRestorationPipeline
from errors import FFmpegMissing, SetupRequired, friendly_message
from version import __version__


class ComparisonWorker(QThread):
    """Renders the side-by-side and split comparison videos off the UI thread."""

    rendered = Signal(str, str)
    error = Signal(str)

    def __init__(self, original_path: str, restored_path: str):
        super().__init__()
        self.original_path = Path(original_path)
        self.restored_path = Path(restored_path)

    def run(self):
        try:
            out_dir = config.COMPARISONS_DIR
            out_dir.mkdir(parents=True, exist_ok=True)
            sbs = out_dir / f"{self.original_path.stem}_vs_{self.restored_path.stem}_sbs.mp4"
            split = out_dir / f"{self.original_path.stem}_vs_{self.restored_path.stem}_split.mp4"
            comparison_generator.generate_side_by_side(self.original_path, self.restored_path, sbs)
            comparison_generator.generate_split_screen(self.original_path, self.restored_path, split)
            self.rendered.emit(str(sbs), str(split))
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Video Restore Studio {__version__}")
        self.resize(1440, 900)
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(DARK_THEME_QSS)
        self.setAcceptDrops(True)

        self.client = ComfyUIClient()
        self.queue_manager = QueueManager(VideoRestorationPipeline(self.client))
        self.loader = VideoLoader(self)
        self.status_monitor = SystemStatusMonitor(self.client, parent=self)

        self.current_raw_video: Optional[str] = None
        self.current_restored_video: Optional[str] = None
        self.cmp_worker: Optional[ComparisonWorker] = None
        self._last_status: tuple = (None, None)  # ComfyUI stats, free VRAM: re-applied when the engine changes
        self.load_dialog = ProgressDialog(self, "Opening video")

        self._init_ui()
        self._connect_services()
        self.status_monitor.poll()
        # First run (or anything missing): offer the downloads once the window is up
        QTimer.singleShot(400, lambda: setup_manager.needs_setup() and self.open_setup())

    # -----------------------------------------------------------------
    # Layout and wiring
    # -----------------------------------------------------------------
    def _init_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 8, 12, 4)
        main_layout.setSpacing(8)

        menu = QMenu(self)
        menu.addAction("Load sample video", self._load_sample)
        menu.addAction("Make comparison video…", self._on_generate_comparison)
        menu.addSeparator()
        self.act_launch_comfy = menu.addAction("Launch ComfyUI", self._launch_comfyui)
        self.act_launch_comfy.setVisible(self._comfyui_launcher().exists())
        menu.addAction("Set up downloads…", self.open_setup)
        self.header = HeaderBar(menu)
        self.header.setup_clicked.connect(self.open_setup)
        self.header.outputs_clicked.connect(self._open_outputs_folder)
        self.header.set_setup_needed(setup_manager.needs_setup())
        main_layout.addWidget(self.header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.settings_panel = SettingsPanel(self, comfyui_available=setup_manager.comfyui_installed())
        splitter.addWidget(self.settings_panel)

        self.player_widget = VideoPlayerWidget()
        self.queue_panel = QueuePanel(
            self.queue_manager,
            params_provider=self.settings_panel.get_current_restoration_params,
            status_callback=lambda msg: self.status_bar.showMessage(msg, 8000),
            parent=self,
        )
        right = QSplitter(Qt.Vertical)
        right.setChildrenCollapsible(False)
        right.addWidget(self.player_widget)
        right.addWidget(self.queue_panel)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 1)
        right.setSizes([580, 260])
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1080])
        main_layout.addWidget(splitter, stretch=1)

        self.status_bar = QStatusBar()
        self.status_bar.setSizeGripEnabled(False)
        self.setStatusBar(self.status_bar)

    def _connect_services(self):
        sp = self.settings_panel
        sp.videos_selected.connect(self._on_videos_selected)
        sp.start_restore_clicked.connect(self._on_start_restoration)
        sp.add_to_queue_clicked.connect(self._on_add_to_queue)
        sp.rescan_requested.connect(self._rescan)
        sp.engine_changed.connect(lambda _engine: self._apply_system_status(*self._last_status))

        self.player_widget.videos_selected.connect(self._on_videos_selected)
        self.player_widget.in_out_changed.connect(sp.on_in_out_changed)
        self.player_widget.compare_requested.connect(self._on_generate_comparison)
        self.queue_panel.preview_requested.connect(self._on_queue_preview_requested)

        self.loader.step.connect(self.load_dialog.set_step)
        self.loader.info_ready.connect(self._on_video_info)
        self.loader.scanned.connect(self._on_video_scanned)
        self.loader.failed.connect(self._on_video_load_failed)

        qm = self.queue_manager
        qm.job_started.connect(self._on_queue_job_started)
        qm.job_updated.connect(self._on_queue_job_updated)
        qm.job_completed.connect(self._on_queue_job_completed)
        qm.job_failed.connect(self._on_queue_job_failed)
        qm.queue_status_changed.connect(self._on_queue_status_changed)

        self.status_monitor.status.connect(self._apply_system_status)

    def keyPressEvent(self, event):
        """Player shortcuts (Space, I, O, X, arrows, Home/End) work anywhere in the window;
        keys a focused input uses (typing, combo arrows) never reach here."""
        if not self.player_widget.handle_key(event.key(), bool(event.modifiers() & Qt.ShiftModifier)):
            super().keyPressEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and any(
                Path(u.toLocalFile()).suffix.lower() in VIDEO_EXTS for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        video_files = [u.toLocalFile() for u in event.mimeData().urls()
                       if Path(u.toLocalFile()).suffix.lower() in VIDEO_EXTS]
        if video_files:
            self._on_videos_selected(video_files)
            event.acceptProposedAction()

    def closeEvent(self, event):
        """Asks before closing while the queue is working."""
        if not self.queue_manager.is_running:
            event.accept()
            return
        ret = QMessageBox.question(
            self, "Restoration in Progress",
            "A video restoration job is currently actively processing.\n\n"
            "Are you sure you want to cancel the queue and exit?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ret == QMessageBox.Yes:
            self.queue_manager.stop_queue()
            event.accept()
        else:
            event.ignore()

    # -----------------------------------------------------------------
    # Setup and system status
    # -----------------------------------------------------------------
    def open_setup(self, preselect: tuple = ()):
        dlg = SetupDialog(self, preselect)
        dlg.changed.connect(lambda: self.header.set_setup_needed(setup_manager.needs_setup()))
        dlg.exec()
        self.header.set_setup_needed(setup_manager.needs_setup())

    def _apply_system_status(self, stats: Optional[dict], free_gb: Optional[float]):
        self._last_status = (stats, free_gb)
        online = stats is not None
        needed = job_submission.needs_comfyui({"engine_type": self.settings_panel.current_engine()})
        self.header.set_status(online, free_gb, needed, setup_manager.comfyui_installed())
        self.act_launch_comfy.setEnabled(not online)

    @staticmethod
    def _comfyui_launcher() -> Path:
        return config.PROJECT_DIR / "start_comfyui.bat"

    def _launch_comfyui(self):
        # Its own console on purpose: ComfyUI's log is the place to watch it start
        subprocess.Popen(["cmd.exe", "/c", str(self._comfyui_launcher())], creationflags=subprocess.CREATE_NEW_CONSOLE)
        self.status_bar.showMessage("Launching ComfyUI in background...", 5000)

    # -----------------------------------------------------------------
    # Opening videos
    # -----------------------------------------------------------------
    def _on_videos_selected(self, video_files: list):
        """One video: load it. Several (browse multi-select or drag & drop): load the first in the
        player and queue them all with the current settings (each at full length)."""
        if not video_files:
            return
        self._open_video(video_files[0])
        if len(video_files) < 2:
            return
        out_dir = choose_batch_output_folder(self, len(video_files))
        if out_dir is None:  # cancelled: the first video stays loaded, nothing is queued
            self.status_bar.showMessage("Batch not queued — no output folder chosen.", 6000)
            return
        params = self.settings_panel.get_current_restoration_params() or {}
        preset_id, preset_name = job_submission.preset_identity(params, "Batch Restore")
        added = self.queue_manager.add_batch_files(file_paths=video_files, base_params=params, preset_id=preset_id,
                                                   preset_name=preset_name, output_dir=out_dir)
        self.status_bar.showMessage(
            f"Queued {len(added)} videos → {out_dir} — press Start queue to process them all.", 8000)

    def _load_sample(self):
        sample_path = config.INPUTS_DIR / "sample_archive_540p.mp4"
        if sample_path.exists():
            self._open_video(str(sample_path))
        else:
            self.status_bar.showMessage(f"Sample not found: {sample_path}", 6000)

    def _open_video(self, file_path: str):
        """Reads the video and scans it on the loader; the dialog shows each step if it takes a moment."""
        self.current_raw_video = file_path
        self.current_restored_video = None
        user_settings.set_last_input_folder(Path(file_path).parent)
        self.settings_panel.begin_video(file_path)
        self.load_dialog.start(Path(file_path).name)
        self.loader.load(file_path)

    def _rescan(self, file_path: str):
        self.settings_panel.show_scanning()
        self.loader.scan(file_path)

    def _on_video_info(self, path: str, info: Dict[str, Any]):
        self.load_dialog.set_step("Opening the player")
        self.settings_panel.set_video_metadata(info)
        if self.current_restored_video and Path(path) == Path(self.current_raw_video or ""):
            self.player_widget.load_pair(path, self.current_restored_video)  # previewing a finished job
        else:
            self.player_widget.load_video(path, is_restored=False)

    def _on_video_scanned(self, path: str, res: Dict[str, Any]):
        self.settings_panel.apply_detection(res)
        self.load_dialog.finish()
        self.status_bar.showMessage(f"Loaded {Path(path).name}", 4000)

    def _on_video_load_failed(self, path: str, error: Exception):
        self.current_raw_video = None
        self.settings_panel.clear_video()
        if isinstance(error, FFmpegMissing):
            self.load_dialog.finish()
            self.open_setup()
            return
        self.load_dialog.fail("Couldn't open this video",
                              f"FFmpeg couldn't read this file. It may be damaged or not a video.\n\n"
                              f"{friendly_message(error)}")

    # -----------------------------------------------------------------
    # Submitting jobs
    # -----------------------------------------------------------------
    def _ask_output_file(self, params: Dict[str, Any], title: str) -> Optional[Path]:
        """Save dialog prefilled with an unused <source>_<tag>.mp4; remembers the folder chosen."""
        chosen, _ = QFileDialog.getSaveFileName(self, title, str(job_submission.suggested_output_path(params)),
                                                "MP4 Video (*.mp4);;All Files (*.*)")
        if not chosen:
            return None
        path = Path(chosen)
        user_settings.set_last_output_folder(path.parent)
        params["output_file"] = str(path)
        params["output_dir"] = str(path.parent)
        return path

    def _engine_ready(self, params: Dict[str, Any]) -> bool:
        """True when the job's engine can run; otherwise offers the setup (or explains)."""
        missing = job_submission.missing_requirements(params)
        if not missing:
            return True
        engine_type = params.get("engine_type", "diffusion")
        if engine_type == "relayvsr":
            self.open_setup()
            return not job_submission.missing_requirements(params)
        if engine_type == "superres":
            QMessageBox.warning(self, "Upscale model missing", "Not installed:\n- " + "\n- ".join(missing))
            return False
        # LTX / SoL-Refiner: download the models, or run anyway (delivers the preprocessed video)
        box = QMessageBox(self)
        box.setWindowTitle("Models required")
        box.setIcon(QMessageBox.Information)
        box.setText("This engine's models aren't installed yet:\n- " + "\n- ".join(missing))
        btn_setup = box.addButton("Set up downloads…", QMessageBox.AcceptRole)
        btn_anyway = box.addButton("Continue (preprocessed video only)", QMessageBox.ActionRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(btn_setup)
        box.exec()
        if box.clickedButton() is btn_setup:
            self.open_setup(preselect=("sol",) if engine_type == "sol_refiner" else ("ltx",))
            return False
        return box.clickedButton() is btn_anyway

    def _confirm_long_footage(self, params: Dict[str, Any]) -> bool:
        """For long LTX ranges, offers a short preview render first. False = cancelled."""
        long = job_submission.long_footage(params)
        if long is None:
            return True
        box = QMessageBox(self)
        box.setWindowTitle("Long Footage Detected")
        box.setIcon(QMessageBox.Information)
        box.setText(
            f"Footage: {Path(params['input_path']).name}\n"
            f"Selected Range: {long['frames']} frames (~{long['seconds']:.1f}s)\n\n"
            f"Neural diffusion processes video in temporal latent windows.\n"
            f"Processing {long['frames']} frames in a single pass requires ~{long['minutes']} minutes on RTX 5060 Ti.\n\n"
            f"Would you like to render a fast {job_submission.PREVIEW_FRAMES}-frame preview clip (~2.5s) first "
            f"to verify quality and settings?")
        btn_preview = box.addButton(f"Render {job_submission.PREVIEW_FRAMES}-Frame Preview (Recommended)",
                                    QMessageBox.ActionRole)
        box.addButton(f"Render Selected Range ({long['frames']} frames)", QMessageBox.ActionRole)
        btn_cancel = box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(btn_preview)
        box.exec()
        if box.clickedButton() is btn_cancel:
            return False
        if box.clickedButton() is btn_preview:
            params["max_frames"] = job_submission.PREVIEW_FRAMES
        return True

    def _queue(self, params: Dict[str, Any]) -> QueueJob:
        preset_id, preset_name = job_submission.preset_identity(params)
        return self.queue_manager.add_job(input_path=params["input_path"], output_path=params["output_file"],
                                          params=params, preset_id=preset_id, preset_name=preset_name)

    def _on_add_to_queue(self, params: Dict[str, Any]):
        if self._ask_output_file(params, "Queue Restored Video Output As...") is None:
            return
        job = self._queue(params)
        self.status_bar.showMessage(f"Added to queue: {job.filename} ➔ {job.output_filename} "
                                    f"({len(self.queue_manager.jobs)} total in queue)", 5000)

    def _on_start_restoration(self, params: Dict[str, Any]):
        """Checks the engine can run, asks for the output file, and starts it at once."""
        if not self._engine_ready(params):
            return
        if job_submission.needs_comfyui(params) and not self.status_monitor.comfy_online:
            QMessageBox.warning(self, "ComfyUI Offline",
                                f"ComfyUI is not reachable at {self.client.base_url}.\nPlease launch ComfyUI first.")
            return
        if self._ask_output_file(params, "Save Restored Video As...") is None:
            return
        if not self._confirm_long_footage(params):
            return
        job = self._queue(params)
        if not self.queue_manager.is_running:
            self.queue_manager.start_queue()
        else:
            self.status_bar.showMessage(f"Added to active queue: {job.filename} (processing in order)", 5000)

    # -----------------------------------------------------------------
    # Queue events
    # -----------------------------------------------------------------
    def _pending_jobs(self) -> int:
        return sum(1 for j in self.queue_manager.jobs if j.status == "pending")

    def _on_queue_job_started(self, job: QueueJob):
        self.settings_panel.start_tracking()
        self.settings_panel.set_progress(0, 100, f"{job.filename} — starting…")

    def _on_queue_job_updated(self, job: QueueJob):
        current = self.queue_manager.current_job
        if current and current.job_id == job.job_id:
            self.settings_panel.set_progress(job.progress, 100, job.current_step_text)

    def _on_queue_job_completed(self, job: QueueJob, result: Dict[str, Any]):
        final_vid = result.get("final_output_path") or job.output_path
        # Only switch the player to the result when it belongs to the video being shown
        if final_vid and Path(final_vid).exists() and self.current_raw_video \
                and Path(job.input_path) == Path(self.current_raw_video):
            self.current_restored_video = str(final_vid)
            self.player_widget.load_video(str(final_vid), is_restored=True)
        pending = self._pending_jobs()
        if pending == 0:
            self.settings_panel.finish_tracking(success=True)
        self.status_bar.showMessage(
            f"Finished {job.output_filename}" + (f" — {pending} left in the queue" if pending else ""), 8000)

    def _on_queue_job_failed(self, job: QueueJob, error: Exception):
        """No modal box for ordinary failures: it would hold up the rest of the queue until dismissed;
        the queue row keeps the error and a Resume button."""
        if self._pending_jobs() == 0:
            self.settings_panel.finish_tracking(success=False)
        message = (job.error_message or "unknown error").strip().splitlines()[0]
        self.status_bar.showMessage(f"{job.filename} failed: {message}", 15000)
        if isinstance(error, (SetupRequired, FFmpegMissing)):
            self.open_setup()

    def _on_queue_status_changed(self, is_running: bool, pending: int, completed: int):
        self.settings_panel.btn_start.setEnabled(bool(self.settings_panel.current_video_path))

    def _on_queue_preview_requested(self, job: QueueJob):
        """Shows a finished job against its own source (not whatever video is loaded)."""
        if not Path(job.output_path).exists():
            return
        if Path(job.input_path).exists():
            self._open_video(job.input_path)
            self.current_restored_video = job.output_path  # _on_video_info then shows the pair
        else:
            self.current_restored_video = job.output_path
            self.player_widget.load_pair(job.input_path, job.output_path)
        self.status_bar.showMessage(f"Comparing {job.output_filename} with its source", 4000)

    # -----------------------------------------------------------------
    # Comparison videos and the outputs folder
    # -----------------------------------------------------------------
    def _on_generate_comparison(self):
        if self.cmp_worker is not None and self.cmp_worker.isRunning():
            # Replacing a running QThread would abort the app
            self.status_bar.showMessage("A comparison is already rendering…", 4000)
            return
        if not self.current_raw_video:
            self.status_bar.showMessage("Open a source video first.", 5000)
            return
        target_restored = self.current_restored_video
        if not target_restored:
            target_restored, _ = QFileDialog.getOpenFileName(
                self, "Select Restored Video for Comparison", str(user_settings.get_last_output_folder()),
                "Videos (*.mp4 *.mov)")
            if not target_restored:
                return
        self.status_bar.showMessage("Rendering side-by-side and split comparison videos…")
        self.cmp_worker = ComparisonWorker(self.current_raw_video, target_restored)
        self.cmp_worker.rendered.connect(self._on_comparison_completed)
        self.cmp_worker.error.connect(lambda e: QMessageBox.critical(self, "Comparison Error", e))
        self.cmp_worker.start()

    def _on_comparison_completed(self, sbs_path: str, split_path: str):
        self.status_bar.showMessage(f"Comparison videos saved: {Path(sbs_path).name}, {Path(split_path).name}", 10000)
        os.startfile(str(Path(sbs_path).parent))

    def _open_outputs_folder(self):
        folder = user_settings.get_last_output_folder()
        if not folder.exists():
            folder = config.OUTPUTS_DIR
            folder.mkdir(parents=True, exist_ok=True)
        os.startfile(str(folder))

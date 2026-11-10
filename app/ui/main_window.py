"""
Main window: header (status, outputs, menu), the settings sidebar, and the player above the queue.
"""

import os
import sys
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, Any

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QFileDialog, QMessageBox,
    QStatusBar, QInputDialog, QToolButton, QMenu
)

import config
import model_downloader
import video_preprocessor
import comparison_generator
from restore_engine import VideoRestorationPipeline, ComfyUIClient

from app.ui import styles
from app.ui.styles import DARK_THEME_QSS
from app.ui.video_viewport import VideoPlayerWidget, VIDEO_EXTS
from app.ui.settings_panel import SettingsPanel
from app.core.queue_manager import QueueManager, QueueJob
from app.ui.queue_panel import QueuePanel, choose_batch_output_folder
from app.ui.widgets import ProgressDialog



class ComparisonWorker(QThread):
    """Background worker for rendering comparison videos."""

    finished = Signal(str, str)
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
            
            self.finished.emit(str(sbs), str(split))
        except Exception as e:
            self.error.emit(str(e))


class VideoLoadWorker(QThread):
    """Reads a video's details and scans it for defects off the UI thread, reporting each step."""
    step = Signal(str, int, int)  # text, current, total (total 0 = busy)
    info_ready = Signal(dict)
    analysis_ready = Signal(dict)
    failed = Signal(str)

    def __init__(self, video_path: str):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            self.step.emit("Reading video details", 0, 0)
            self.info_ready.emit(video_preprocessor.get_video_info(self.video_path))
        except Exception as e:
            self.failed.emit(f"Couldn't read this video: {e}")
            return
        try:
            res = video_preprocessor.analyze_video_defects(
                self.video_path, progress=lambda cur, tot, text: self.step.emit(text, cur, tot))
        except Exception as e:
            res = {"error": str(e)}
        self.analysis_ready.emit(res)


class MainWindow(QMainWindow):
    """Main application window for LTX-2.5 Video Restoration."""
    status_ready = Signal(object, object)  # ComfyUI /system_stats (None = offline), free VRAM GB (None = unknown)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Restore Studio")
        self.resize(1440, 900)
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(DARK_THEME_QSS)
        self.setAcceptDrops(True)

        self.client = ComfyUIClient()
        self.pipeline = VideoRestorationPipeline(self.client)
        self.queue_manager = QueueManager(self.pipeline)

        self.current_raw_video: Optional[str] = None
        self.current_restored_video: Optional[str] = None
        self._load_workers: set = set()  # a running QThread must not be dropped
        self.load_dialog = ProgressDialog(self, "Opening video")

        self._init_ui()
        self._init_status_timer()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if Path(url.toLocalFile()).suffix.lower() in VIDEO_EXTS:
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event):
        video_files = [
            url.toLocalFile() for url in event.mimeData().urls()
            if Path(url.toLocalFile()).suffix.lower() in VIDEO_EXTS
        ]
        if video_files:
            self._on_videos_selected(video_files)
            event.acceptProposedAction()

    def _on_videos_selected(self, video_files: list):
        """One video: load it. Several (browse multi-select or drag & drop): load the first in the
        player and queue them all with the current settings (each at full length)."""
        if not video_files:
            return
        self._on_video_loaded(video_files[0])
        if len(video_files) > 1:
            out_dir = choose_batch_output_folder(self, len(video_files))
            if out_dir is None:  # cancelled: the first video stays loaded, nothing is queued
                self.status_bar.showMessage("Batch not queued — no output folder chosen.", 6000)
                return
            params = self.settings_panel.get_current_restoration_params() or {}
            preset_id = params.get("preset", config.DEFAULT_PRESET)
            added = self.queue_manager.add_batch_files(
                file_paths=video_files,
                base_params=params,
                preset_id=preset_id,
                preset_name=params.get("preset_name", config.get_preset(preset_id).get("name", "Batch Restore")),
                output_dir=out_dir,
            )
            self.status_bar.showMessage(
                f"Queued {len(added)} videos → {out_dir} — press Start queue to process them all.", 8000
            )

    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 8, 12, 4)
        main_layout.setSpacing(8)

        # Header: name · ComfyUI / GPU status · Outputs · menu
        header = QHBoxLayout()
        header.setContentsMargins(4, 0, 0, 0)
        header.setSpacing(12)
        logo = QLabel()
        dpr = self.devicePixelRatioF()
        pm = QPixmap(str(styles._ASSETS_DIR / "logo.png"))
        pm = pm.scaled(round(22 * dpr), round(22 * dpr), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pm.setDevicePixelRatio(dpr)
        logo.setPixmap(pm)
        header.addWidget(logo)
        header.addSpacing(-4)
        title_lbl = QLabel("Video Restore")
        title_lbl.setObjectName("title")
        header.addWidget(title_lbl)
        header.addStretch()

        self.lbl_server_status = QLabel("ComfyUI …")
        self.lbl_server_status.setObjectName("faint")
        self.lbl_server_status.setToolTip("Needed for the LTX and SoL engines; RelayVSR runs without it")
        header.addWidget(self.lbl_server_status)
        self.lbl_vram = QLabel("")
        self.lbl_vram.setObjectName("faint")
        self.lbl_vram.setToolTip("Free GPU memory (all apps). RelayVSR keeps part of its model in system RAM when this is short of ~14 GB.")
        header.addWidget(self.lbl_vram)

        self.btn_open_folder = QPushButton("Outputs")
        self.btn_open_folder.setToolTip("Open the output folder")
        self.btn_open_folder.clicked.connect(self._open_outputs_folder)
        header.addWidget(self.btn_open_folder)

        menu = QMenu(self)
        menu.addAction("Load sample video", self._load_sample)
        menu.addAction("Make comparison video…", self._on_generate_comparison)
        menu.addSeparator()
        self.act_launch_comfy = menu.addAction("Launch ComfyUI", self._launch_comfyui)
        menu.addAction("Models…", self._open_models_status)
        self.btn_more = QToolButton()
        self.btn_more.setText("☰")
        self.btn_more.setObjectName("icon")
        self.btn_more.setToolTip("More")
        self.btn_more.setPopupMode(QToolButton.InstantPopup)
        self.btn_more.setMenu(menu)
        header.addWidget(self.btn_more)
        main_layout.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        self.settings_panel = SettingsPanel(self)
        self.settings_panel.videos_selected.connect(self._on_videos_selected)
        self.settings_panel.start_restore_clicked.connect(self._on_start_restoration)
        self.settings_panel.add_to_queue_clicked.connect(self._on_add_to_queue)
        splitter.addWidget(self.settings_panel)

        self.player_widget = VideoPlayerWidget()
        self.player_widget.videos_selected.connect(self._on_videos_selected)
        self.player_widget.in_out_changed.connect(self.settings_panel.on_in_out_changed)
        self.player_widget.compare_requested.connect(self._on_generate_comparison)

        self.queue_panel = QueuePanel(
            self.queue_manager,
            params_provider=self.settings_panel.get_current_restoration_params,
            status_callback=lambda msg: self.status_bar.showMessage(msg, 8000),
            parent=self
        )
        self.queue_panel.preview_requested.connect(self._on_queue_preview_requested)

        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setChildrenCollapsible(False)
        right_splitter.addWidget(self.player_widget)
        right_splitter.addWidget(self.queue_panel)
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 1)
        right_splitter.setSizes([580, 260])

        splitter.addWidget(right_splitter)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1080])
        main_layout.addWidget(splitter, stretch=1)

        self.queue_manager.job_started.connect(self._on_queue_job_started)
        self.queue_manager.job_updated.connect(self._on_queue_job_updated)
        self.queue_manager.job_completed.connect(self._on_queue_job_completed)
        self.queue_manager.job_failed.connect(self._on_queue_job_failed)
        self.queue_manager.queue_status_changed.connect(self._on_queue_status_changed)

        self.status_bar = QStatusBar()
        self.status_bar.setSizeGripEnabled(False)
        self.setStatusBar(self.status_bar)

    def keyPressEvent(self, event):
        """Player shortcuts (Space, I, O, X, arrows, Home/End) work anywhere in the window;
        keys a focused input uses (typing, combo arrows) never reach here."""
        if not self.player_widget.handle_key(event.key(), bool(event.modifiers() & Qt.ShiftModifier)):
            super().keyPressEvent(event)

    def _load_sample(self):
        sample_path = config.INPUTS_DIR / "sample_archive_540p.mp4"
        if sample_path.exists():
            self._on_video_loaded(str(sample_path))
        else:
            self.status_bar.showMessage(f"Sample not found: {sample_path}", 6000)

    def _init_status_timer(self):
        # The HTTP probe runs on a worker thread (it blocked the UI for ~250 ms online and
        # ~2 s offline every tick); results come back through a queued signal.
        self._status_online: Optional[bool] = None
        self._status_inflight = False
        self.status_ready.connect(self._apply_system_status)
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._check_system_status)
        self.status_timer.start(4000)
        self._check_system_status()

    def _check_system_status(self):
        if self._status_inflight:
            return
        self._status_inflight = True

        def probe():
            self.status_ready.emit(self.client.try_system_stats(), VideoRestorationPipeline._gpu_free_gb())
        threading.Thread(target=probe, daemon=True).start()

    def _apply_system_status(self, stats: Optional[dict], free_gb: Optional[float]):
        self._status_inflight = False
        is_online = stats is not None
        if is_online != self._status_online:  # restyle only on transitions
            self._status_online = is_online
            self.lbl_server_status.setText("● ComfyUI" if is_online else "○ ComfyUI offline")
            styles.set_role(self.lbl_server_status, "faint" if is_online else "status-warn")
            self.act_launch_comfy.setEnabled(not is_online)
        # nvidia-smi sees every process; ComfyUI's own figure is per-process on Windows and misleading
        self.lbl_vram.setText(f"GPU {free_gb:.1f} GB free" if free_gb is not None else "")

    def _launch_comfyui(self):
        bat = config.PROJECT_DIR / "start_comfyui.bat"
        if bat.exists():
            subprocess.Popen(["cmd.exe", "/c", str(bat)], creationflags=subprocess.CREATE_NEW_CONSOLE)
            self.status_bar.showMessage("Launching ComfyUI in background...", 5000)

    def _open_models_status(self):
        status = model_downloader.get_model_status()
        missing = [k for k, v in status.items() if not v["installed"]]
        cur_tok = model_downloader.get_hf_token()
        tok_str = f"Configured ({cur_tok[:7]}...)" if cur_tok else "Not Set"

        if not missing:
            QMessageBox.information(
                self, "Models Ready",
                f"All required LTX-2.5 models and IC-LoRAs are installed!\n\n"
                f"Hugging Face Token: {tok_str}"
            )
            return

        msg = f"The following models are currently missing:\n- " + "\n- ".join(missing)
        msg += f"\n\nHugging Face Token: {tok_str}"
        if not cur_tok:
            msg += "\n\n* Note: Base LTX-2.5 models are gated on Hugging Face."
            msg += "\n  Accept license: https://huggingface.co/Lightricks/LTX-2.5"
            msg += "\n  Get free token: https://huggingface.co/settings/tokens"

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("LTX-2.5 Models Manager")
        msg_box.setText(msg)
        msg_box.setIcon(QMessageBox.Information)

        btn_dl = msg_box.addButton("Open Downloader Console", QMessageBox.ActionRole)
        btn_token = msg_box.addButton("Set HF Token", QMessageBox.ActionRole)
        btn_cancel = msg_box.addButton("Close", QMessageBox.RejectRole)

        msg_box.exec()
        clicked = msg_box.clickedButton()

        if clicked == btn_dl:
            dl_bat = config.PROJECT_DIR / "download_models.bat"
            subprocess.Popen(["cmd.exe", "/c", str(dl_bat)], creationflags=subprocess.CREATE_NEW_CONSOLE)
        elif clicked == btn_token:
            tok, ok = QInputDialog.getText(
                self, "Hugging Face Access Token",
                "Paste your Hugging Face Read Token (starts with hf_...):\n\n"
                "(Obtain from https://huggingface.co/settings/tokens)",
                text=cur_tok or ""
            )
            if ok and tok.strip():
                model_downloader.set_hf_token(tok.strip())
                QMessageBox.information(self, "Token Saved", "Hugging Face Token saved successfully!")

    def _on_video_loaded(self, file_path: str):
        """Reads the video and scans it on a worker; the dialog shows each step if it takes a moment."""
        self.current_raw_video = file_path
        self.current_restored_video = None
        config.set_last_input_folder(Path(file_path).parent)
        self.settings_panel.current_video_path = file_path  # a batch queued right after reads settings now
        self.load_dialog.start(Path(file_path).name)

        worker = VideoLoadWorker(file_path)
        current = lambda: self.current_raw_video == file_path  # ignore a load superseded by a newer one
        worker.step.connect(lambda text, cur, tot: current() and self.load_dialog.set_step(text, cur, tot))
        worker.info_ready.connect(lambda info: current() and self._on_video_info(info))
        worker.analysis_ready.connect(lambda res: current() and self._on_video_scanned(file_path, res))
        worker.failed.connect(lambda msg: current() and self._on_video_load_failed(msg))
        worker.finished.connect(lambda w=worker: self._load_workers.discard(w))
        self._load_workers.add(worker)
        worker.start()

    def _on_video_info(self, info: Dict[str, Any]):
        self.load_dialog.set_step("Opening the player")
        self.settings_panel.set_video_metadata(info, scan=False)
        self.player_widget.load_video(info["path"], is_restored=False)

    def _on_video_scanned(self, file_path: str, res: Dict[str, Any]):
        self.settings_panel.apply_detection(res)  # already known current; its path may differ in slashes
        self.load_dialog.finish()
        self.status_bar.showMessage(f"Loaded {Path(file_path).name}", 4000)

    def _on_video_load_failed(self, message: str):
        self.current_raw_video = None
        self.load_dialog.finish()
        self.status_bar.showMessage(message, 10000)

    def _on_add_to_queue(self, params: Dict[str, Any]):
        """Adds current video and settings as a new job in the batch queue."""
        input_p = Path(params["input_path"])
        last_out_folder = config.get_last_output_folder()
        engine_type = params.get("engine_type", "diffusion")
        if engine_type == "superres":
            sr_name = params.get("superres_model", "RealESRGAN_x2plus.pth").split(".")[0].lower()
            mode_tag = f"superres_{sr_name}"
        elif engine_type == "relayvsr":
            mode_tag = "relayvsr"
        elif engine_type == "sol_refiner":
            mode_tag = "sol_refiner"
        else:
            mode_tag = params.get("mode", "restored")
            if mode_tag == "restore":
                mode_tag = "restored"

        suggested_path = config.get_unique_output_path(
            output_dir=last_out_folder,
            base_stem=f"{input_p.stem}_{mode_tag}",
            suffix=".mp4"
        )

        chosen_file, _ = QFileDialog.getSaveFileName(
            self,
            "Queue Restored Video Output As...",
            str(suggested_path),
            "MP4 Video (*.mp4);;All Files (*.*)"
        )
        if not chosen_file:
            return

        chosen_file_p = Path(chosen_file)
        config.set_last_output_folder(chosen_file_p.parent)
        params["output_file"] = str(chosen_file_p)
        params["output_dir"] = str(chosen_file_p.parent)

        preset_id = params.get("preset", config.DEFAULT_PRESET)
        preset_name = params.get("preset_name", config.get_preset(preset_id).get("name", "Restoration Job"))

        job = self.queue_manager.add_job(
            input_path=params["input_path"],
            output_path=str(chosen_file_p),
            params=params,
            preset_id=preset_id,
            preset_name=preset_name
        )
        total_q = len(self.queue_manager.jobs)
        self.status_bar.showMessage(f"Added to queue: {job.filename} ➔ {job.output_filename} ({total_q} total in queue)", 5000)

    def _on_start_restoration(self, params: Dict[str, Any]):
        """Validates settings, prompts for output file, and starts queue processing immediately."""
        # RelayVSR runs in its own venv and does not need ComfyUI
        if params.get("engine_type") != "relayvsr" and not self._status_online:
            QMessageBox.warning(
                self, "ComfyUI Offline",
                "ComfyUI is not reachable at 127.0.0.1:8188.\nPlease launch ComfyUI first."
            )
            return

        # 1. Ask user for the output file with pre-populated suggestion (with collision-avoidance numbering)
        input_p = Path(params["input_path"])
        last_out_folder = config.get_last_output_folder()
        engine_type = params.get("engine_type", "superres")
        if engine_type == "superres":
            sr_name = params.get("superres_model", "RealESRGAN_x2plus.pth").split(".")[0].lower()
            mode_tag = f"superres_{sr_name}"
        elif engine_type == "relayvsr":
            mode_tag = "relayvsr"
        elif engine_type == "sol_refiner":
            mode_tag = "sol_refiner"
        else:
            mode_tag = params.get("mode", "restored")
            if mode_tag == "restore":
                mode_tag = "restored"

        suggested_path = config.get_unique_output_path(
            output_dir=last_out_folder,
            base_stem=f"{input_p.stem}_{mode_tag}",
            suffix=".mp4"
        )
        
        chosen_file, _ = QFileDialog.getSaveFileName(
            self,
            "Save Restored Video As...",
            str(suggested_path),
            "MP4 Video (*.mp4);;All Files (*.*)"
        )
        if not chosen_file:
            return

        chosen_file_p = Path(chosen_file)
        config.set_last_output_folder(chosen_file_p.parent)
        params["output_file"] = str(chosen_file_p)
        params["output_dir"] = str(chosen_file_p.parent)

        # 2. Check model status & long footage based on engine type
        if engine_type == "superres":
            sr_model = params.get("superres_model", "RealESRGAN_x2plus.pth")
            model_p = config.UPSCALE_MODELS_DIR / sr_model
            if not model_p.exists():
                QMessageBox.warning(
                    self, "Upscale Model Missing",
                    f"Super-resolution model not found: {sr_model}\nPath: {model_p}"
                )
                return
        elif engine_type == "relayvsr":
            required = [
                config.RELAYVSR_PYTHON,
                config.RELAYVSR_WEIGHTS_DIR / "wan.safetensors",
                config.RELAYVSR_WEIGHTS_DIR / "flash.safetensors",
            ]
            missing = [str(p) for p in required if not p.exists()]
            if missing:
                QMessageBox.warning(
                    self, "RelayVSR Not Installed",
                    "RelayVSR environment or weights are missing:\n- " + "\n- ".join(missing)
                    + "\n\nRun: python restore_cli.py download --relayvsr"
                )
                return
        elif engine_type == "sol_refiner":
            model_status = model_downloader.get_model_status()
            sol_keys = ["sol_transformer", "sol_vae", "spatial_upscaler", "text_encoder"]
            missing = [k for k in sol_keys if not model_status.get(k, {}).get("installed", False)]
            if missing:
                msg = (
                    f"Notice: NVIDIA SoL-Refiner neural weights are required.\n\n"
                    f"Missing components:\n- " + "\n- ".join(missing) + "\n\n"
                    f"Would you like to open the Models Downloader now to install them?\n\n"
                    f"• Click 'Yes' to launch the Models Downloader console.\n"
                    f"• Click 'No' to proceed with GPU NVENC canvas preprocessing only."
                )
                ret = QMessageBox.question(
                    self, "SoL-Refiner Models Required", msg, QMessageBox.Yes | QMessageBox.No
                )
                if ret == QMessageBox.Yes:
                    dl_bat = config.PROJECT_DIR / "download_models.bat"
                    subprocess.Popen(["cmd.exe", "/c", str(dl_bat)], creationflags=subprocess.CREATE_NEW_CONSOLE)
                    return
        else:
            # Check for long video in diffusion mode
            input_info = video_preprocessor.get_video_info(input_p)
            total_frames = input_info.get("total_frames", 0)

            in_f = params.get("start_frame")
            out_f = params.get("end_frame")
            if in_f is not None or out_f is not None:
                eff_in = in_f if in_f is not None else 0
                eff_out = out_f if out_f is not None else (total_frames - 1)
                target_span = max(1, eff_out - eff_in + 1)
            else:
                target_span = total_frames

            if target_span > 161 and not params.get("max_frames"):
                fps_val = input_info.get("fps") or 24.0
                span_sec = target_span / fps_val
                est_mins = round((target_span / 65) * (1.5 if params.get("output_size") == "FullHD" else 11), 1)
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Long Footage Detected")
                msg_box.setIcon(QMessageBox.Information)
                msg_box.setText(
                    f"Footage: {input_p.name}\n"
                    f"Selected Range: {target_span} frames (~{span_sec:.1f}s)\n\n"
                    f"Neural diffusion processes video in temporal latent windows.\n"
                    f"Processing {target_span} frames in a single pass requires ~{est_mins} minutes on RTX 5060 Ti.\n\n"
                    f"Would you like to render a fast 65-frame preview clip (~2.5s) first to verify quality and settings?"
                )
                btn_preview = msg_box.addButton("Render 65-Frame Preview (Recommended)", QMessageBox.ActionRole)
                btn_full = msg_box.addButton(f"Render Selected Range ({target_span} frames)", QMessageBox.ActionRole)
                btn_cancel = msg_box.addButton("Cancel", QMessageBox.RejectRole)
                msg_box.setDefaultButton(btn_preview)
                msg_box.exec()

                clicked = msg_box.clickedButton()
                if clicked == btn_cancel:
                    return
                elif clicked == btn_preview:
                    params["max_frames"] = 65

            # Check LTX model status
            model_status = model_downloader.get_model_status()
            missing = [k for k in model_status.items() if not k[1]["installed"]]
            if missing:
                missing_names = [k[0] for k in missing]
                msg = (
                    f"Notice: LTX-2.5 AI restoration models (~37 GB) are required for full GPU neural diffusion.\n\n"
                    f"Missing components:\n- " + "\n- ".join(missing_names) + "\n\n"
                    f"Would you like to open the Models Downloader now to install them?\n\n"
                    f"• Click 'Yes' to launch the Models Downloader console.\n"
                    f"• Click 'No' to proceed with GPU NVENC canvas preprocessing (32-multiple, 8n+1 frames) "
                    f"and workflow preparation."
                )
                ret = QMessageBox.question(
                    self, "LTX-2.5 Models Required", msg, QMessageBox.Yes | QMessageBox.No
                )
                if ret == QMessageBox.Yes:
                    dl_bat = config.PROJECT_DIR / "download_models.bat"
                    subprocess.Popen(["cmd.exe", "/c", str(dl_bat)], creationflags=subprocess.CREATE_NEW_CONSOLE)
                    return

        # Queue the job and begin execution
        preset_id = params.get("preset", config.DEFAULT_PRESET)
        preset_name = params.get("preset_name", config.get_preset(preset_id).get("name", "Restoration Job"))

        job = self.queue_manager.add_job(
            input_path=params["input_path"],
            output_path=params["output_file"],
            params=params,
            preset_id=preset_id,
            preset_name=preset_name
        )

        if not self.queue_manager.is_running:
            self.queue_manager.start_queue()
        else:
            self.status_bar.showMessage(f"Added to active queue: {job.filename} (processing in order)", 5000)

    def _on_queue_job_started(self, job: QueueJob):
        """Called by QueueManager when a new job starts processing."""
        self.settings_panel.start_tracking()
        self.settings_panel.set_progress(0, 100, f"{job.filename} — starting…")

    def _on_queue_job_updated(self, job: QueueJob):
        """Called by QueueManager on progress updates for the active job."""
        if self.queue_manager.current_job and self.queue_manager.current_job.job_id == job.job_id:
            self.settings_panel.set_progress(job.progress, 100, job.current_step_text)

    def _on_queue_job_completed(self, job: QueueJob, result: Dict[str, Any]):
        """Called by QueueManager when a job completes successfully."""
        final_vid = result.get("final_output_path") or job.output_path
        # Only switch the player to the result when it belongs to the video being shown
        if final_vid and Path(final_vid).exists() and self.current_raw_video \
                and Path(job.input_path) == Path(self.current_raw_video):
            self.current_restored_video = str(final_vid)
            self.player_widget.load_video(str(final_vid), is_restored=True)

        pending = sum(1 for j in self.queue_manager.jobs if j.status == "pending")
        if pending == 0:
            self.settings_panel.finish_tracking(success=True)
        self.status_bar.showMessage(
            f"Finished {job.output_filename}" + (f" — {pending} left in the queue" if pending else ""), 8000)

    def _on_queue_job_failed(self, job: QueueJob, err_msg: str):
        """Called by QueueManager when a job fails. No dialog: a modal box here would hold up
        the rest of the queue until dismissed; the queue row keeps the error and a Resume button."""
        pending = sum(1 for j in self.queue_manager.jobs if j.status == "pending")
        if pending == 0:
            self.settings_panel.finish_tracking(success=False)
        first_line = err_msg.strip().splitlines()[0] if err_msg.strip() else "unknown error"
        self.status_bar.showMessage(f"{job.filename} failed: {first_line}", 15000)

    def _on_queue_status_changed(self, is_running: bool, pending: int, completed: int):
        """Called by QueueManager when running/paused state or job counts change."""
        self.settings_panel.btn_start.setEnabled(bool(self.settings_panel.current_video_path))

    def _on_queue_preview_requested(self, job: QueueJob):
        """Shows a finished job against its own source (not whatever video is loaded)."""
        if not Path(job.output_path).exists():
            return
        if Path(job.input_path).exists():
            self.current_raw_video = job.input_path
            self.settings_panel.set_video_metadata(video_preprocessor.get_video_info(job.input_path))
        self.current_restored_video = job.output_path
        self.player_widget.load_pair(job.input_path, job.output_path)
        self.status_bar.showMessage(f"Comparing {job.output_filename} with its source", 4000)

    def closeEvent(self, event):
        """Prevents accidental window exit while restoration queue is running."""
        if self.queue_manager.is_running:
            ret = QMessageBox.question(
                self,
                "Restoration in Progress",
                "A video restoration job is currently actively processing.\n\n"
                "Are you sure you want to cancel the queue and exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if ret == QMessageBox.Yes:
                self.queue_manager.stop_queue()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def _on_generate_comparison(self):
        if getattr(self, "cmp_worker", None) is not None and self.cmp_worker.isRunning():
            # Replacing a running QThread would abort the app
            self.status_bar.showMessage("A comparison is already rendering…", 4000)
            return
        if not self.current_raw_video:
            self.status_bar.showMessage("Open a source video first.", 5000)
            return

        # If no restored video loaded, prompt user to select one
        target_restored = self.current_restored_video
        if not target_restored:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Restored Video for Comparison", str(config.get_last_output_folder()), "Videos (*.mp4 *.mov)"
            )
            if file_path:
                target_restored = file_path
            else:
                return

        self.status_bar.showMessage("Rendering side-by-side and split comparison videos…")
        self.cmp_worker = ComparisonWorker(self.current_raw_video, target_restored)
        self.cmp_worker.finished.connect(self._on_comparison_completed)
        self.cmp_worker.error.connect(lambda e: QMessageBox.critical(self, "Comparison Error", e))
        self.cmp_worker.start()

    def _on_comparison_completed(self, sbs_path: str, split_path: str):
        self.status_bar.showMessage(f"Comparison videos saved: {Path(sbs_path).name}, {Path(split_path).name}", 10000)
        os.startfile(str(Path(sbs_path).parent))

    def _open_outputs_folder(self):
        folder = config.get_last_output_folder()
        if folder.exists():
            os.startfile(str(folder))
        else:
            config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            os.startfile(str(config.OUTPUTS_DIR))

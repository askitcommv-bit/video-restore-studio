"""
Left sidebar: the video, the preset, and Start / Add to queue.

Everything beyond the preset lives in one collapsed "Customize" section, and within it only
the controls that apply to the selected engine are shown (RelayVSR needs none of them).
"""

from pathlib import Path
import re
import time
import threading
from typing import Optional, Dict, Any
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTextEdit, QComboBox, QSpinBox, QCheckBox,
    QProgressBar, QFileDialog, QScrollArea, QFrame,
    QRadioButton, QButtonGroup, QSlider
)

import config
import presets
import user_settings
import video_encoders
from app.ui.styles import set_role
from app.ui.widgets import ElidedLabel


def plain_name(name: str) -> str:
    """Preset names in config carry a leading emoji and trailing '(…)' / '[…]' notes;
    the UI shows just the name (the notes are in the description below the combo)."""
    name = re.sub(r"^[^\w(]+", "", name or "")
    return re.sub(r"\s*[(\[].*$", "", name).strip() or (name or "").strip()


def _label(text: str, role: str = "", wrap: bool = False) -> QLabel:
    lbl = QLabel(text)
    if role:
        lbl.setObjectName(role)
    lbl.setWordWrap(wrap)
    return lbl


def _divider() -> QFrame:
    line = QFrame()
    line.setObjectName("divider")
    return line


class SettingsPanel(QFrame):
    """Left-hand controls panel for video restoration configuration."""

    videos_selected = Signal(list)  # Browse multi-select: first is loaded, all are queued
    start_restore_clicked = Signal(dict)
    add_to_queue_clicked = Signal(dict)
    rescan_requested = Signal(str)  # "Detect": scan the current video again (the app's VideoLoader does it)
    engine_changed = Signal(str)  # the selected engine: 'relayvsr', 'superres', 'sol_refiner' or 'diffusion'

    def __init__(self, parent=None, comfyui_available: bool = True):
        """comfyui_available=False (no ComfyUI on this PC) offers only what runs without it: RelayVSR."""
        super().__init__(parent)
        self.comfyui_available = comfyui_available
        self.setObjectName("sidebar")
        self.setMinimumWidth(320)
        self.current_video_path: Optional[str] = None
        self.current_in_point: Optional[int] = None
        self.current_out_point: Optional[int] = None
        self.total_frames: int = 0
        self.fps: float = 24.0
        self._syncing_ticks: bool = False
        self._custom_prompt_edited: bool = False
        self.selected_reference_image: Optional[str] = None

        # Live restoration metrics & timer state
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._on_timer_tick)
        self._start_time: Optional[float] = None
        self._smoothed_eta: Optional[float] = None
        self._current_progress: float = 0.0
        self._is_active: bool = False
        self._is_applying_preset: bool = True

        self._init_ui()

    # -----------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------
    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(14)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(18)

        layout.addLayout(self._build_video_section())
        layout.addLayout(self._build_preset_section())
        layout.addWidget(self._build_customize_section())
        layout.addStretch()

        scroll.setWidget(content)
        root.addWidget(scroll, stretch=1)
        root.addWidget(self._build_action_area())

        # Initial preset setup: presets.DEFAULT_PRESET
        self._is_applying_preset = False
        default_idx = max(0, self.combo_presets.findData(presets.DEFAULT_PRESET))
        self.combo_presets.setCurrentIndex(default_idx)
        self._on_preset_dropdown_changed(default_idx)

    def _build_video_section(self) -> QVBoxLayout:
        box = QVBoxLayout()
        box.setSpacing(6)
        box.addWidget(_label("VIDEO", "section"))

        self.lbl_file_name = ElidedLabel("No video loaded", elide=Qt.ElideMiddle)
        self.lbl_file_name.setObjectName("heading")
        box.addWidget(self.lbl_file_name)

        self.lbl_file_meta = _label("Drop files on the window or open them below.", "muted", wrap=True)
        box.addWidget(self.lbl_file_meta)

        self.lbl_range = _label("", "faint")
        self.lbl_range.setVisible(False)
        box.addWidget(self.lbl_range)

        self.btn_browse = QPushButton("Open videos…")
        self.btn_browse.setToolTip("Pick one video to load it, or several to queue them all")
        self.btn_browse.clicked.connect(self._on_browse_clicked)
        box.addSpacing(4)
        box.addWidget(self.btn_browse)
        return box

    def _build_preset_section(self) -> QVBoxLayout:
        box = QVBoxLayout()
        box.setSpacing(6)
        box.addWidget(_label("PRESET", "section"))

        self.combo_presets = QComboBox()
        self.combo_presets.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo_presets.setMinimumContentsLength(12)
        for pid, pdata in presets.RESTORATION_PRESETS.items():
            if self.comfyui_available or pdata.get("engine_type") == "relayvsr":
                self.combo_presets.addItem(plain_name(pdata["name"]), pid)
        self.combo_presets.addItem("Custom settings", "custom")
        self.combo_presets.view().setMinimumWidth(self.combo_presets.view().sizeHintForColumn(0) + 24)
        self.combo_presets.currentIndexChanged.connect(self._on_preset_dropdown_changed)
        box.addWidget(self.combo_presets)

        self.lbl_preset_desc = _label("", "muted", wrap=True)
        box.addWidget(self.lbl_preset_desc)
        self.lbl_preset_meta = _label("", "faint", wrap=True)
        box.addWidget(self.lbl_preset_meta)
        return box

    def _build_customize_section(self) -> QWidget:
        wrapper = QWidget()
        box = QVBoxLayout(wrapper)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(10)

        self.btn_toggle_fine_tuning = QPushButton("▸  Customize")
        self.btn_toggle_fine_tuning.setObjectName("disclosure")
        self.btn_toggle_fine_tuning.clicked.connect(self._toggle_fine_tuning)
        box.addWidget(self.btn_toggle_fine_tuning)

        self.fine_tuning_content = QWidget()
        inner = QVBoxLayout(self.fine_tuning_content)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(16)
        inner.addWidget(self._build_engine_group())
        inner.addWidget(self._build_output_group())
        self.group_fixes = self._build_fixes_group()
        inner.addWidget(self.group_fixes)
        self.card_character_3d = self._build_reference_group()
        inner.addWidget(self.card_character_3d)
        self.group_prompts = self._build_prompts_group()
        inner.addWidget(self.group_prompts)

        self.fine_tuning_content.setVisible(False)
        box.addWidget(self.fine_tuning_content)
        return wrapper

    @staticmethod
    def _group(title: str) -> tuple:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        v.addWidget(_divider())
        v.addSpacing(4)
        v.addWidget(_label(title, "section"))
        return w, v

    def _build_engine_group(self) -> QWidget:
        w, v = self._group("ENGINE")
        self.btn_group_engine = QButtonGroup(self)

        self.radio_superres = QRadioButton("Upscaler (RelayVSR, Real-ESRGAN, UltraSharp)")
        self.radio_superres.setToolTip("Upscale only: fast, faithful, no ComfyUI needed for RelayVSR.")
        self.radio_superres.setChecked(True)
        self.btn_group_engine.addButton(self.radio_superres, 0)
        v.addWidget(self.radio_superres)

        self.superres_opts = QWidget()
        sr = QVBoxLayout(self.superres_opts)
        sr.setContentsMargins(22, 0, 0, 4)
        self.combo_superres_model = QComboBox()
        self.combo_superres_model.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo_superres_model.setMinimumContentsLength(10)
        self.combo_superres_model.addItem("RelayVSR 4x — generative detail", config.RELAYVSR_MODEL_ID)
        self.combo_superres_model.addItem("Real-ESRGAN x2 — clean, fast", "RealESRGAN_x2plus.pth")
        self.combo_superres_model.addItem("4x-UltraSharp — crisp texture", "4x-UltraSharp.pth")
        self.combo_superres_model.currentIndexChanged.connect(self._on_superres_model_changed)
        sr.addWidget(self.combo_superres_model)
        v.addWidget(self.superres_opts)

        self.radio_diffusion = QRadioButton("LTX-2.5 generative redraw")
        self.radio_diffusion.setToolTip("Diffusion redraw guided by prompts. Slow; needs ComfyUI.")
        self.btn_group_engine.addButton(self.radio_diffusion, 1)
        v.addWidget(self.radio_diffusion)

        self.diffusion_opts = QWidget()
        d = QVBoxLayout(self.diffusion_opts)
        d.setContentsMargins(22, 0, 0, 4)
        d.setSpacing(4)
        self.lbl_strength_val = _label("", "faint")
        self.slider_strength = QSlider(Qt.Horizontal)
        self.slider_strength.setRange(40, 95)  # 0.40 to 0.95
        self.slider_strength.setValue(54)
        self.slider_strength.valueChanged.connect(self._on_strength_slider_changed)
        self.lbl_cfg_val = _label("", "faint")
        self.slider_cfg = QSlider(Qt.Horizontal)
        self.slider_cfg.setRange(10, 50)  # 1.0 to 5.0
        self.slider_cfg.setValue(30)
        self.slider_cfg.valueChanged.connect(self._on_cfg_slider_changed)
        for wdg in (self.lbl_strength_val, self.slider_strength, self.lbl_cfg_val, self.slider_cfg):
            d.addWidget(wdg)
        self.diffusion_opts.setVisible(False)
        v.addWidget(self.diffusion_opts)

        self.radio_sol = QRadioButton("NVIDIA SoL-Refiner (1-step)")
        self.radio_sol.setToolTip("One-pass 1080p detail refinement. Needs ComfyUI.")
        self.btn_group_engine.addButton(self.radio_sol, 2)
        v.addWidget(self.radio_sol)

        self.sol_opts = QWidget()
        s = QVBoxLayout(self.sol_opts)
        s.setContentsMargins(22, 0, 0, 4)
        s.setSpacing(4)
        self.lbl_sol_sigma_val = _label("", "faint")
        self.slider_sol_sigma = QSlider(Qt.Orientation.Horizontal)
        self.slider_sol_sigma.setRange(30, 70)  # 0.30 to 0.70
        self.slider_sol_sigma.setValue(50)
        self.slider_sol_sigma.valueChanged.connect(self._on_sol_sigma_changed)
        s.addWidget(self.lbl_sol_sigma_val)
        s.addWidget(self.slider_sol_sigma)
        self.sol_opts.setVisible(False)
        v.addWidget(self.sol_opts)

        if not self.comfyui_available:
            # The LTX, SoL-Refiner and Real-ESRGAN / UltraSharp engines run in ComfyUI
            self.radio_diffusion.setVisible(False)
            self.radio_sol.setVisible(False)
            for i in reversed(range(self.combo_superres_model.count())):
                if self.combo_superres_model.itemData(i) != config.RELAYVSR_MODEL_ID:
                    self.combo_superres_model.removeItem(i)

        self.btn_group_engine.idToggled.connect(self._on_engine_toggled)
        self._on_strength_slider_changed(self.slider_strength.value())
        self._on_cfg_slider_changed(self.slider_cfg.value())
        self._on_sol_sigma_changed(self.slider_sol_sigma.value())
        return w

    def _build_output_group(self) -> QWidget:
        w, v = self._group("OUTPUT")
        v.addWidget(_label("Size", "faint"))
        self.combo_res = QComboBox()
        self.combo_res.addItems(["1080p (Full HD)", "4K UHD", "8K UHD", "720p (HD)"])
        self.combo_res.currentIndexChanged.connect(self._on_res_changed)
        v.addWidget(self.combo_res)
        self.lbl_res_desc = _label("", "faint", wrap=True)
        v.addWidget(self.lbl_res_desc)

        v.addSpacing(4)
        v.addWidget(_label("Format", "faint"))
        self.combo_codec = QComboBox()
        self.combo_codec.addItem("Auto — best for this PC", "auto")
        self.combo_codec.addItem("HEVC — about half the size", "hevc")
        self.combo_codec.addItem("H.264 — plays everywhere", "h264")
        self.combo_codec.setToolTip("Audio is always AAC. HEVC is encoded on the GPU's video encoder when there is one.")
        self.combo_codec.currentIndexChanged.connect(self._update_codec_label)
        v.addWidget(self.combo_codec)
        self.lbl_codec = _label("Checking video encoders…", "faint", wrap=True)
        v.addWidget(self.lbl_codec)
        # Probing encoders runs ffmpeg a few times, so it happens on a plain thread that touches no
        # Qt objects; the UI polls for the cached result instead of being called back from it
        threading.Thread(target=video_encoders.best_hevc, daemon=True).start()
        self._codec_poll = QTimer(self)
        self._codec_poll.setInterval(250)
        self._codec_poll.timeout.connect(self._update_codec_label)
        self._codec_poll.start()
        return w

    def _update_codec_label(self, *_):
        if not video_encoders.probed():
            return
        self._codec_poll.stop()
        self.lbl_codec.setText(f"Output: {video_encoders.describe(self.combo_codec.currentData())} · AAC audio")

    def _build_fixes_group(self) -> QWidget:
        w, v = self._group("FIXES")
        row = QHBoxLayout()
        self.lbl_auto_status = _label("Detected automatically when a video loads.", "faint", wrap=True)
        row.addWidget(self.lbl_auto_status, stretch=1)
        self.btn_autodetect = QPushButton("Detect")
        self.btn_autodetect.setObjectName("ghost")
        self.btn_autodetect.setToolTip("Scan frames for interlacing, color, noise and damage")
        self.btn_autodetect.clicked.connect(
            lambda: self.current_video_path and self.rescan_requested.emit(self.current_video_path))
        row.addWidget(self.btn_autodetect, alignment=Qt.AlignTop)
        v.addLayout(row)

        def tick(text, tip, checked):
            chk = QCheckBox(text)
            chk.setToolTip(tip)
            chk.setChecked(checked)
            chk.stateChanged.connect(self._on_tick_changed)
            v.addWidget(chk)
            return chk

        self.chk_dirt = tick("Dirt, scratches && tears", "Removes film specks, gate hair, dust and tears.", True)
        self.chk_jitter = tick("Gate jitter && flicker", "Smooths gate weave and exposure breathing.", True)
        self.chk_deint = tick("De-interlace", "Merges interlaced fields into progressive frames.", False)
        self.chk_noise = tick("Noise && compression", "Removes macroblocks, tape hiss, chroma smear and grain.", True)
        self.chk_colorize = tick("Colorize black && white", "Infers period colors and skin tones.", False)
        self.chk_film = tick("Keep film character", "Preserves real film texture instead of CGI smoothing.", True)
        self.chk_micro = tick("2-pass micro-detail", "Adds fine skin, fabric and texture detail in a second pass.", False)
        return w

    def _build_reference_group(self) -> QWidget:
        w, v = self._group("REFERENCE")
        row = QHBoxLayout()
        self.btn_select_ref = QPushButton("Choose portrait…")
        self.btn_select_ref.setToolTip("Guide faces with a reference portrait")
        self.btn_select_ref.clicked.connect(self._on_select_reference_clicked)
        row.addWidget(self.btn_select_ref)
        self.btn_clear_ref = QPushButton("Clear")
        self.btn_clear_ref.setObjectName("ghost")
        self.btn_clear_ref.setEnabled(False)
        self.btn_clear_ref.clicked.connect(self._on_clear_reference_clicked)
        row.addWidget(self.btn_clear_ref)
        row.addStretch()
        v.addLayout(row)
        self.lbl_ref_path = ElidedLabel("No portrait — faces are generated freely", elide=Qt.ElideMiddle)
        self.lbl_ref_path.setObjectName("faint")
        v.addWidget(self.lbl_ref_path)
        self.chk_depth = QCheckBox("Also export a depth map")
        self.chk_depth.setToolTip("Depth Anything V2 depth video alongside the result")
        v.addWidget(self.chk_depth)
        return w

    def _build_prompts_group(self) -> QWidget:
        w, v = self._group("PROMPTS & SAMPLING")
        row = QHBoxLayout()
        self.lbl_engine_badge = ElidedLabel("")
        self.lbl_engine_badge.setObjectName("faint")
        row.addWidget(self.lbl_engine_badge, stretch=1)
        self.btn_reset_prompts = QPushButton("Reset")
        self.btn_reset_prompts.setObjectName("ghost")
        self.btn_reset_prompts.setToolTip("Rebuild prompts from the fixes above")
        self.btn_reset_prompts.clicked.connect(self._reset_prompts_to_ticks)
        row.addWidget(self.btn_reset_prompts)
        v.addLayout(row)

        self.txt_pos = QTextEdit()
        self.txt_pos.setPlaceholderText("Positive prompt (built from the fixes)")
        self.txt_pos.setFixedHeight(64)
        self.txt_pos.textChanged.connect(self._on_prompt_manually_edited)
        v.addWidget(self.txt_pos)
        self.txt_neg = QTextEdit()
        self.txt_neg.setPlaceholderText("Negative prompt")
        self.txt_neg.setFixedHeight(52)
        self.txt_neg.textChanged.connect(self._on_prompt_manually_edited)
        v.addWidget(self.txt_neg)

        grid = QHBoxLayout()
        col1 = QVBoxLayout()
        col1.addWidget(_label("Tile", "faint"))
        self.combo_tile = QComboBox()
        self.combo_tile.addItems(["FullHD (1024×576)", "HD (960×544)", "qHD (640×360)"])
        col1.addWidget(self.combo_tile)
        grid.addLayout(col1)
        col2 = QVBoxLayout()
        col2.addWidget(_label("Seed (-1 random)", "faint"))
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(-1, 2147483647)
        self.spin_seed.setValue(-1)
        col2.addWidget(self.spin_seed)
        grid.addLayout(col2)
        v.addLayout(grid)

        v.addWidget(_label("Clip length", "faint"))
        self.combo_frames = QComboBox()
        self.combo_frames.addItem("Auto (offer a preview for long clips)", None)
        self.combo_frames.addItem("65 frames (~2.5 s)", 65)
        self.combo_frames.addItem("97 frames (~4 s)", 97)
        self.combo_frames.addItem("129 frames (~5.4 s)", 129)
        self.combo_frames.addItem("Whole video", -1)
        v.addWidget(self.combo_frames)

        self.chk_teacache = QCheckBox("TeaCache (about 2x faster)")
        self.chk_teacache.setToolTip("Reuses similar timestep activations across DiT blocks.")
        self.chk_teacache.stateChanged.connect(self._on_teacache_toggled)
        v.addWidget(self.chk_teacache)
        self.lbl_teacache_thresh = _label("", "faint")
        v.addWidget(self.lbl_teacache_thresh)
        self.slider_teacache_thresh = QSlider(Qt.Horizontal)
        self.slider_teacache_thresh.setRange(2, 15)  # 0.02 to 0.15
        self.slider_teacache_thresh.setValue(6)
        self.slider_teacache_thresh.valueChanged.connect(self._on_teacache_thresh_changed)
        v.addWidget(self.slider_teacache_thresh)
        self._update_teacache_thresh_label(0.06)
        return w

    def _build_action_area(self) -> QWidget:
        area = QWidget()
        v = QVBoxLayout(area)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        self.tracking_container = QWidget()
        t = QVBoxLayout(self.tracking_container)
        t.setContentsMargins(0, 0, 0, 4)
        t.setSpacing(5)
        self.lbl_status = _label("", wrap=True)
        t.addWidget(self.lbl_status)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        t.addWidget(self.progress_bar)
        self.lbl_metrics = _label("", "faint")
        t.addWidget(self.lbl_metrics)
        self.tracking_container.setVisible(False)
        v.addWidget(self.tracking_container)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_start = QPushButton("Start")
        self.btn_start.setObjectName("primary")
        self.btn_start.setMinimumHeight(38)
        self.btn_start.setToolTip("Choose where to save, then restore the loaded video")
        self.btn_start.clicked.connect(self._on_start_clicked)
        self.btn_start.setEnabled(False)
        row.addWidget(self.btn_start, stretch=3)

        self.btn_add_to_queue = QPushButton("Add to queue")
        self.btn_add_to_queue.setMinimumHeight(38)
        self.btn_add_to_queue.setToolTip("Queue the loaded video with these settings and range")
        self.btn_add_to_queue.clicked.connect(self._on_add_to_queue_clicked)
        self.btn_add_to_queue.setEnabled(False)
        row.addWidget(self.btn_add_to_queue, stretch=2)
        v.addLayout(row)
        return area

    # -----------------------------------------------------------------
    # Presets & engine
    # -----------------------------------------------------------------
    def _toggle_fine_tuning(self):
        show = not self.fine_tuning_content.isVisible()
        self.fine_tuning_content.setVisible(show)
        self.btn_toggle_fine_tuning.setText(("▾" if show else "▸") + "  Customize")

    def current_engine(self) -> str:
        if self.radio_superres.isChecked():
            # RelayVSR lives in the upscaler model list but runs on its own engine path
            is_relay = self.combo_superres_model.currentData() == config.RELAYVSR_MODEL_ID
            return "relayvsr" if is_relay else "superres"
        if self.radio_sol.isChecked():
            return "sol_refiner"
        return "diffusion"

    def _sync_engine_sections(self):
        """Shows only the options the selected engine actually uses."""
        engine = self.current_engine()
        self.engine_changed.emit(engine)
        self.superres_opts.setVisible(engine in ("superres", "relayvsr"))
        self.diffusion_opts.setVisible(engine == "diffusion")
        self.sol_opts.setVisible(engine == "sol_refiner")
        self.group_fixes.setVisible(engine != "relayvsr")
        self.card_character_3d.setVisible(engine in ("diffusion", "sol_refiner"))
        self.group_prompts.setVisible(engine in ("diffusion", "sol_refiner"))
        self.lbl_res_desc.setText({
            "relayvsr": "RelayVSR renders 4x and is resampled to this size. 720p recommended.",
            "superres": "Fast CUDA upscale, no temporal boiling.",
            "sol_refiner": "One-step generative refine at 1080p.",
        }.get(engine, "4K and 8K use a two-stage redraw (clean, then refine)."))

    def _on_preset_dropdown_changed(self, index: int):
        preset_id = self.combo_presets.currentData()
        if preset_id == "custom" or not preset_id:
            self._show_custom_summary()
            return
        self._apply_preset_settings(presets.get_preset(preset_id))

    def _show_custom_summary(self):
        self.lbl_preset_desc.setText("Your own settings — adjust them under Customize.")
        self.lbl_preset_meta.setText("")

    def _apply_preset_settings(self, p: dict):
        self._is_applying_preset = True
        self._syncing_ticks = True
        try:
            engine_type = p.get("engine_type", "diffusion")
            if engine_type in ("superres", "relayvsr"):
                self.radio_superres.setChecked(True)
                model_fn = config.RELAYVSR_MODEL_ID if engine_type == "relayvsr" else p.get("superres_model", "RealESRGAN_x2plus.pth")
                idx = self.combo_superres_model.findData(model_fn)
                if idx >= 0:
                    self.combo_superres_model.setCurrentIndex(idx)
            elif engine_type == "sol_refiner":
                self.radio_sol.setChecked(True)
                self.slider_sol_sigma.setValue(int(p.get("sol_sigma", 0.50) * 100))
            else:
                self.radio_diffusion.setChecked(True)
                self.slider_strength.setValue(int(p.get("guide_strength", 0.72) * 100))
                self.slider_cfg.setValue(int(p.get("cfg", 2.5) * 10))

            out_res = p.get("output_size", "FullHD")
            self.combo_res.setCurrentIndex({"4K": 1, "8K": 2, "HD": 3}.get(out_res, 0))
            self.combo_tile.setCurrentIndex({"HD": 1, "qHD": 2}.get(p.get("tile_size", "FullHD"), 0))

            defects = p.get("defects", {})
            self.chk_dirt.setChecked(defects.get("dirt_scratches", True))
            self.chk_jitter.setChecked(defects.get("jitter_flicker", True))
            self.chk_deint.setChecked(defects.get("deinterlace", False))
            self.chk_noise.setChecked(defects.get("noise_compression", True))
            self.chk_colorize.setChecked(defects.get("colorize", False))
            self.chk_film.setChecked(defects.get("film_roll", True))
            self.chk_micro.setChecked(defects.get("micro_detail", False))

            self.txt_pos.setPlainText(p.get("positive", ""))
            self.txt_neg.setPlainText(p.get("negative", ""))

            self.chk_teacache.setChecked(p.get("enable_teacache", False))
            tc_th = p.get("teacache_thresh", 0.06)
            self.slider_teacache_thresh.setValue(int(round(tc_th * 100)))
            self._update_teacache_thresh_label(tc_th)

            self.lbl_preset_desc.setText(p.get("description", ""))
            self.lbl_preset_meta.setText(p.get("badge", ""))
            self.lbl_preset_meta.setToolTip(p.get("est_speed", ""))
            self._update_engine_badge(p.get("mode"))
            self._sync_engine_sections()
        finally:
            self._syncing_ticks = False
            self._is_applying_preset = False

    def _update_engine_badge(self, mode: Optional[str] = None):
        if mode == "two_stage":
            self.lbl_engine_badge.setText("Two-stage: clean, then refine")
        elif self.chk_teacache.isChecked():
            self.lbl_engine_badge.setText("Single pass, 8 steps + TeaCache")
        else:
            self.lbl_engine_badge.setText("Single pass, 8 steps")

    def _mark_custom_settings(self):
        if self._syncing_ticks or self._is_applying_preset:
            return
        self.combo_presets.blockSignals(True)
        idx = self.combo_presets.findData("custom")
        if idx >= 0:
            self.combo_presets.setCurrentIndex(idx)
        self.combo_presets.blockSignals(False)
        self._show_custom_summary()

    def _on_engine_toggled(self, id_val: int, checked: bool):
        if not checked:
            return
        self._mark_custom_settings()
        self._sync_engine_sections()
        if id_val == 1 and not self._syncing_ticks:
            self._update_pipeline_from_ticks()

    def _on_superres_model_changed(self, _index: int):
        self._mark_custom_settings()
        self._sync_engine_sections()

    # -----------------------------------------------------------------
    # Video & range
    # -----------------------------------------------------------------
    def begin_video(self, path: str):
        """A video is being opened: its settings apply from now on (a batch queued right away uses them)."""
        self.current_video_path = path
        self.lbl_auto_status.setText("Scanning frames…")

    def clear_video(self):
        """The video couldn't be opened."""
        self.current_video_path = None
        self.lbl_auto_status.setText("Detected automatically when a video loads.")

    def set_video_metadata(self, meta: Dict[str, Any]):
        self.current_video_path = meta["path"]
        self.lbl_file_name.setText(Path(meta["path"]).name)
        self.total_frames = meta.get("total_frames", 0)
        self.fps = meta.get("fps", 24.0) or 24.0
        duration = self._format_duration(self.total_frames / self.fps) if self.fps else "–"
        parts = [f"{meta['width']}×{meta['height']}", f"{meta['fps']:.2f} fps", duration,
                 "audio" if meta.get("has_audio") else "no audio"]
        self.lbl_file_meta.setText("  ·  ".join(parts))
        self.current_in_point = None
        self.current_out_point = None
        self._update_range_display()

        self.btn_start.setEnabled(True)
        self.btn_add_to_queue.setEnabled(True)

    def on_in_out_changed(self, in_f: Optional[int], out_f: Optional[int]):
        self.current_in_point = in_f
        self.current_out_point = out_f
        self._update_range_display()

    def _update_range_display(self):
        if self.current_in_point is None and self.current_out_point is None:
            self.lbl_range.setVisible(False)
            return
        in_val = self.current_in_point if self.current_in_point is not None else 0
        out_val = self.current_out_point if self.current_out_point is not None else max(0, self.total_frames - 1)
        tc = lambda f: self._timecode(f / self.fps if self.fps > 0 else 0)
        self.lbl_range.setText(f"Range {tc(in_val)} → {tc(out_val)}  ·  {max(1, out_val - in_val + 1)} frames")
        self.lbl_range.setVisible(True)

    @staticmethod
    def _timecode(sec: float) -> str:
        return f"{int(sec // 60):02d}:{int(sec % 60):02d}.{int((sec % 1) * 10)}"

    def _on_browse_clicked(self):
        start_dir = str(user_settings.get_last_input_folder())
        files, _ = QFileDialog.getOpenFileNames(
            self, "Open Videos — pick several to queue them all", start_dir,
            "Video Files (*.mp4 *.mov *.avi *.mkv *.webm)"
        )
        if files:
            user_settings.set_last_input_folder(Path(files[0]).parent)
            self.videos_selected.emit(files)

    def _on_select_reference_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Choose Reference Portrait", str(config.INPUTS_DIR), "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if file_path:
            self.selected_reference_image = file_path
            self.lbl_ref_path.setText(Path(file_path).name)
            self.btn_clear_ref.setEnabled(True)

    def _on_clear_reference_clicked(self):
        self.selected_reference_image = None
        self.lbl_ref_path.setText("No portrait — faces are generated freely")
        self.btn_clear_ref.setEnabled(False)

    # -----------------------------------------------------------------
    # Fixes (defects), resolution & prompts
    # -----------------------------------------------------------------
    def show_scanning(self):
        self.lbl_auto_status.setText("Scanning frames…")

    def apply_detection(self, res: dict):
        """Sets the fix ticks from a defect scan (the caller only passes the current video's)."""
        if "error" in res or not res:
            self.lbl_auto_status.setText("Couldn't scan this video — set the fixes by hand.")
            return
        self.lbl_auto_status.setText(f"Detected: {res.get('summary_text', 'analysis complete')}")

        suggested = res.get("suggested_defects", {})
        self._syncing_ticks = True
        try:
            self.chk_dirt.setChecked(suggested.get("dirt_scratches", True))
            self.chk_jitter.setChecked(suggested.get("jitter_flicker", True))
            self.chk_deint.setChecked(suggested.get("deinterlace", False))
            self.chk_noise.setChecked(suggested.get("noise_compression", True))
            self.chk_colorize.setChecked(suggested.get("colorize", False))
            self.chk_film.setChecked(suggested.get("film_roll", True))
            self.chk_micro.setChecked(suggested.get("micro_detail", False))
        finally:
            self._syncing_ticks = False

        self._custom_prompt_edited = False
        self._update_pipeline_from_ticks()

    def _on_tick_changed(self):
        if self._syncing_ticks:
            return
        self._custom_prompt_edited = False
        self._mark_custom_settings()
        self._update_pipeline_from_ticks()

    def _on_res_changed(self, index: int):
        if self._syncing_ticks:
            return
        self._mark_custom_settings()
        self._syncing_ticks = True
        try:
            self.chk_micro.setChecked(self.get_selected_resolution() in ("4K", "8K"))
        finally:
            self._syncing_ticks = False
        self._update_pipeline_from_ticks()

    def _on_prompt_manually_edited(self):
        if not self._syncing_ticks:
            self._custom_prompt_edited = True
            self._mark_custom_settings()

    def _reset_prompts_to_ticks(self):
        self._custom_prompt_edited = False
        self._update_pipeline_from_ticks()

    def get_defect_ticks(self) -> Dict[str, bool]:
        return {
            "dirt_scratches": self.chk_dirt.isChecked(),
            "jitter_flicker": self.chk_jitter.isChecked(),
            "deinterlace": self.chk_deint.isChecked(),
            "noise_compression": self.chk_noise.isChecked(),
            "colorize": self.chk_colorize.isChecked(),
            "film_roll": self.chk_film.isChecked(),
            "micro_detail": self.chk_micro.isChecked(),
        }

    def get_selected_resolution(self) -> str:
        return {1: "4K", 2: "8K", 3: "HD"}.get(self.combo_res.currentIndex(), "FullHD")

    def _update_pipeline_from_ticks(self):
        target_res = self.get_selected_resolution()
        pipeline = presets.build_pipeline_from_defects(self.get_defect_ticks(), target_res)
        self._update_engine_badge(pipeline["mode"])
        if pipeline["mode"] == "two_stage":
            self.combo_tile.setCurrentIndex(2 if target_res == "FullHD" else 1)  # qHD / HD stage 1
        else:
            self.combo_tile.setCurrentIndex(0)  # FullHD

        # Update prompts only if user hasn't typed custom overrides
        if not self._custom_prompt_edited:
            self._syncing_ticks = True
            try:
                self.txt_pos.setPlainText(pipeline["positive_prompt"])
                self.txt_neg.setPlainText(pipeline["negative_prompt"])
            finally:
                self._syncing_ticks = False

    def _on_strength_slider_changed(self, value: int):
        self._mark_custom_settings()
        val_f = value / 100.0
        desc = ("faithful, little new detail" if val_f >= 0.75 else
                "subtle texture" if val_f >= 0.62 else
                "crisp new detail" if val_f >= 0.50 else
                "free redraw, may change structure")
        self.lbl_strength_val.setText(f"Redraw freedom {val_f:.2f} — {desc}")

    def _on_cfg_slider_changed(self, value: int):
        self._mark_custom_settings()
        val_f = value / 10.0
        desc = ("prompt off" if val_f <= 1.0 else "subtle" if val_f <= 2.0 else
                "active" if val_f <= 3.0 else "heavy")
        self.lbl_cfg_val.setText(f"Prompt influence {val_f:.1f} — {desc}")

    def _on_sol_sigma_changed(self, value: int):
        self._mark_custom_settings()
        sigma = value / 100.0
        tag = "subtle" if sigma <= 0.40 else "balanced" if sigma <= 0.55 else "rich detail"
        self.lbl_sol_sigma_val.setText(f"Detail strength {sigma:.2f} — {tag}")

    def _on_teacache_toggled(self, state: int):
        self._mark_custom_settings()
        self._update_engine_badge()

    def _on_teacache_thresh_changed(self, value: int):
        self._mark_custom_settings()
        self._update_teacache_thresh_label(value / 100.0)

    def _update_teacache_thresh_label(self, thresh: float):
        desc = "1.5x, lossless" if thresh <= 0.04 else "2x, recommended" if thresh <= 0.07 else "2.5x, slightly softer"
        self.lbl_teacache_thresh.setText(f"Cache threshold {thresh:.2f} — {desc}")

    # -----------------------------------------------------------------
    # Parameters & actions
    # -----------------------------------------------------------------
    def get_current_restoration_params(self) -> Optional[Dict[str, Any]]:
        if not self.current_video_path:
            return None

        engine_type = self.current_engine()
        is_superres = (engine_type == "superres")
        selected_superres = self.combo_superres_model.currentData() or "RealESRGAN_x2plus.pth"

        ticks = self.get_defect_ticks()
        target_res = self.get_selected_resolution()
        pipeline = presets.build_pipeline_from_defects(ticks, target_res)

        preset_id = self.combo_presets.currentData() or presets.DEFAULT_PRESET
        preset_info = presets.RESTORATION_PRESETS.get(preset_id, {})
        mode = preset_info.get("mode") or pipeline["mode"]

        sel_limit = self.combo_frames.currentData()
        max_f = None if (sel_limit is None or sel_limit == -1) else int(sel_limit)

        return {
            "input_path": self.current_video_path,
            "engine_type": engine_type,
            "superres_model": selected_superres,
            "guide_strength": self.slider_strength.value() / 100.0,
            "cfg": self.slider_cfg.value() / 10.0,
            "sol_sigma": self.slider_sol_sigma.value() / 100.0,
            "enable_teacache": self.chk_teacache.isChecked(),
            "teacache_thresh": self.slider_teacache_thresh.value() / 100.0,
            "reference_image": self.selected_reference_image,
            "extract_depth": self.chk_depth.isChecked(),
            "mode": mode,
            "preset": preset_id,
            "preset_name": plain_name(preset_info.get("name", "Custom settings")),
            "positive_prompt": self.txt_pos.toPlainText() or pipeline["positive_prompt"],
            "negative_prompt": self.txt_neg.toPlainText() or pipeline["negative_prompt"],
            "output_size": target_res,
            "output_codec": self.combo_codec.currentData(),
            "tile_size": self.combo_tile.currentText().split(" ")[0],
            "seed": None if self.spin_seed.value() == -1 else self.spin_seed.value(),
            "deinterlace": self.chk_deint.isChecked(),
            "clean_noise": self.chk_noise.isChecked() or is_superres,
            "stabilize_jitter": self.chk_jitter.isChecked() or is_superres,
            "max_frames": max_f,
            "start_frame": self.current_in_point,
            "end_frame": self.current_out_point
        }

    def _on_start_clicked(self):
        params = self.get_current_restoration_params()
        if params:
            self.start_restore_clicked.emit(params)

    def _on_add_to_queue_clicked(self):
        params = self.get_current_restoration_params()
        if params:
            self.add_to_queue_clicked.emit(params)

    # -----------------------------------------------------------------
    # Progress
    # -----------------------------------------------------------------
    @staticmethod
    def _format_duration(seconds: float) -> str:
        s = max(0, int(round(seconds)))
        hours, mins, secs = s // 3600, (s % 3600) // 60, s % 60
        return f"{hours}:{mins:02d}:{secs:02d}" if hours else f"{mins:02d}:{secs:02d}"

    def _refresh_metrics(self):
        if self._start_time is None:
            return
        parts = [f"{int(self._current_progress)}%", f"elapsed {self._format_duration(time.time() - self._start_time)}"]
        # Engines that report their own time left (RelayVSR) are not second-guessed here
        if self._smoothed_eta and " left" not in self.lbl_status.text():
            parts.append(f"~{self._format_duration(self._smoothed_eta)} left")
        self.lbl_metrics.setText("  ·  ".join(parts))

    def start_tracking(self):
        """Starts the elapsed timer and resets the ETA estimator."""
        self._start_time = time.time()
        self._smoothed_eta = None
        self._current_progress = 0.0
        self._base_progress: Optional[float] = None  # resumed jobs start part-way
        self._is_active = True
        set_role(self.progress_bar, "")
        self.progress_bar.setValue(0)
        self.tracking_container.setVisible(True)
        self._refresh_metrics()
        self._timer.start()

    def _on_timer_tick(self):
        """Periodic 1-second tick to update elapsed time and smooth ETA countdown."""
        if not self._is_active or self._start_time is None:
            return
        if self._smoothed_eta:
            self._smoothed_eta = max(1.0, self._smoothed_eta - 1.0)
        self._refresh_metrics()

    def set_progress(self, current: int, total: int, message: str):
        """Updates progress bar, status text, and recalculates ETA."""
        if not self._is_active and current < total:
            self.start_tracking()
        self.progress_bar.setMaximum(max(1, total))
        self.progress_bar.setValue(current)
        pct = (current / max(1, total)) * 100.0
        self._current_progress = pct
        self.lbl_status.setText(message)

        if self._is_active and self._start_time is not None and pct < 98.0:
            # ETA from the rate since tracking started, measured from the first progress seen
            # (a resumed job starts part-way, so absolute progress / elapsed would be wrong)
            if self._base_progress is None:
                self._base_progress = pct
            done = pct - self._base_progress
            elapsed = time.time() - self._start_time
            if done >= 2.0 and elapsed > 5:
                raw_eta = elapsed / done * (100.0 - pct)
                self._smoothed_eta = raw_eta if self._smoothed_eta is None else 0.65 * self._smoothed_eta + 0.35 * raw_eta
        self._refresh_metrics()

        if current >= total and total > 0:
            self.finish_tracking(success=True)

    def finish_tracking(self, success: bool = True):
        """Stops the timer and shows the finished or stopped state."""
        self._is_active = False
        self._timer.stop()
        if self._start_time is None:
            return
        dur = self._format_duration(time.time() - self._start_time)
        set_role(self.progress_bar, "done" if success else "stopped")
        self.lbl_metrics.setText(f"Finished in {dur}" if success else f"Stopped after {dur}")

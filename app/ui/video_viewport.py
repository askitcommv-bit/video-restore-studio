"""
Video viewport: drop-zone empty state, frame-accurate playback, In/Out range marking,
and Original / Restored / Split / Side-by-side comparison once a result is loaded.
Zoom with the wheel, pan by dragging.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QPushButton, QSlider, QLabel, QStackedWidget,
    QFrame, QFileDialog, QButtonGroup, QStyle, QStyleOptionSlider
)

import config

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


class RangeTimelineSlider(QSlider):
    """
    Precision timeline scrubber with In/Out range highlighting,
    click-to-seek, and visual bracket markers.
    """

    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.in_point: Optional[int] = None
        self.out_point: Optional[int] = None

    def set_in_point(self, frame: Optional[int]):
        self.in_point = frame
        self.update()

    def set_out_point(self, frame: Optional[int]):
        self.out_point = frame
        self.update()

    def clear_in_out(self):
        self.in_point = None
        self.out_point = None
        self.update()

    def _pixel_to_value(self, x: float) -> int:
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        groove_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderGroove, self
        )
        handle_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderHandle, self
        )
        half_h = handle_rect.width() / 2.0
        usable_left = groove_rect.left() + half_h
        usable_right = groove_rect.right() - half_h
        usable_w = max(1.0, usable_right - usable_left)
        ratio = max(0.0, min(1.0, (x - usable_left) / usable_w))
        return int(round(self.minimum() + ratio * (self.maximum() - self.minimum())))

    def _value_to_pixel(self, val: int) -> float:
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        groove_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderGroove, self
        )
        handle_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderHandle, self
        )
        half_h = handle_rect.width() / 2.0
        usable_left = groove_rect.left() + half_h
        usable_right = groove_rect.right() - half_h
        usable_w = max(1.0, usable_right - usable_left)
        total_span = max(1, self.maximum() - self.minimum())
        ratio = max(0.0, min(1.0, (val - self.minimum()) / float(total_span)))
        return usable_left + ratio * usable_w

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos_x = event.position().x() if hasattr(event, "position") else event.x()
            val = self._pixel_to_value(pos_x)
            self.setValue(val)
            self.sliderMoved.emit(val)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            pos_x = event.position().x() if hasattr(event, "position") else event.x()
            val = self._pixel_to_value(pos_x)
            self.setValue(val)
            self.sliderMoved.emit(val)
        super().mouseMoveEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)

        if self.maximum() <= self.minimum():
            return

        if self.in_point is None and self.out_point is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        groove_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderGroove, self
        )

        track_top = groove_rect.top() - 2
        track_h = groove_rect.height() + 4

        start_val = self.in_point if self.in_point is not None else self.minimum()
        end_val = self.out_point if self.out_point is not None else self.maximum()

        if start_val > end_val:
            start_val, end_val = end_val, start_val

        px_start = self._value_to_pixel(start_val)
        px_end = self._value_to_pixel(end_val)
        w_range = max(3.0, px_end - px_start)

        # Highlight region with translucent electric cyan fill
        painter.fillRect(int(px_start), track_top, int(w_range), track_h, QColor(91, 140, 255, 60))

        # In Point marker: Cyan bracket [
        if self.in_point is not None:
            x_in = int(self._value_to_pixel(self.in_point))
            pen_in = QPen(QColor(91, 140, 255), 2)
            painter.setPen(pen_in)
            painter.drawLine(x_in, track_top - 3, x_in, track_top + track_h + 3)
            painter.drawLine(x_in, track_top - 3, x_in + 4, track_top - 3)
            painter.drawLine(x_in, track_top + track_h + 3, x_in + 4, track_top + track_h + 3)

        # Out Point marker: Emerald bracket ]
        if self.out_point is not None:
            x_out = int(self._value_to_pixel(self.out_point))
            pen_out = QPen(QColor(91, 140, 255), 2)
            painter.setPen(pen_out)
            painter.drawLine(x_out, track_top - 3, x_out, track_top + track_h + 3)
            painter.drawLine(x_out - 4, track_top - 3, x_out, track_top - 3)
            painter.drawLine(x_out - 4, track_top + track_h + 3, x_out, track_top + track_h + 3)

        painter.end()


class VideoCanvas(QGraphicsView):
    """Zoomable, pannable graphics viewport for video frames."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("background-color: #08090B; border: none; border-radius: 8px;")

        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self._frame_a: Optional[np.ndarray] = None  # Original
        self._frame_b: Optional[np.ndarray] = None  # Restored
        self.view_mode: str = "split"               # "restored", "split", "sbs", "original"
        self.split_percent: float = 0.5            # 0.0 to 1.0

    def set_frames(self, frame_a: Optional[np.ndarray], frame_b: Optional[np.ndarray] = None):
        self._frame_a = frame_a
        self._frame_b = frame_b
        self.update_display()

    def set_view_mode(self, mode: str):
        self.view_mode = mode
        self.update_display()

    def set_split_percent(self, percent: float):
        self.split_percent = max(0.0, min(1.0, percent))
        self.update_display()

    def update_display(self):
        if self._frame_a is None and self._frame_b is None:
            return

        display_bgr = None

        if self.view_mode == "original":
            if self._frame_a is not None:
                display_bgr = self._frame_a.copy()
            elif self._frame_b is not None:
                display_bgr = self._frame_b.copy()

        elif self.view_mode == "restored":
            if self._frame_b is not None:
                display_bgr = self._frame_b.copy()
            elif self._frame_a is not None:
                display_bgr = self._frame_a.copy()

        elif self.view_mode == "sbs":
            # Side-by-side mode
            if self._frame_a is not None and self._frame_b is not None:
                h_b, w_b = self._frame_b.shape[:2]
                h_a, w_a = self._frame_a.shape[:2]
                if h_a != h_b:
                    target_w_a = int(w_a * (h_b / float(h_a)))
                    frame_a_matched = cv2.resize(self._frame_a, (target_w_a, h_b), interpolation=cv2.INTER_LANCZOS4)
                else:
                    frame_a_matched = self._frame_a

                display_bgr = np.hstack([frame_a_matched, self._frame_b])
                # Badges
                cv2.rectangle(display_bgr, (15, 15), (150, 48), (15, 20, 28), -1)
                cv2.putText(display_bgr, "ORIGINAL", (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.rectangle(display_bgr, (frame_a_matched.shape[1] + 15, 15), (frame_a_matched.shape[1] + 165, 48), (15, 20, 28), -1)
                cv2.putText(display_bgr, "RESTORED", (frame_a_matched.shape[1] + 25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            elif self._frame_b is not None:
                display_bgr = self._frame_b.copy()
            else:
                display_bgr = self._frame_a.copy()

        else:
            # Default: Split Wipe mode
            if self._frame_a is not None and self._frame_b is not None:
                h_b, w_b = self._frame_b.shape[:2]
                h_a, w_a = self._frame_a.shape[:2]
                if (h_a, w_a) != (h_b, w_b):
                    frame_a_matched = cv2.resize(self._frame_a, (w_b, h_b), interpolation=cv2.INTER_LANCZOS4)
                else:
                    frame_a_matched = self._frame_a

                display_bgr = frame_a_matched.copy()
                split_x = int(w_b * self.split_percent)

                # Right side shows restored
                display_bgr[:, split_x:] = self._frame_b[:, split_x:]

                # Divider line
                cv2.line(display_bgr, (split_x, 0), (split_x, h_b), (255, 255, 255), 1)

                # Translucent badges
                cv2.rectangle(display_bgr, (15, 15), (150, 46), (15, 20, 28), -1)
                cv2.putText(display_bgr, "ORIGINAL", (25, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.rectangle(display_bgr, (w_b - 165, 15), (w_b - 15, 46), (15, 20, 28), -1)
                cv2.putText(display_bgr, "RESTORED", (w_b - 155, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            elif self._frame_b is not None:
                display_bgr = self._frame_b.copy()
            elif self._frame_a is not None:
                display_bgr = self._frame_a.copy()

        if display_bgr is None:
            return

        h, w, ch = display_bgr.shape
        bytes_per_line = ch * w
        rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        first_time = self.pixmap_item.pixmap().isNull()
        self.pixmap_item.setPixmap(pix)
        self.scene.setSceneRect(0, 0, w, h)

        if first_time:
            self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def reset_fit(self):
        """The next frame shown is fitted to the view (call when a new video is loaded)."""
        self.pixmap_item.setPixmap(QPixmap())

    def wheelEvent(self, event):
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(zoom_factor, zoom_factor)

    def fit_to_view(self):
        if not self.scene.sceneRect().isEmpty():
            self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)


class VideoPlayerWidget(QWidget):
    """Video playback with source/result comparison, range marking and an empty drop state."""

    frame_changed = Signal(int)
    videos_selected = Signal(list)  # browse multi-select / multi-file drop: first is loaded, all are queued
    in_out_changed = Signal(object, object)  # (in_frame_idx, out_frame_idx)
    compare_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.cap_a: Optional[cv2.VideoCapture] = None
        self.cap_b: Optional[cv2.VideoCapture] = None
        self.total_frames: int = 0
        self.fps: float = 24.0
        self.current_frame_idx: int = 0
        self.is_playing: bool = False
        self.in_point: Optional[int] = None
        self.out_point: Optional[int] = None

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self._on_timer_tick)

        self._init_ui()

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedWidget(self)
        self.stack.addWidget(self._build_empty_page())
        self.stack.addWidget(self._build_player_page())
        root_layout.addWidget(self.stack)

    def _build_empty_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(24, 24, 24, 24)
        drop = QFrame()
        drop.setObjectName("dropZone")
        drop.setStyleSheet("QFrame#dropZone { border: 1px dashed #353D4A; border-radius: 10px; }")
        v = QVBoxLayout(drop)
        v.setAlignment(Qt.AlignCenter)
        v.setSpacing(10)
        title = QLabel("Drop videos here")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        v.addWidget(title)
        sub = QLabel("MP4, MOV, MKV, AVI or WEBM — drop several to queue them all")
        sub.setObjectName("muted")
        sub.setAlignment(Qt.AlignCenter)
        v.addWidget(sub)
        btn = QPushButton("Open videos…")
        btn.setObjectName("primary")
        btn.clicked.connect(self._on_browse_clicked)
        v.addSpacing(6)
        v.addWidget(btn, alignment=Qt.AlignCenter)
        outer.addWidget(drop)
        return page

    def _build_player_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Comparison bar: only shown once there is a result to compare against
        self.compare_bar = QWidget()
        top = QHBoxLayout(self.compare_bar)
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(2)
        self.btn_mode_orig = self._segment("Original")
        self.btn_mode_restored = self._segment("Restored")
        self.btn_mode_split = self._segment("Split")
        self.btn_mode_sbs = self._segment("Side by side")
        self.view_btn_group = QButtonGroup(self)
        for i, b in enumerate((self.btn_mode_orig, self.btn_mode_restored, self.btn_mode_split, self.btn_mode_sbs)):
            self.view_btn_group.addButton(b, i)
            top.addWidget(b)
        self.btn_mode_split.setChecked(True)
        self.view_btn_group.buttonToggled.connect(self._on_view_mode_changed)

        top.addSpacing(10)
        self.split_slider = QSlider(Qt.Horizontal)
        self.split_slider.setRange(0, 100)
        self.split_slider.setValue(50)
        self.split_slider.setFixedWidth(110)
        self.split_slider.setToolTip("Divider position")
        self.split_slider.setFocusPolicy(Qt.NoFocus)
        self.split_slider.valueChanged.connect(lambda v: self.canvas.set_split_percent(v / 100.0))
        top.addWidget(self.split_slider)
        top.addStretch()
        self.btn_export_compare = QPushButton("Export comparison")
        self.btn_export_compare.setObjectName("ghost")
        self.btn_export_compare.setToolTip("Render side-by-side and split comparison videos")
        self.btn_export_compare.clicked.connect(self.compare_requested.emit)
        top.addWidget(self.btn_export_compare)
        self.compare_bar.setVisible(False)
        layout.addWidget(self.compare_bar)

        self.canvas = VideoCanvas(self)
        layout.addWidget(self.canvas, stretch=1)
        # Keys go to the window's player shortcuts, not to scrolling the canvas or slider
        self.canvas.setFocusPolicy(Qt.NoFocus)

        self.timeline_slider = RangeTimelineSlider(Qt.Horizontal, self)
        self.timeline_slider.setRange(0, 0)
        self.timeline_slider.sliderMoved.connect(self.seek_frame)
        self.timeline_slider.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self.timeline_slider)

        bar = QHBoxLayout()
        bar.setSpacing(4)
        self.btn_play = self._icon_button("▶", "Play / pause (Space)", self.toggle_play, role="icon")
        self.btn_play.setFixedWidth(44)
        bar.addWidget(self.btn_play)
        bar.addWidget(self._icon_button("⏴", "Back 1 frame (←, Shift+← for 10)",
                                        lambda: self.seek_frame(self.current_frame_idx - 1), role="icon"))
        bar.addWidget(self._icon_button("⏵", "Forward 1 frame (→, Shift+→ for 10)",
                                        lambda: self.seek_frame(self.current_frame_idx + 1), role="icon"))
        bar.addSpacing(8)
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setObjectName("muted")
        bar.addWidget(self.lbl_time)
        bar.addStretch()

        self.lbl_in_out = QLabel("")
        self.lbl_in_out.setObjectName("faint")
        bar.addWidget(self.lbl_in_out)
        bar.addSpacing(6)
        self.btn_mark_in = self._icon_button("In", "Start the range here (I)", lambda: self.set_in_point())
        self.btn_mark_out = self._icon_button("Out", "End the range here (O)", lambda: self.set_out_point())
        self.btn_clear_inout = self._icon_button("✕", "Clear the range — restore the whole video (X)", self.clear_in_out)
        for b in (self.btn_mark_in, self.btn_mark_out, self.btn_clear_inout):
            bar.addWidget(b)
        bar.addSpacing(6)
        bar.addWidget(self._icon_button("Fit", "Fit the frame to the view (scroll to zoom, drag to pan)",
                                        self.canvas.fit_to_view))
        layout.addLayout(bar)
        return page

    @staticmethod
    def _segment(text: str) -> QPushButton:
        b = QPushButton(text)
        b.setObjectName("segment")
        b.setCheckable(True)
        b.setFocusPolicy(Qt.NoFocus)
        return b

    @staticmethod
    def _icon_button(text: str, tip: str, slot, role: str = "ghost") -> QPushButton:
        b = QPushButton(text)
        b.setObjectName(role)
        b.setToolTip(tip)
        b.setFocusPolicy(Qt.NoFocus)  # keyboard shortcuts stay with the window
        b.clicked.connect(slot)
        return b

    # -----------------------------------------------------------------
    # Drag & Drop Support
    # -----------------------------------------------------------------
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if Path(url.toLocalFile()).suffix.lower() in VIDEO_EXTS:
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls() if Path(u.toLocalFile()).suffix.lower() in VIDEO_EXTS]
        if files:
            self.videos_selected.emit(files)
            event.acceptProposedAction()

    def _on_browse_clicked(self):
        start_dir = str(config.get_last_input_folder())
        files, _ = QFileDialog.getOpenFileNames(
            self, "Open Videos — pick several to queue them all", start_dir,
            "Video Files (*.mp4 *.mov *.avi *.mkv *.webm)"
        )
        if files:
            config.set_last_input_folder(Path(files[0]).parent)
            self.videos_selected.emit(files)

    def _on_view_mode_changed(self, *_):
        mode = {0: "original", 1: "restored", 2: "split", 3: "sbs"}.get(self.view_btn_group.checkedId(), "split")
        self.split_slider.setVisible(mode == "split")
        self.canvas.set_view_mode(mode)

    # -----------------------------------------------------------------
    # Video Loading & Playback Engine
    # -----------------------------------------------------------------
    def load_video(self, video_path: Path | str, is_restored: bool = False):
        """Loads a source (is_restored=False, drops any old result) or a result for the current source."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return
        if self.is_playing:
            self.toggle_play()
        if is_restored:
            if self.cap_b:
                self.cap_b.release()
            self.cap_b = cap
        else:
            for old in (self.cap_a, self.cap_b):
                if old:
                    old.release()
            self.cap_a, self.cap_b = cap, None
        self._on_media_changed(keep_range=is_restored)

    def load_pair(self, source_path: Optional[str], restored_path: str):
        """Shows a finished job: its own source next to its result."""
        if self.is_playing:
            self.toggle_play()
        caps = [cv2.VideoCapture(str(p)) if p and Path(p).exists() else None for p in (source_path, restored_path)]
        for old in (self.cap_a, self.cap_b):
            if old:
                old.release()
        self.cap_a, self.cap_b = (c if c is not None and c.isOpened() else None for c in caps)
        self._on_media_changed(keep_range=False)

    def _on_media_changed(self, keep_range: bool):
        self.stack.setCurrentIndex(1)
        frames = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in (self.cap_a, self.cap_b) if c]
        self.total_frames = max(frames, default=0)
        active_cap = self.cap_a or self.cap_b
        self.fps = active_cap.get(cv2.CAP_PROP_FPS) if active_cap else 24.0
        if not self.fps or self.fps <= 0 or np.isnan(self.fps):
            self.fps = 24.0
        self.timeline_slider.setRange(0, max(0, self.total_frames - 1))

        both = self.cap_a is not None and self.cap_b is not None
        self.compare_bar.setVisible(both)
        (self.btn_mode_split if both else (self.btn_mode_restored if self.cap_b else self.btn_mode_orig)).setChecked(True)
        self._on_view_mode_changed()
        self.canvas.reset_fit()

        if not keep_range:
            self.clear_in_out()
        start = min(self.current_frame_idx, max(0, self.total_frames - 1)) if keep_range else 0
        self.current_frame_idx = -2  # forces a real seek for the first frame
        self.seek_frame(start)

    def set_in_point(self, frame_idx: Optional[int] = None):
        if self.total_frames <= 0:
            return
        if frame_idx is None:
            frame_idx = self.current_frame_idx
        frame_idx = max(0, min(self.total_frames - 1, frame_idx))
        if self.out_point is not None and frame_idx > self.out_point:
            self.out_point = None  # an In after the Out starts a new range running to the end
        self.in_point = frame_idx
        self._apply_range()

    def set_out_point(self, frame_idx: Optional[int] = None):
        if self.total_frames <= 0:
            return
        if frame_idx is None:
            frame_idx = self.current_frame_idx
        frame_idx = max(0, min(self.total_frames - 1, frame_idx))
        if self.in_point is not None and frame_idx < self.in_point:
            self.in_point = None
        self.out_point = frame_idx
        self._apply_range()

    def clear_in_out(self):
        self.in_point = None
        self.out_point = None
        self._apply_range()

    def _apply_range(self):
        self.timeline_slider.set_in_point(self.in_point)
        self.timeline_slider.set_out_point(self.out_point)
        self._update_in_out_ui()
        self.in_out_changed.emit(self.in_point, self.out_point)

    def jump_to_in(self):
        self.seek_frame(self.in_point if self.in_point is not None else 0)

    def jump_to_out(self):
        self.seek_frame(self.out_point if self.out_point is not None else self.total_frames - 1)

    def _timecode(self, frame: int) -> str:
        sec = frame / self.fps if self.fps > 0 else 0
        return f"{int(sec // 60):02d}:{int(sec % 60):02d}.{int((sec % 1) * 10)}"

    def _update_in_out_ui(self):
        has_range = self.in_point is not None or self.out_point is not None
        self.btn_clear_inout.setVisible(has_range)
        if not has_range or self.total_frames <= 0:
            self.lbl_in_out.setText("Whole video")
            return
        in_f = self.in_point if self.in_point is not None else 0
        out_f = self.out_point if self.out_point is not None else self.total_frames - 1
        self.lbl_in_out.setText(f"{self._timecode(in_f)} → {self._timecode(out_f)}  ·  {max(1, out_f - in_f + 1)} frames")

    def seek_frame(self, frame_idx: int):
        frame_idx = max(0, min(int(frame_idx), max(0, self.total_frames - 1)))
        # Playback reads the next frame directly: seeking every frame is very slow for
        # long-GOP codecs (HEVC/H.264), so only random access seeks
        sequential = frame_idx == self.current_frame_idx + 1
        self.current_frame_idx = frame_idx
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(frame_idx)
        self.timeline_slider.blockSignals(False)

        frames = []
        for cap in (self.cap_a, self.cap_b):
            frame = None
            if cap:
                if not sequential:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    frame = None
            frames.append(frame)

        self.canvas.set_frames(*frames)
        self._update_time_label()
        self.frame_changed.emit(frame_idx)

    def toggle_play(self):
        if self.is_playing:
            self.is_playing = False
            self.btn_play.setText("▶")
            self.timer.stop()
        elif self.total_frames > 0:
            self.is_playing = True
            self.btn_play.setText("⏸")
            self.timer.start(max(1, int(round(1000.0 / self.fps))))

    def _on_timer_tick(self):
        next_idx = self.current_frame_idx + 1
        self.seek_frame(0 if next_idx >= self.total_frames else next_idx)

    def _update_time_label(self):
        fmt = lambda sec: f"{int(sec // 60):02d}:{int(sec % 60):02d}"
        cur = self.current_frame_idx / self.fps if self.fps > 0 else 0
        tot = self.total_frames / self.fps if self.fps > 0 else 0
        self.lbl_time.setText(f"{fmt(cur)} / {fmt(tot)}")
        self.lbl_time.setToolTip(f"Frame {self.current_frame_idx + 1} of {self.total_frames} · {self.fps:.3f} fps")

    def handle_key(self, key: int, shift: bool) -> bool:
        """Window-wide player shortcuts (wired from the main window). Returns True if handled."""
        if self.stack.currentIndex() != 1:
            return False
        step = 10 if shift else 1
        actions = {
            Qt.Key_Space: self.toggle_play,
            Qt.Key_I: self.set_in_point,
            Qt.Key_O: self.set_out_point,
            Qt.Key_X: self.clear_in_out,
            Qt.Key_Home: self.jump_to_in,
            Qt.Key_End: self.jump_to_out,
            Qt.Key_Left: lambda: self.seek_frame(self.current_frame_idx - step),
            Qt.Key_Right: lambda: self.seek_frame(self.current_frame_idx + step),
        }
        action = actions.get(key)
        if action is None:
            return False
        action()
        return True

"""Small shared widgets for the desktop UI."""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (QDialog, QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QSizePolicy, QVBoxLayout)

from app.ui import styles


class ElidedLabel(QLabel):
    """Single-line label that shortens long text with '…' instead of widening its layout.
    The full text is kept as the tooltip."""

    def __init__(self, text: str = "", parent=None, elide: Qt.TextElideMode = Qt.ElideRight):
        super().__init__(parent)
        self._elide = elide
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setText(text)

    def setText(self, text: str):
        super().setText(text)
        self.setToolTip(text if text else "")
        self.update()

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setWidth(0)
        return hint

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.contentsRect()
        text = self.fontMetrics().elidedText(self.text(), self._elide, rect.width())
        painter.setPen(self.palette().color(self.foregroundRole()))
        painter.drawText(rect, self.alignment() | Qt.AlignVCenter, text)


class ProgressDialog(QDialog):
    """Small modal card that says what the app is doing (e.g. while a video loads).
    It appears only if the work takes longer than `delay_ms`, so quick loads show nothing.
    Esc hides it; the work carries on."""

    def __init__(self, parent, title: str, delay_ms: int = 250):
        super().__init__(parent, Qt.Dialog | Qt.FramelessWindowHint)
        self.setWindowModality(Qt.WindowModal)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedWidth(380)

        card = QFrame(self)
        card.setObjectName("panel")
        card.setStyleSheet(f"QFrame#panel {{ background-color: {styles.RAISED}; border: 1px solid {styles.BORDER};"
                           " border-radius: 10px; }")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(card)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(6)

        self.lbl_title = QLabel(title)
        self.lbl_title.setObjectName("heading")
        self.lbl_subject = ElidedLabel()
        self.lbl_subject.setObjectName("muted")
        self.lbl_step = QLabel("")
        self.lbl_step.setWordWrap(True)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(4)
        self.bar.setRange(0, 0)  # busy until a step reports a count
        self.lbl_count = QLabel("")
        self.lbl_count.setObjectName("faint")
        lay.addWidget(self.lbl_title)
        lay.addWidget(self.lbl_subject)
        lay.addSpacing(8)
        lay.addWidget(self.lbl_step)
        lay.addWidget(self.bar)
        lay.addWidget(self.lbl_count)

        # Error state: the message stays until the user closes it
        self.lbl_error = QLabel("")
        self.lbl_error.setWordWrap(True)
        self.lbl_error.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_error.setStyleSheet(f"color: {styles.TEXT};")
        self.btn_ok = QPushButton("OK")
        self.btn_ok.setObjectName("primary")
        self.btn_ok.clicked.connect(self.hide)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(self.btn_ok)
        lay.addWidget(self.lbl_error)
        lay.addSpacing(6)
        lay.addLayout(row)
        self._set_error_mode(False)

        self._title = title
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._show_now)
        self._delay_ms = delay_ms
        self._done = False

    def start(self, subject: str):
        self._done = False
        self._set_error_mode(False)
        self.lbl_title.setText(self._title)
        self.lbl_subject.setText(subject)
        self.set_step("Starting…")
        self._timer.start(self._delay_ms)

    def set_step(self, text: str, current: int = 0, total: int = 0):
        self.lbl_step.setText(text)
        if total > 0:
            self.bar.setRange(0, total)
            self.bar.setValue(current)
            self.lbl_count.setText(f"{current} of {total}")
        else:
            self.bar.setRange(0, 0)
            self.lbl_count.setText("")

    def fail(self, title: str, message: str):
        """Shows what went wrong right away and keeps it on screen until OK / Esc."""
        self._done = True
        self._timer.stop()
        self.lbl_title.setText(title)
        self.lbl_error.setText(message)
        self._set_error_mode(True)
        self._show_now(force=True)
        self.btn_ok.setFocus()

    def _set_error_mode(self, error: bool):
        for wdg in (self.lbl_step, self.bar, self.lbl_count):
            wdg.setVisible(not error)
        self.lbl_error.setVisible(error)
        self.btn_ok.setVisible(error)

    def finish(self):
        self._done = True
        self._timer.stop()
        self.hide()

    def _show_now(self, force: bool = False):
        if self._done and not force:
            return
        self.adjustSize()
        parent = self.parentWidget()
        if parent is not None:
            geo = parent.frameGeometry()
            self.move(geo.center() - self.rect().center())
        self.show()
        self.raise_()

    def reject(self):  # Esc: just hide; whatever is running keeps going
        self.hide()

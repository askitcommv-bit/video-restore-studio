"""Small shared widgets for the desktop UI."""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QDialog, QFrame, QLabel, QProgressBar, QSizePolicy, QVBoxLayout

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

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._show_now)
        self._delay_ms = delay_ms
        self._done = False

    def start(self, subject: str):
        self._done = False
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

    def finish(self):
        self._done = True
        self._timer.stop()
        self.hide()

    def _show_now(self):
        if self._done:
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

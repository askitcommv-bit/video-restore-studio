"""Small shared widgets for the desktop UI."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QLabel, QSizePolicy


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

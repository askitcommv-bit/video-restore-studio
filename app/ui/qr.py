"""QR codes as crisp pixmaps (drawn from the code's module grid, no image library involved)."""

import qrcode
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPixmap


def qr_pixmap(text: str, size: int, device_pixel_ratio: float = 1.0, quiet_zone: int = 2) -> QPixmap:
    """`text` as a black-on-white QR code, `size` logical pixels square (white background included,
    so phone cameras read it on the dark theme)."""
    code = qrcode.QRCode(border=quiet_zone, error_correction=qrcode.constants.ERROR_CORRECT_M)
    code.add_data(text)
    code.make(fit=True)
    matrix = code.get_matrix()  # quiet zone included

    px = max(1, round(size * device_pixel_ratio))
    pm = QPixmap(px, px)
    pm.fill(Qt.white)
    cell = px / len(matrix)
    painter = QPainter(pm)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor("#000000"))
    for y, row in enumerate(matrix):
        for x, dark in enumerate(row):
            if dark:
                # Snap to whole pixels so neighbouring modules meet without hairline gaps
                left, top = round(x * cell), round(y * cell)
                painter.drawRect(QRectF(left, top, round((x + 1) * cell) - left, round((y + 1) * cell) - top))
    painter.end()
    pm.setDevicePixelRatio(device_pixel_ratio)
    return pm

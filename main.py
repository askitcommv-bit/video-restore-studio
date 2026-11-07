import sys
import os
from pathlib import Path

# Ensure the root directory is in sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from app.ui.main_window import MainWindow

LOGO = Path(root_dir) / "app" / "ui" / "assets" / "logo.ico"


def main():
    # High DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setWindowIcon(QIcon(str(LOGO)))
    if sys.platform == "win32":
        # Own taskbar identity, so Windows shows the app's icon instead of python.exe's when run from source
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("askit.VideoRestoreStudio")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

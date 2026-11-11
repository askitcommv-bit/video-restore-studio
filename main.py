"""Desktop app entry point (also VideoRestoreStudio.exe)."""

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

import bootstrap

LOGO = Path(__file__).resolve().parent / "app" / "ui" / "assets" / "logo.ico"


def main():
    bootstrap.init()
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setWindowIcon(QIcon(str(LOGO)))
    if sys.platform == "win32":
        # Own taskbar identity, so Windows shows the app's icon instead of python.exe's when run from source
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("askit.VideoRestoreStudio")

    from app.ui.main_window import MainWindow  # after init(): the UI reads settings from the project folder
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

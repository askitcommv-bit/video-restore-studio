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


def hide_child_consoles():
    """A GUI process has no console, so Windows opens a black window for every console child
    (ffprobe, ffmpeg, nvidia-smi, git). Give them none, unless a caller asks for a console of its own."""
    import subprocess

    class _NoWindowPopen(subprocess.Popen):
        def __init__(self, *args, creationflags=0, **kwargs):
            if not creationflags & (subprocess.CREATE_NEW_CONSOLE | subprocess.DETACHED_PROCESS):
                creationflags |= subprocess.CREATE_NO_WINDOW
            super().__init__(*args, creationflags=creationflags, **kwargs)

    subprocess.Popen = _NoWindowPopen


def main():
    # High DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setWindowIcon(QIcon(str(LOGO)))
    if sys.platform == "win32":
        hide_child_consoles()
        # Own taskbar identity, so Windows shows the app's icon instead of python.exe's when run from source
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("askit.VideoRestoreStudio")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

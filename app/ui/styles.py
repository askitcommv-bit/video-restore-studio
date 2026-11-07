"""
Dark theme for Video Restore Studio.

One neutral palette and one accent. Widgets pick a role through objectName instead of
inline styles:
  QLabel   title · section · muted · faint · status-ok/warn/error/info
  QPushButton  primary · ghost · segment (checkable) · danger
  QFrame   sidebar · panel · divider
"""

# Palette (referenced from Python code where a color has to be set dynamically)
BG = "#0F1115"
SURFACE = "#161A21"
RAISED = "#1D222B"
BORDER = "#272D37"
TEXT = "#E6E8EB"
MUTED = "#8B94A1"
FAINT = "#5E6672"
ACCENT = "#5B8CFF"
ACCENT_HOVER = "#7AA2FF"
OK = "#3FB97F"
WARN = "#E0A84A"
ERROR = "#E5665F"

DARK_THEME_QSS = f"""
QMainWindow, QDialog, QMessageBox {{
    background-color: {BG};
}}

QWidget {{
    color: {TEXT};
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
}}

QToolTip {{
    background-color: {RAISED};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 4px 6px;
}}

/* ---- Surfaces ---------------------------------------------------------- */
QFrame#sidebar, QFrame#panel {{
    background-color: {SURFACE};
    border: none;
    border-radius: 10px;
}}

QFrame#divider {{
    background-color: {BORDER};
    max-height: 1px;
    min-height: 1px;
    border: none;
}}

QScrollArea, QScrollArea > QWidget > QWidget {{
    background: transparent;
    border: none;
}}

/* ---- Text roles -------------------------------------------------------- */
QLabel {{ background: transparent; }}
QLabel#title {{ font-size: 15px; font-weight: 600; }}
QLabel#section {{
    color: {FAINT};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.6px;
}}
QLabel#heading {{ font-size: 14px; font-weight: 600; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#faint {{ color: {FAINT}; font-size: 12px; }}
QLabel#status-ok {{ color: {OK}; font-weight: 600; }}
QLabel#status-warn {{ color: {WARN}; font-weight: 600; }}
QLabel#status-error {{ color: {ERROR}; font-weight: 600; }}
QLabel#status-info {{ color: {ACCENT}; font-weight: 600; }}
QLabel#status-idle {{ color: {MUTED}; font-weight: 600; }}

/* ---- Buttons ----------------------------------------------------------- */
QPushButton, QToolButton {{
    background-color: {RAISED};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 12px;
    color: {TEXT};
}}
QPushButton:hover, QToolButton:hover {{ background-color: #252B36; border-color: #353D4A; }}
QPushButton:pressed, QToolButton:pressed {{ background-color: #181C23; }}
QPushButton:disabled {{ color: {FAINT}; background-color: #171B21; border-color: #1E232B; }}
QToolButton::menu-indicator {{ image: none; width: 0; }}

QPushButton#primary {{
    background-color: {ACCENT};
    border: 1px solid {ACCENT};
    color: #FFFFFF;
    font-weight: 600;
}}
QPushButton#primary:hover {{ background-color: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QPushButton#primary:disabled {{ background-color: #23314F; border-color: #23314F; color: #7B8AA8; }}

QPushButton#ghost, QToolButton#ghost {{
    background: transparent;
    border: 1px solid transparent;
    color: {MUTED};
    padding: 4px 8px;
}}
QPushButton#ghost:hover, QToolButton#ghost:hover {{ background-color: {RAISED}; color: {TEXT}; }}
QPushButton#ghost:disabled {{ color: #3A414C; background: transparent; border-color: transparent; }}

QPushButton#icon, QToolButton#icon {{
    background: transparent;
    border: 1px solid transparent;
    color: {TEXT};
    font-size: 16px;
    padding: 2px 8px;
}}
QPushButton#icon:hover, QToolButton#icon:hover {{ background-color: {RAISED}; }}

QPushButton#danger {{ background: transparent; border: 1px solid transparent; color: {MUTED}; padding: 4px 8px; }}
QPushButton#danger:hover {{ background-color: #2A1C1D; color: {ERROR}; }}

QPushButton#segment {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    color: {MUTED};
    padding: 4px 10px;
}}
QPushButton#segment:hover {{ color: {TEXT}; }}
QPushButton#segment:checked {{ background-color: {RAISED}; border-color: {BORDER}; color: {TEXT}; }}

QPushButton#disclosure {{
    background: transparent;
    border: none;
    color: {MUTED};
    text-align: left;
    padding: 4px 0;
    font-weight: 600;
}}
QPushButton#disclosure:hover {{ color: {TEXT}; }}

/* ---- Inputs ------------------------------------------------------------ */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QComboBox {{
    background-color: {RAISED};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT};
}}
QComboBox:hover {{ border-color: #353D4A; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{ image: url(__CHEVRON_DOWN__); width: 12px; height: 12px; margin-right: 8px; }}
QComboBox QAbstractItemView {{
    background-color: {RAISED};
    border: 1px solid {BORDER};
    selection-background-color: #2A3342;
    selection-color: {TEXT};
    outline: none;
    padding: 4px;
}}

QCheckBox, QRadioButton {{ spacing: 8px; background: transparent; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 14px; height: 14px;
    border: 1px solid #3A424F;
    background-color: {RAISED};
}}
QCheckBox::indicator {{ border-radius: 4px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:checked {{ background-color: {ACCENT}; border-color: {ACCENT}; }}
QRadioButton::indicator:checked {{
    border-color: {ACCENT};
    background-color: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
        stop:0 {ACCENT}, stop:0.5 {ACCENT}, stop:0.62 {RAISED}, stop:1 {RAISED});
}}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {ACCENT}; }}

/* ---- Sliders & progress ------------------------------------------------ */
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {TEXT};
    width: 12px;
    margin: -4px 0;
    border-radius: 6px;
}}
QSlider::handle:horizontal:hover {{ background: #FFFFFF; }}

QProgressBar {{
    background-color: {BORDER};
    border: none;
    border-radius: 2px;
    max-height: 4px;
    min-height: 4px;
}}
QProgressBar::chunk {{ background-color: {ACCENT}; border-radius: 2px; }}
QProgressBar#done::chunk {{ background-color: {OK}; }}
QProgressBar#stopped::chunk {{ background-color: {FAINT}; }}

/* ---- Chrome ------------------------------------------------------------ */
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:horizontal {{ width: 8px; }}
QSplitter::handle:vertical {{ height: 8px; }}

QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #2C333E; border-radius: 3px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: #3A4350; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}

QMenu {{
    background-color: {RAISED};
    border: 1px solid {BORDER};
    padding: 4px;
}}
QMenu::item {{ padding: 6px 18px; border-radius: 4px; }}
QMenu::item:selected {{ background-color: #2A3342; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 6px; }}

QStatusBar {{ background: {BG}; color: {MUTED}; }}
QStatusBar::item {{ border: none; }}
"""

# Qt stylesheets need absolute, forward-slash paths for sub-control images
_ASSETS_DIR = __import__("pathlib").Path(__file__).resolve().parent / "assets"
DARK_THEME_QSS = DARK_THEME_QSS.replace("__CHEVRON_DOWN__", (_ASSETS_DIR / "chevron_down.svg").as_posix())


def set_role(widget, role: str) -> None:
    """Switches a widget's style role (objectName) and re-applies the stylesheet."""
    if widget.objectName() != role:
        widget.setObjectName(role)
        widget.style().unpolish(widget)
        widget.style().polish(widget)

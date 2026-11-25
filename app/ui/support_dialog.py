"""
Supporting the app: the one-time welcome card (first launch) and the support window the header's ♥
opens (support link and supporters wall). The support window is not modal, so the player, the queue
and a running restore carry on while it is open.
"""

from typing import Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout,
                               QWidget)

import config
from app.core.supporters import Shoutout, SupportersFeed, Wall
from app.ui import styles

APP_NAME = "Video Restore Studio"
SUPPORT_BUTTON_TEXT = f"♥  Support {APP_NAME}"


def open_support_page() -> None:
    QDesktopServices.openUrl(QUrl(config.SUPPORT_URL))


def _label(text: str, role: str = "", wrap: bool = True) -> QLabel:
    """Plain text always: names and messages come from the shout-outs API."""
    lbl = QLabel(text)
    lbl.setTextFormat(Qt.PlainText)
    if role:
        lbl.setObjectName(role)
    lbl.setWordWrap(wrap)
    return lbl


def _support_link() -> QLabel:
    """The support page's address, clickable (opens the default browser)."""
    shown = config.SUPPORT_URL.split("://", 1)[-1]
    lbl = QLabel(f'<a href="{config.SUPPORT_URL}" style="color: {styles.HEART}; text-decoration: none;">{shown}</a>')
    lbl.setOpenExternalLinks(True)
    lbl.setToolTip("Opens in your browser")
    return lbl


def _support_button(on_click=None) -> QPushButton:
    btn = QPushButton(SUPPORT_BUTTON_TEXT)
    btn.setObjectName("primary")
    btn.clicked.connect(on_click or open_support_page)
    return btn


class WelcomeSupportDialog(QDialog):
    """Shown once, on the first launch. However it is closed (button, Esc, ✕), `finished` fires and the
    window remembers it was seen."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Welcome to {APP_NAME}")
        self.setMinimumWidth(460)
        self.setMaximumWidth(520)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 18)
        lay.setSpacing(10)
        lay.addWidget(_label(f"Welcome to {APP_NAME}", "heading"))
        lay.addWidget(_label(f"{APP_NAME} is free, and everything runs on your own PC: your videos never "
                             "leave it, and there is no account or subscription."))
        lay.addWidget(_label("If it brings your old footage back to life and you'd like to help its "
                             "development, you can support it on askit.fun.", "muted"))
        lay.addWidget(_support_link())
        lay.addWidget(_label("This is the only time we'll ask. The ♥ at the top right opens the support page "
                             "and the supporters wall whenever you like.", "faint"))
        lay.addSpacing(6)

        row = QHBoxLayout()
        row.addStretch()
        btn_continue = QPushButton("Continue")
        btn_continue.clicked.connect(self.accept)
        row.addWidget(btn_continue)
        row.addWidget(_support_button(self._on_support))
        lay.addLayout(row)
        btn_continue.setFocus()  # Enter just continues; supporting is a deliberate click

    def _on_support(self):
        open_support_page()
        self.accept()


class ShoutoutCard(QFrame):
    def __init__(self, item: Shoutout, parent=None):
        super().__init__(parent)
        self.setObjectName("shoutout")
        self.setStyleSheet(f"QFrame#shoutout {{ background-color: {styles.RAISED}; border: 1px solid {styles.BORDER};"
                           " border-radius: 8px; }")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 9, 12, 9)
        lay.setSpacing(3)

        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(_label(item.name, "heading", wrap=False))
        badge = _label(item.badge, wrap=False)
        badge.setStyleSheet(f"color: {styles.HEART}; font-size: 11px; font-weight: 600;")
        top.addWidget(badge)
        if item.amount_text:
            top.addWidget(_label(item.amount_text, "faint", wrap=False))
        top.addStretch()
        top.addWidget(_label(item.date, "faint", wrap=False))
        lay.addLayout(top)

        if item.message:
            lay.addWidget(_label(f"“{item.message}”", "muted"))


class SupportDialog(QDialog):
    """The ♥ window: the support page and the supporters wall (cached, refreshed each time it opens)."""

    def __init__(self, feed: SupportersFeed, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Support {APP_NAME}")
        self.setModal(False)
        self.resize(560, 600)
        self.setMinimumSize(460, 420)
        self.feed = feed

        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 18)
        lay.setSpacing(10)
        lay.addWidget(_label(f"Support {APP_NAME}", "heading"))
        lay.addWidget(_label(f"{APP_NAME} is free and runs entirely on your own PC. Supporters keep it "
                             "that way and help it improve.", "muted"))
        link_row = QHBoxLayout()
        link_row.addWidget(_support_link())
        link_row.addStretch()
        link_row.addWidget(_support_button())
        lay.addLayout(link_row)

        divider = QFrame()
        divider.setObjectName("divider")
        lay.addSpacing(4)
        lay.addWidget(divider)
        lay.addSpacing(2)

        head = QHBoxLayout()
        head.addWidget(_label("SUPPORTERS WALL", "section", wrap=False))
        head.addStretch()
        self.lbl_count = _label("", "faint", wrap=False)
        head.addWidget(self.lbl_count)
        lay.addLayout(head)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        lay.addWidget(self.scroll, stretch=1)

        row = QHBoxLayout()
        row.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        row.addWidget(btn_close)
        lay.addLayout(row)

        feed.updated.connect(self._show_wall)
        feed.loading_changed.connect(self._on_loading_changed)
        self._show_wall(feed.wall)

    def showEvent(self, event):
        super().showEvent(event)
        self.feed.refresh()

    def _on_loading_changed(self, _loading: bool):
        if not (self.feed.wall and self.feed.wall.shoutouts):  # only the empty state says "Loading…"
            self._show_wall(self.feed.wall)

    def _show_wall(self, wall: Optional[Wall]):
        body = QWidget()
        col = QVBoxLayout(body)
        col.setContentsMargins(0, 0, 6, 0)
        col.setSpacing(8)
        if wall and wall.shoutouts:
            self.lbl_count.setText(f"{wall.total} supporter" + ("" if wall.total == 1 else "s"))
            for item in wall.shoutouts:
                col.addWidget(ShoutoutCard(item))
        else:
            self.lbl_count.setText("")
            col.addSpacing(40)
            if self.feed.loading:
                col.addWidget(self._centered("Loading shout-outs…", "faint"))
            else:
                col.addWidget(self._centered("No shout-outs to show yet.", "muted"))
                col.addWidget(self._centered(f"Be among the first to support {APP_NAME}!", "faint"))
        col.addStretch()
        self.scroll.setWidget(body)  # replaces (and deletes) the previous list

    @staticmethod
    def _centered(text: str, role: str) -> QLabel:
        lbl = _label(text, role)
        lbl.setAlignment(Qt.AlignHCenter)
        return lbl

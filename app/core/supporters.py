"""
The supporters wall: shout-outs from people who support the app on askit.fun.

The list comes from the shout-outs API (config.SHOUTOUTS_URL; response format in the askit.fun
website's api/README_SHOUTOUTS.md), is cached on disk, and is shown from the cache when offline.
Nothing here raises: an unreachable or not-yet-live API leaves the last list (or none).
"""

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import requests
from PySide6.QtCore import QObject, Signal

import config
from version import __version__

REQUEST_TIMEOUT = (5, 10)  # connect, read (seconds)


@dataclass(frozen=True)
class Shoutout:
    name: str
    message: str
    badge: str  # "Supporter" / "Monthly Backer"
    amount: float
    currency: str
    date: str  # YYYY-MM-DD

    @property
    def amount_text(self) -> str:
        """'25 EUR', '7.50 USD', or '' when there is no amount."""
        if self.amount <= 0:
            return ""
        value = f"{self.amount:.0f}" if self.amount == int(self.amount) else f"{self.amount:.2f}"
        return f"{value} {self.currency}".strip()


@dataclass(frozen=True)
class Wall:
    shoutouts: Tuple[Shoutout, ...]
    total: int  # all published shout-outs (the response may hold only the newest page)


def _shoutout(item: dict) -> Optional[Shoutout]:
    if not isinstance(item, dict):
        return None
    name = str(item.get("name") or "").strip()
    if not name:
        return None
    try:
        amount = float(item.get("amount") or 0)
    except (TypeError, ValueError):
        amount = 0.0
    recurring = item.get("frequency") == "recurring"
    return Shoutout(
        name=name,
        message=str(item.get("message") or "").strip(),
        badge=str(item.get("frequency_label") or ("Monthly Backer" if recurring else "Supporter")),
        amount=amount,
        currency=str(item.get("currency") or ""),
        date=str(item.get("date") or ""),
    )


def parse(payload) -> Optional[Wall]:
    """The wall from a shout-outs response; None unless it is a successful one."""
    if not isinstance(payload, dict) or payload.get("success") is not True:
        return None
    items = payload.get("shoutouts")
    if not isinstance(items, list):
        return None
    shoutouts = tuple(s for s in map(_shoutout, items) if s is not None)
    try:
        total = int(payload.get("total"))
    except (TypeError, ValueError):
        total = len(shoutouts)
    return Wall(shoutouts, max(total, len(shoutouts)))


def fetch(url: str = config.SHOUTOUTS_URL, api_key: str = config.SHOUTOUTS_API_KEY) -> Optional[dict]:
    """The API's response if it is a successful one; None offline, on any HTTP error (404 until the
    website is live, 401 when a key is needed) or on anything that isn't the expected JSON."""
    headers = {"Accept": "application/json", "User-Agent": f"VideoRestoreStudio/{__version__}"}
    if api_key:
        headers["X-API-Key"] = api_key
    try:
        r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return None
        payload = r.json()
    except (requests.RequestException, ValueError):
        return None
    return payload if parse(payload) is not None else None


def load_cache(path: Path = config.SUPPORTERS_CACHE_PATH) -> Optional[Wall]:
    try:
        return parse(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


def save_cache(payload: dict, path: Path = config.SUPPORTERS_CACHE_PATH) -> None:
    try:
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass  # the wall still shows this time; it just isn't there offline next time


class SupportersFeed(QObject):
    """The current wall for the UI: the cached one at once, refreshed from the API off the UI thread."""

    updated = Signal(object)  # Wall
    loading_changed = Signal(bool)
    _fetched = Signal(object)  # the response or None, emitted from the fetch thread

    def __init__(self, cache_path: Path = config.SUPPORTERS_CACHE_PATH, parent=None):
        super().__init__(parent)
        self.cache_path = cache_path
        self.wall: Optional[Wall] = load_cache(cache_path)
        self.loading = False
        self._fetched.connect(self._on_fetched)

    def refresh(self) -> None:
        """Fetches the wall in the background; `updated` follows when one arrived (offline: nothing)."""
        if self.loading:
            return
        self._set_loading(True)
        threading.Thread(target=lambda: self._fetched.emit(fetch()), daemon=True).start()

    def _on_fetched(self, payload: Optional[dict]) -> None:
        wall = parse(payload)
        if wall is not None:
            save_cache(payload, self.cache_path)
            self.wall = wall
            self.updated.emit(wall)
        self._set_loading(False)

    def _set_loading(self, on: bool) -> None:
        self.loading = on
        self.loading_changed.emit(on)

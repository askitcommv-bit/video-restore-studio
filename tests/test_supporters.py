"""Supporters wall and the one-time support card. Run from the project root:
python -m unittest discover tests
"""

import json
import os
import re
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import requests  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

import user_settings  # noqa: E402
from app.core import supporters  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

PAYLOAD = {
    "success": True, "total": 124, "count": 3, "page": 1, "limit": 50,
    "shoutouts": [
        {"id": 42, "name": "CinemaLover", "message": "My wedding tape looks new!", "frequency": "recurring",
         "frequency_label": "Monthly Backer", "currency": "EUR", "amount": 25.0, "date": "2026-09-08",
         "created_at": "2026-09-08T12:30:00+02:00"},
        {"id": 41, "name": "<b>Fan</b>", "message": "", "frequency": "single", "frequency_label": "Supporter",
         "currency": "USD", "amount": 7.5, "date": "2026-09-07", "created_at": "2026-09-07T19:40:00+02:00"},
        {"id": 40, "name": "  ", "message": "nameless entries are skipped"},
    ],
}


def _response(status: int, body) -> mock.Mock:
    r = mock.Mock(status_code=status)
    if isinstance(body, Exception):
        r.json.side_effect = body
    else:
        r.json.return_value = body
    return r


class ParseTests(unittest.TestCase):
    def test_success_response(self):
        wall = supporters.parse(PAYLOAD)
        self.assertEqual(wall.total, 124)
        self.assertEqual([s.name for s in wall.shoutouts], ["CinemaLover", "<b>Fan</b>"])
        first, second = wall.shoutouts
        self.assertEqual((first.badge, first.amount_text, first.date), ("Monthly Backer", "25 EUR", "2026-09-08"))
        self.assertEqual((second.message, second.amount_text), ("", "7.50 USD"))

    def test_rejects_anything_else(self):
        for bad in (None, [], {"success": False, "error": "Unauthorized"}, {"success": True},
                    {"success": True, "shoutouts": "nope"}):
            self.assertIsNone(supporters.parse(bad), bad)

    def test_total_falls_back_to_the_list(self):
        wall = supporters.parse({"success": True, "total": None, "shoutouts": PAYLOAD["shoutouts"][:1]})
        self.assertEqual(wall.total, 1)

    def test_missing_fields_and_bad_amount(self):
        s = supporters.parse({"success": True, "shoutouts": [{"name": "A", "amount": "x", "frequency": "recurring"}]})
        item = s.shoutouts[0]
        self.assertEqual((item.badge, item.amount_text, item.message), ("Monthly Backer", "", ""))


class FetchTests(unittest.TestCase):
    def test_no_key_sends_no_key_header(self):
        with mock.patch.object(supporters.requests, "get", return_value=_response(200, PAYLOAD)) as get:
            self.assertEqual(supporters.fetch("https://example.test/api", api_key=""), PAYLOAD)
        headers = get.call_args.kwargs["headers"]
        self.assertNotIn("X-API-Key", headers)
        self.assertEqual(headers["Accept"], "application/json")

    def test_key_is_sent_as_x_api_key(self):
        with mock.patch.object(supporters.requests, "get", return_value=_response(200, PAYLOAD)) as get:
            supporters.fetch("https://example.test/api", api_key="k123")
        self.assertEqual(get.call_args.kwargs["headers"]["X-API-Key"], "k123")

    def test_fails_quietly(self):
        cases = [
            mock.Mock(side_effect=requests.ConnectionError("offline")),
            mock.Mock(side_effect=requests.Timeout("slow")),
            mock.Mock(return_value=_response(404, {"success": False})),  # until the website is live
            mock.Mock(return_value=_response(401, {"success": False, "error": "Unauthorized"})),
            mock.Mock(return_value=_response(200, ValueError("HTML, not JSON"))),
            mock.Mock(return_value=_response(200, {"success": False})),
        ]
        for get in cases:
            with mock.patch.object(supporters.requests, "get", get):
                self.assertIsNone(supporters.fetch("https://example.test/api", api_key=""))

    def test_urls(self):
        import config
        self.assertEqual(config.SUPPORT_URL, "https://askit.fun/video-restore/support/")
        self.assertEqual(config.SHOUTOUTS_URL, "https://askit.fun/api/videorestore-shoutouts.php")


class CacheAndFeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name) / "supporters_cache.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_cache_round_trip_and_bad_cache(self):
        self.assertIsNone(supporters.load_cache(self.cache))
        supporters.save_cache(PAYLOAD, self.cache)
        self.assertEqual(supporters.load_cache(self.cache).total, 124)
        self.cache.write_text("{not json", encoding="utf-8")
        self.assertIsNone(supporters.load_cache(self.cache))

    def test_feed_caches_a_fetched_wall_and_keeps_it_offline(self):
        feed = supporters.SupportersFeed(cache_path=self.cache)
        self.assertIsNone(feed.wall)
        got = []
        feed.updated.connect(got.append)
        feed._on_fetched(PAYLOAD)
        self.assertEqual(len(got), 1)
        self.assertTrue(self.cache.exists())
        feed._on_fetched(None)  # offline later: nothing changes
        self.assertEqual(len(got), 1)
        self.assertEqual(feed.wall.total, 124)
        self.assertFalse(feed.loading)
        self.assertEqual(supporters.SupportersFeed(cache_path=self.cache).wall.total, 124)  # next launch

    def test_refresh_runs_off_the_ui_thread(self):
        feed = supporters.SupportersFeed(cache_path=self.cache)
        with mock.patch.object(supporters, "fetch", return_value=PAYLOAD):
            feed.refresh()
            self.assertTrue(feed.loading)
            feed.refresh()  # a second click while loading starts nothing
            for _ in range(200):
                self.app.processEvents()
                if not feed.loading:
                    break
                time.sleep(0.01)
        self.assertFalse(feed.loading)
        self.assertEqual(feed.wall.total, 124)


class WelcomeFlagTests(unittest.TestCase):
    def test_seen_once_is_remembered(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(user_settings, "CONFIG_JSON_PATH", Path(tmp) / "config.json"):
                self.assertFalse(user_settings.support_welcome_seen())
                user_settings.mark_support_welcome_seen()
                self.assertTrue(user_settings.support_welcome_seen())
                saved = json.loads((Path(tmp) / "config.json").read_text(encoding="utf-8"))
                self.assertIs(saved["support_welcome_seen"], True)


class DialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name) / "supporters_cache.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _texts(self, widget):
        return [lbl.text() for lbl in widget.findChildren(QLabel)]

    def test_welcome_card_dismissed_any_way_reports_finished(self):
        from app.ui.support_dialog import WelcomeSupportDialog
        for close in ("accept", "reject"):
            dlg = WelcomeSupportDialog()
            done = []
            dlg.finished.connect(done.append)
            dlg.open()
            getattr(dlg, close)()
            self.assertEqual(len(done), 1, close)
        text = " ".join(self._texts(WelcomeSupportDialog()))
        self.assertIn("free", text)
        self.assertIn("own PC", text)

    def test_support_button_opens_the_support_page(self):
        from app.ui import support_dialog
        dlg = support_dialog.WelcomeSupportDialog()
        with mock.patch.object(support_dialog.QDesktopServices, "openUrl") as open_url:
            btn = next(b for b in dlg.findChildren(support_dialog.QPushButton)
                       if b.text() == support_dialog.SUPPORT_BUTTON_TEXT)
            btn.click()
        self.assertEqual(open_url.call_args.args[0].toString(), "https://askit.fun/video-restore/support/")
        self.assertEqual(btn.text(), "♥  Support Video Restore Studio")

    def test_support_window_shows_the_wall_and_empty_state(self):
        from app.ui.support_dialog import SupportDialog
        feed = supporters.SupportersFeed(cache_path=self.cache)
        with mock.patch.object(supporters, "fetch", return_value=None):
            dlg = SupportDialog(feed)
            self.assertFalse(dlg.isModal())
            self.assertIn("No shout-outs to show yet.", self._texts(dlg))
            feed._on_fetched(PAYLOAD)
            texts = self._texts(dlg)
            self.assertIn("124 supporters", texts)
            self.assertIn("CinemaLover", texts)
            self.assertIn("<b>Fan</b>", texts)  # shown literally, never as markup
            self.assertIn("“My wedding tape looks new!”", texts)


class QrTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_qr_pixmap_is_square_black_on_white(self):
        from app.ui.qr import qr_pixmap
        pm = qr_pixmap("https://askit.fun/video-restore/support/", 150, device_pixel_ratio=2.0)
        img = pm.toImage()
        self.assertEqual((img.width(), img.height()), (300, 300))
        self.assertEqual(pm.deviceIndependentSize().width(), 150)
        self.assertEqual(img.pixelColor(0, 0).name(), "#ffffff")  # quiet zone
        colors = {img.pixelColor(x, y).name() for x in range(0, 300, 3) for y in range(0, 300, 3)}
        self.assertEqual(colors, {"#ffffff", "#000000"})

    def test_both_dialogs_show_the_qr_code(self):
        from app.ui.support_dialog import SupportDialog, WelcomeSupportDialog
        with tempfile.TemporaryDirectory() as tmp:
            feed = supporters.SupportersFeed(cache_path=Path(tmp) / "c.json")
            for dlg in (WelcomeSupportDialog(), SupportDialog(feed)):
                qr = [lbl for lbl in dlg.findChildren(QLabel)
                      if lbl.accessibleName() == "QR code for https://askit.fun/video-restore/support/"]
                self.assertEqual(len(qr), 1, type(dlg).__name__)
                self.assertFalse(qr[0].pixmap().isNull())


class WordingTests(unittest.TestCase):
    def test_no_donate_wording(self):
        """Stripe requires 'support' wording: nothing the app ships may say donate / donation."""
        sources = [ROOT / "app" / "ui" / "support_dialog.py", ROOT / "app" / "ui" / "header_bar.py",
                   ROOT / "app" / "core" / "supporters.py", ROOT / "app" / "ui" / "main_window.py",
                   ROOT / "config.py", ROOT / "user_settings.py"]
        for path in sources:
            self.assertIsNone(re.search(r"donat", path.read_text(encoding="utf-8"), re.I), path)


if __name__ == "__main__":
    unittest.main()

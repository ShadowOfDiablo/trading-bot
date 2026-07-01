"""
Offline test suite — no T212 API, no yfinance, no real models needed.

Usage:
    python tests.py
"""

import sys
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ohlcv(closes, seed=42):
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    n = len(closes)
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "open":   closes,
        "high":   [c * (1 + rng.uniform(0, 0.005)) for c in closes],
        "low":    [c * (1 - rng.uniform(0, 0.005)) for c in closes],
        "close":  closes,
        "volume": rng.integers(500_000, 2_000_000, n).astype(float),
    }, index=pd.date_range("2024-01-01", periods=n, freq="h"))
    return df


# ── features.py ───────────────────────────────────────────────────────────────

from features import build_features, _ema, _rsi, _atr
from t212_client import T212Client
from bot import Bot


class TestEma(unittest.TestCase):
    def test_flat(self):
        s = pd.Series([10.0] * 60)
        self.assertAlmostEqual(_ema(s, 20).iloc[-1], 10.0, places=4)

    def test_fast_above_slow_on_uptrend(self):
        s = pd.Series(range(1, 101), dtype=float)
        self.assertGreater(_ema(s, 5).iloc[-1], _ema(s, 20).iloc[-1])


class TestRsi(unittest.TestCase):
    def test_overbought(self):
        s = pd.Series(range(1, 80), dtype=float)
        self.assertGreater(_rsi(s, 14).iloc[-1], 70)

    def test_oversold(self):
        s = pd.Series(range(80, 1, -1), dtype=float)
        self.assertLess(_rsi(s, 14).iloc[-1], 30)

    def test_bounds(self):
        s = pd.Series([100 + (i % 5) * 2 for i in range(80)], dtype=float)
        r = _rsi(s, 14).dropna()
        self.assertTrue((r >= 0).all() and (r <= 100).all())


class TestBuildFeatures(unittest.TestCase):
    def _df(self, n=300):
        closes = [100 + i * 0.1 for i in range(n)]
        return _ohlcv(closes)

    def test_returns_dataframe(self):
        feat = build_features(self._df())
        self.assertIsInstance(feat, pd.DataFrame)
        self.assertFalse(feat.empty)

    def test_no_nans(self):
        feat = build_features(self._df())
        self.assertFalse(feat.isnull().any().any())

    def test_expected_columns(self):
        feat = build_features(self._df())
        for col in ("close_to_ema20", "rsi_14", "atr_14", "volume_ratio",
                    "mom_4h", "bb_pos"):
            self.assertIn(col, feat.columns)

    def test_needs_enough_rows(self):
        # Too few rows → all NaN → empty result
        feat = build_features(_ohlcv([100.0] * 10))
        self.assertTrue(feat.empty)


# ── model.py ──────────────────────────────────────────────────────────────────

from model import train, predict


class TestModel(unittest.TestCase):
    def _df(self, n=600):
        rng = np.random.default_rng(0)
        closes = 100 + np.cumsum(rng.normal(0, 0.5, n))
        return _ohlcv(closes.tolist())

    def test_train_returns_metrics(self):
        import tempfile, os
        df = self._df()
        with tempfile.TemporaryDirectory() as tmp:
            with patch("model.MODELS_DIR", tmp):
                metrics = train("TEST", df)
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0)
        self.assertLessEqual(metrics["accuracy"], 1)

    def test_predict_signal_is_binary(self):
        import tempfile
        df = self._df()
        with tempfile.TemporaryDirectory() as tmp:
            with patch("model.MODELS_DIR", tmp):
                train("TEST", df)
                from model import load
                md = load("TEST")
                with patch("model.MODELS_DIR", tmp):
                    sig, conf = predict(md, df)
        self.assertIn(sig, (0, 1))
        self.assertGreaterEqual(conf, 0)
        self.assertLessEqual(conf, 1)

    def test_predict_empty_df_returns_flat(self):
        md = {"model": MagicMock(), "feature_cols": ["close_to_ema20"]}
        sig, conf = predict(md, _ohlcv([100.0] * 5))
        self.assertEqual(sig, 0)
        self.assertEqual(conf, 0.0)


# ── strategy.py ───────────────────────────────────────────────────────────────

from strategy import Signal, compute_signal


class TestStrategy(unittest.TestCase):
    def _df(self, n=300):
        closes = [100 + i * 0.1 for i in range(n)]
        return _ohlcv(closes)

    def test_no_model_returns_flat(self):
        with patch("strategy.load", return_value=None):
            sig, ind = compute_signal(self._df(), "FAKE")
        self.assertEqual(sig, Signal.FLAT)
        self.assertIn("warning", ind)

    def test_long_signal(self):
        mock_md = {"model": MagicMock(), "feature_cols": []}
        with patch("strategy.load", return_value=mock_md), \
             patch("strategy.predict", return_value=(1, 0.72)):
            sig, ind = compute_signal(self._df(), "QQQ")
        self.assertEqual(sig, Signal.LONG)
        self.assertEqual(ind["confidence"], 0.72)

    def test_flat_signal(self):
        mock_md = {"model": MagicMock(), "feature_cols": []}
        with patch("strategy.load", return_value=mock_md), \
             patch("strategy.predict", return_value=(0, 0.41)):
            sig, _ = compute_signal(self._df(), "QQQ")
        self.assertEqual(sig, Signal.FLAT)


# ── risk_manager.py ───────────────────────────────────────────────────────────

from risk_manager import position_size, daily_loss_exceeded, open_position_count


class TestPositionSize(unittest.TestCase):
    def test_basic(self):
        # 1% of £10,000 = £100. At £50/share → 2 shares
        self.assertEqual(position_size(10_000, 50.0, 0.01), 2.0)

    def test_zero_price(self):
        self.assertEqual(position_size(10_000, 0.0, 0.01), 0.0)

    def test_fractional_result(self):
        # 0.5% of £10,000 = £50. At £480/share → 0.1 shares
        qty = position_size(10_000, 480.0, 0.005)
        self.assertGreaterEqual(qty, 0.0)


class TestKillSwitch(unittest.TestCase):
    def test_no_loss(self):
        self.assertFalse(daily_loss_exceeded(10_000, 10_000, 0, 0.03))

    def test_small_loss_no_trigger(self):
        self.assertFalse(daily_loss_exceeded(10_000, 9_900, 0, 0.03))

    def test_exceeds_threshold(self):
        self.assertTrue(daily_loss_exceeded(10_000, 9_500, 0, 0.03))

    def test_open_pnl_included(self):
        self.assertTrue(daily_loss_exceeded(10_000, 10_000, -400, 0.03))

    def test_zero_start(self):
        self.assertFalse(daily_loss_exceeded(0, 0, 0, 0.03))


class TestOpenPositionCount(unittest.TestCase):
    def test_counts_tracked_only(self):
        portfolio = [
            {"ticker": "QQQ_US_EQ"},
            {"ticker": "UNRELATED_EQ"},
        ]
        tracked = ["QQQ_US_EQ", "SPY_US_EQ", "TSLA_US_EQ", "NVDA_US_EQ"]
        self.assertEqual(open_position_count(portfolio, tracked), 1)

    def test_empty_portfolio(self):
        self.assertEqual(open_position_count([], ["QQQ_US_EQ"]), 0)

    def test_all_open(self):
        portfolio = [{"ticker": t} for t in ["QQQ_US_EQ", "NVDA_US_EQ"]]
        self.assertEqual(open_position_count(portfolio, ["QQQ_US_EQ", "NVDA_US_EQ"]), 2)


# ── notifier.py ───────────────────────────────────────────────────────────────

import notifier


class TestNotifier(unittest.TestCase):
    @patch("notifier.requests.post")
    def test_skipped_without_config(self, mock_post):
        with patch("notifier.cfg") as c:
            c.TELEGRAM_BOT_TOKEN = ""
            c.TELEGRAM_CHAT_ID   = ""
            notifier._send("test")
        mock_post.assert_not_called()

    @patch("notifier.requests.post")
    def test_called_with_token(self, mock_post):
        mock_post.return_value = MagicMock(ok=True)
        with patch("notifier.cfg") as c:
            c.TELEGRAM_BOT_TOKEN = "tok"
            c.TELEGRAM_CHAT_ID   = "123"
            notifier._send("hello")
        mock_post.assert_called_once()

    @patch("notifier._send")
    def test_alert_trade_buy(self, mock_send):
        notifier.alert_trade("BUY", "QQQ_US_EQ", 2.0, 481.0, "conf=0.72")
        msg = mock_send.call_args[0][0]
        self.assertIn("BUY", msg)
        self.assertIn("QQQ_US_EQ", msg)

    @patch("notifier._send")
    def test_kill_switch_alert(self, mock_send):
        notifier.alert_kill_switch(0.035)
        self.assertIn("Kill switch", mock_send.call_args[0][0])


# ── t212_client.py ─────────────────────────────────────────────────────────

class TestT212ClientMockMode(unittest.TestCase):
    @patch("t212_client.requests.get", side_effect=RuntimeError("network blocked"))
    def test_mock_mode_bypasses_auth_check(self, mock_get):
        with patch("t212_client.cfg") as c:
            c.USE_MOCK_T212 = True
            c.T212_API_KEY = "key"
            c.T212_API_SECRET = "secret"
            c.T212_BASE_URL = "https://demo.trading212.com/api/v0"
            client = T212Client()
        ok, status, text = client.check_auth()
        self.assertTrue(ok)
        self.assertEqual(status, 200)
        self.assertIn("mock", text.lower())
        mock_get.assert_not_called()


class TestBotStartup(unittest.TestCase):
    @patch("bot.notifier.alert_error")
    def test_does_not_kill_on_rate_limit(self, mock_alert):
        with patch("bot.T212Client") as client_cls:
            client = client_cls.return_value
            client.check_auth.return_value = (False, 429, "too many requests")
            bot = Bot()
        self.assertFalse(bot.state.killed)


# ── data_feed.py ──────────────────────────────────────────────────────────────

from data_feed import get_ohlcv


class TestDataFeed(unittest.TestCase):
    @patch("data_feed.yf.download")
    def test_returns_100_bars(self, mock_dl):
        n = 110
        mock_dl.return_value = pd.DataFrame({
            "Open":   np.linspace(100, 120, n),
            "High":   np.linspace(101, 121, n),
            "Low":    np.linspace(99,  119, n),
            "Close":  np.linspace(100, 120, n),
            "Volume": [1_000_000] * n,
        })
        df = get_ohlcv("QQQ", "1h", bars=100)
        self.assertEqual(len(df), 100)
        self.assertIn("close", df.columns)

    @patch("data_feed.yf.download")
    def test_empty_raises(self, mock_dl):
        mock_dl.return_value = pd.DataFrame()
        with self.assertRaises(RuntimeError):
            get_ohlcv("FAKE", "1h")


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False)
    sys.exit(0 if result.result.wasSuccessful() else 1)

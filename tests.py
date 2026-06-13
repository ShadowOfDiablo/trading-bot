"""
Tests for strategy, risk manager, and notifier.
Runs entirely offline — no T212 API or yfinance calls needed.

Usage:
    python tests.py
"""

import sys
import math
import unittest
import pandas as pd
from unittest.mock import patch, MagicMock

# ── Strategy tests ─────────────────────────────────────────────────────────────

from strategy import Signal, compute_signal, _ema, _rsi


def _make_df(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"close": closes})


class TestIndicators(unittest.TestCase):
    def test_ema_flat_series(self):
        s = pd.Series([10.0] * 50)
        result = _ema(s, 20)
        self.assertAlmostEqual(result.iloc[-1], 10.0, places=4)

    def test_ema_rising_series(self):
        s = pd.Series(range(1, 101), dtype=float)
        fast = _ema(s, 5)
        slow = _ema(s, 20)
        # Fast EMA should be closer to the end value on a rising series
        self.assertGreater(fast.iloc[-1], slow.iloc[-1])

    def test_rsi_overbought(self):
        # All gains → RSI should be near 100
        s = pd.Series([float(i) for i in range(1, 60)])
        r = _rsi(s, 14)
        self.assertGreater(r.iloc[-1], 70)

    def test_rsi_oversold(self):
        # All losses → RSI should be near 0
        s = pd.Series([float(60 - i) for i in range(60)])
        r = _rsi(s, 14)
        self.assertLess(r.iloc[-1], 30)

    def test_rsi_range(self):
        s = pd.Series([100.0 + (i % 5) * 2 for i in range(80)])
        r = _rsi(s, 14)
        self.assertTrue((r.dropna() >= 0).all())
        self.assertTrue((r.dropna() <= 100).all())


class TestSignal(unittest.TestCase):
    def _trending_up(self, n: int = 200) -> pd.DataFrame:
        # Steady uptrend — fast EMA will be above slow EMA
        closes = [100.0 + i * 0.5 for i in range(n)]
        return _make_df(closes)

    def _trending_down(self, n: int = 200) -> pd.DataFrame:
        closes = [200.0 - i * 0.5 for i in range(n)]
        return _make_df(closes)

    def test_uptrend_gives_long(self):
        df = self._trending_up()
        signal, indicators = compute_signal(df)
        self.assertEqual(signal, Signal.LONG)
        self.assertIn("ema_fast", indicators)
        self.assertIn("rsi", indicators)

    def test_downtrend_gives_flat(self):
        df = self._trending_down()
        signal, _ = compute_signal(df)
        self.assertEqual(signal, Signal.FLAT)

    def test_indicators_keys(self):
        df = self._trending_up()
        _, ind = compute_signal(df)
        for key in ("close", "ema_fast", "ema_slow", "rsi"):
            self.assertIn(key, ind)


# ── Risk manager tests ─────────────────────────────────────────────────────────

from risk_manager import position_size, daily_loss_exceeded


class TestPositionSize(unittest.TestCase):
    def test_basic(self):
        # 1% of £10,000 = £100. At price £50 → 2 shares
        qty = position_size(10_000.0, 50.0)
        self.assertGreater(qty, 0)
        self.assertEqual(qty, 2.0)

    def test_zero_price(self):
        self.assertEqual(position_size(10_000.0, 0.0), 0.0)

    def test_expensive_stock(self):
        # 1% of £1,000 = £10. At price £500 → 0 whole shares possible
        qty = position_size(1_000.0, 500.0)
        # 10/500 = 0.02, floored to 1dp = 0.0
        self.assertEqual(qty, 0.0)


class TestKillSwitch(unittest.TestCase):
    def test_no_loss(self):
        self.assertFalse(daily_loss_exceeded(10_000.0, 10_000.0, 0.0))

    def test_small_loss(self):
        # 1% loss, threshold is 3% → should NOT trigger
        self.assertFalse(daily_loss_exceeded(10_000.0, 9_900.0, 0.0))

    def test_exceeds_threshold(self):
        # 5% loss → triggers
        self.assertTrue(daily_loss_exceeded(10_000.0, 9_500.0, 0.0))

    def test_open_pnl_counts(self):
        # Cash fine but open position is down 4%
        self.assertTrue(daily_loss_exceeded(10_000.0, 10_000.0, -400.0))

    def test_zero_start_cash(self):
        self.assertFalse(daily_loss_exceeded(0.0, 0.0, 0.0))


# ── Notifier tests ─────────────────────────────────────────────────────────────

import notifier


class TestNotifier(unittest.TestCase):
    @patch("notifier.requests.post")
    def test_send_skipped_when_no_config(self, mock_post):
        with patch("notifier.cfg") as mock_cfg:
            mock_cfg.TELEGRAM_BOT_TOKEN = ""
            mock_cfg.TELEGRAM_CHAT_ID   = ""
            notifier._send("test")
        mock_post.assert_not_called()

    @patch("notifier.requests.post")
    def test_send_called_with_token(self, mock_post):
        mock_post.return_value = MagicMock(ok=True)
        with patch("notifier.cfg") as mock_cfg:
            mock_cfg.TELEGRAM_BOT_TOKEN = "fake_token"
            mock_cfg.TELEGRAM_CHAT_ID   = "123"
            notifier._send("hello")
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        self.assertIn("fake_token", call_kwargs[0][0])

    @patch("notifier._send")
    def test_alert_trade_buy(self, mock_send):
        notifier.alert_trade("BUY", "QQQ_US_EQ", 5.0, 480.0, "test")
        mock_send.assert_called_once()
        msg = mock_send.call_args[0][0]
        self.assertIn("BUY", msg)
        self.assertIn("QQQ_US_EQ", msg)

    @patch("notifier._send")
    def test_alert_kill_switch(self, mock_send):
        notifier.alert_kill_switch(0.035)
        mock_send.assert_called_once()
        msg = mock_send.call_args[0][0]
        self.assertIn("Kill switch", msg)


# ── Data feed smoke test (offline mock) ───────────────────────────────────────

from data_feed import get_ohlcv


class TestDataFeed(unittest.TestCase):
    @patch("data_feed.yf.download")
    def test_returns_ohlcv(self, mock_dl):
        import numpy as np
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

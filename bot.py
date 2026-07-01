"""
Multi-symbol bot — one run_cycle() call per candle close.
Iterates over all configured symbols, manages a shared position cap,
and automatically retrains ML models once per week.
"""

import logging
from datetime import datetime
from dataclasses import dataclass, field

from config import cfg
from t212_client import T212Client, T212Error
from data_feed import get_ohlcv
from strategy import Signal, compute_signal
from risk_manager import (
    is_market_open, position_size,
    daily_loss_exceeded, open_position_count,
)
import notifier

log = logging.getLogger(__name__)


@dataclass
class BotState:
    killed:             bool  = False
    start_cash:         float = 0.0
    last_retrain_week:  int   = -1


class Bot:
    def __init__(self):
        self.client = T212Client()
        self.state  = BotState()
        self._init()

    def _init(self):
        # Use our updated ensure_auth method to establish the session correctly
        if not self.client.ensure_auth():
            log.error("T212 auth check failed.")
            notifier.alert_error("T212 auth check failed.")
            # Prevent trading loops from running until credentials are fixed
            self.state.killed = True
            return

        try:
            cash = self.client.get_cash()
            self.state.start_cash = cash
            symbols = [s["yf"] for s in cfg.SYMBOLS]
            log.info("Bot initialised | cash=%.2f | mode=%s | symbols=%s",
                     cash, cfg.T212_MODE, symbols)
            notifier.alert_startup(cfg.T212_MODE, ", ".join(symbols))
        except T212Error as e:
            log.error("Failed to initialize bot parameters: %s", e)
            self.state.killed = True

        symbols = [s["yf"] for s in cfg.SYMBOLS]
        log.info("Bot initialised | cash=%.2f | mode=%s | symbols=%s",
                 cash, cfg.T212_MODE, symbols)
        notifier.alert_startup(cfg.T212_MODE, ", ".join(symbols))

    # ── Main cycle ────────────────────────────────────────────────────────────

    def run_cycle(self):
        if self.state.killed:
            log.info("Kill switch active — skipping cycle")
            return

        if not is_market_open():
            log.info("Market closed — skipping cycle")
            return

        try:
            self._maybe_retrain()
            self._cycle()
        except T212Error as e:
            log.error("T212 API error: %s", e)
            notifier.alert_error(str(e))
        except Exception as e:
            log.exception("Unexpected error: %s", e)
            notifier.alert_error(str(e))

    def _cycle(self):
        cash      = self.client.get_cash()
        portfolio = self.client.get_portfolio()
        t212_tickers = [s["t212"] for s in cfg.SYMBOLS]

        open_pnl = sum(
            float(p["ppl"]) for p in portfolio if p["ticker"] in t212_tickers
        )

        log.info(f"DEBUG VALUES ──> Start Cash: {self.state.start_cash} | Current Cash: {cash} | Open P&L: {open_pnl}")
        # Portfolio-level open P&L for kill switch
        open_pnl = sum(
            float(p["ppl"]) for p in portfolio if p["ticker"] in t212_tickers
        )

        if daily_loss_exceeded(self.state.start_cash, cash, open_pnl, cfg.MAX_DAILY_LOSS):
            self._trigger_kill_switch(cash, open_pnl)
            return

        n_open = open_position_count(portfolio, t212_tickers)

        for sym in cfg.SYMBOLS:
            self._process_symbol(sym, cash, portfolio, n_open)

    def _process_symbol(self, sym: dict, cash: float, portfolio: list, n_open: int):
        yf, t212, risk = sym["yf"], sym["t212"], sym["risk"]
        
        # If config values are wrapped in a tuple, extract the string element cleanly
        if isinstance(yf, tuple): yf = yf[0]
        if isinstance(t212, tuple): t212 = t212[0]

        try:
            df = get_ohlcv(symbol=str(yf))
            signal, indicators = compute_signal(df, str(yf))
            log.info("%s | %s | %s", yf, signal.value, indicators)
        except Exception as e:
            log.error("%s | data/signal error: %s", yf, e)
            return

        pos         = next((p for p in portfolio if p["ticker"] == t212), None)
        in_position = pos is not None
        price       = indicators["close"]

        if signal == Signal.LONG and not in_position:
            if n_open >= cfg.MAX_OPEN_POSITIONS:
                log.info("%s | LONG signal skipped — max positions (%d) reached",
                         yf, cfg.MAX_OPEN_POSITIONS)
                return
            self._open_long(t212, yf, cash, price, risk, indicators)
            n_open += 1

        elif signal == Signal.FLAT and in_position:
            self._close_long(t212, yf, pos, price)

    # ── Trade actions ─────────────────────────────────────────────────────────

    def _open_long(self, t212, yf, cash, price, risk, indicators):
        qty = position_size(cash, price, risk)
        if qty <= 0:
            log.warning("%s | position size is 0 — not enough cash", yf)
            return
        conf = indicators.get("confidence", "?")
        log.info("BUY %s x%.1f @ ~%.4f (conf=%s)", t212, qty, price, conf)
        self.client.place_market_buy(t212, qty)
        notifier.alert_trade("BUY", t212, qty, price, f"ML confidence={conf}")

    def _close_long(self, t212, yf, pos, price):
        qty = float(pos["quantity"])
        log.info("SELL %s x%.1f @ ~%.4f", t212, qty, price)
        self.client.close_position(t212)
        notifier.alert_trade("SELL", t212, qty, price, "ML exit signal")

    def _trigger_kill_switch(self, cash, open_pnl):
        equity   = cash + open_pnl
        drawdown = (self.state.start_cash - equity) / self.state.start_cash
        log.warning("Kill switch triggered: drawdown=%.2f%%", drawdown * 100)
        for sym in cfg.SYMBOLS:
            try:
                self.client.close_position(sym["t212"])
            except T212Error:
                pass
        self.state.killed = True
        notifier.alert_kill_switch(drawdown)

    # ── Weekly retrain ────────────────────────────────────────────────────────

    def _maybe_retrain(self):
        now  = datetime.utcnow()
        week = now.isocalendar()[1]
        if now.weekday() == cfg.RETRAIN_DAY and week != self.state.last_retrain_week:
            log.info("Weekly retrain triggered (week %d)", week)
            self.state.last_retrain_week = week
            try:
                from train import main as retrain
                retrain()
                log.info("Weekly retrain complete")
            except Exception as e:
                log.error("Weekly retrain failed: %s", e)

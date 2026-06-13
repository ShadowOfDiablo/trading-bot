"""
Core bot logic — one run_cycle() call per candle close.
"""

import logging
from dataclasses import dataclass, field

from config import cfg
from t212_client import T212Client, T212Error
from data_feed import get_ohlcv
from strategy import Signal, compute_signal
from risk_manager import is_market_open, position_size, daily_loss_exceeded
import notifier

log = logging.getLogger(__name__)


@dataclass
class BotState:
    killed:          bool  = False
    in_position:     bool  = False
    start_cash:      float = 0.0
    last_signal:     Signal = field(default=Signal.FLAT)


class Bot:
    def __init__(self):
        self.client = T212Client()
        self.state  = BotState()
        self._init()

    def _init(self):
        try:
            cash = self.client.get_cash()
            self.state.start_cash = cash
            pos  = self.client.get_position(cfg.SYMBOL_T212)
            self.state.in_position = pos is not None
            log.info("Bot initialised | cash=%.2f | in_position=%s | mode=%s",
                     cash, self.state.in_position, cfg.T212_MODE)
            notifier.alert_startup(cfg.T212_MODE, cfg.SYMBOL_T212)
        except T212Error as e:
            log.error("Init failed: %s", e)
            raise

    # ── Main cycle ────────────────────────────────────────────────────────────

    def run_cycle(self):
        if self.state.killed:
            log.info("Kill switch active — skipping cycle")
            return

        if not is_market_open():
            log.info("Market closed — skipping cycle")
            return

        try:
            self._cycle()
        except T212Error as e:
            log.error("T212 API error: %s", e)
            notifier.alert_error(str(e))
        except Exception as e:
            log.exception("Unexpected error in cycle: %s", e)
            notifier.alert_error(str(e))

    def _cycle(self):
        # 1. Fetch data and compute signal
        df = get_ohlcv()
        signal, indicators = compute_signal(df)
        log.info("Signal=%s | %s", signal.value, indicators)

        current_price = indicators["close"]

        # 2. Get account state
        cash = self.client.get_cash()
        pos  = self.client.get_position(cfg.SYMBOL_T212)
        self.state.in_position = pos is not None

        open_pnl = float(pos["ppl"]) if pos else 0.0

        # 3. Kill switch check
        if daily_loss_exceeded(self.state.start_cash, cash, open_pnl):
            self._trigger_kill_switch(cash, open_pnl)
            return

        # 4. Trade logic
        if signal == Signal.LONG and not self.state.in_position:
            self._open_long(cash, current_price)

        elif signal == Signal.FLAT and self.state.in_position:
            self._close_long(pos, current_price)

        self.state.last_signal = signal

    # ── Actions ───────────────────────────────────────────────────────────────

    def _open_long(self, cash: float, price: float):
        qty = position_size(cash, price)
        if qty <= 0:
            log.warning("Position size is 0 — not enough cash or price too high")
            return

        log.info("BUY %s x%.1f @ ~%.4f", cfg.SYMBOL_T212, qty, price)
        self.client.place_market_buy(cfg.SYMBOL_T212, qty)
        self.state.in_position = True
        notifier.alert_trade("BUY", cfg.SYMBOL_T212, qty, price,
                             "EMA crossover + RSI confirmation")

    def _close_long(self, pos: dict, price: float):
        qty = float(pos["quantity"])
        log.info("SELL %s x%.1f @ ~%.4f", cfg.SYMBOL_T212, qty, price)
        self.client.close_position(cfg.SYMBOL_T212)
        self.state.in_position = False
        notifier.alert_trade("SELL", cfg.SYMBOL_T212, qty, price,
                             "EMA crossover exit")

    def _trigger_kill_switch(self, cash: float, open_pnl: float):
        current_equity = cash + open_pnl
        drawdown = (self.state.start_cash - current_equity) / self.state.start_cash
        log.warning("Kill switch: drawdown=%.2f%% — closing position and stopping", drawdown * 100)

        if self.state.in_position:
            try:
                self.client.close_position(cfg.SYMBOL_T212)
                log.info("Emergency close executed")
            except T212Error as e:
                log.error("Emergency close failed: %s", e)

        self.state.killed = True
        notifier.alert_kill_switch(drawdown)

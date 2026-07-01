import math
from datetime import datetime, time
import zoneinfo

NY           = zoneinfo.ZoneInfo("America/New_York")
MARKET_OPEN  = time(9, 35)   # 5 min buffer after open to skip the noise spike
MARKET_CLOSE = time(15, 50)  # 10 min before close to avoid end-of-day spread

# ── Per-Trade Risk Thresholds ────────────────────────────────────────────────
STOP_LOSS_PCT   = 0.010  # -1.0% max loss per trade
TAKE_PROFIT_PCT = 0.015  # +1.5% profit target per trade


def is_market_open() -> bool:
    now_ny = datetime.now(NY)
    if now_ny.weekday() >= 5:
        return False
    return MARKET_OPEN <= now_ny.time() <= MARKET_CLOSE


def position_size(cash: float, price: float, risk_fraction: float) -> float:
    """
    Allocates risk_fraction of available cash to this trade.
    Returns shares rounded to 1 decimal place.
    """
    if price <= 0:
        return 0.0
    return math.floor((cash * risk_fraction / price) * 10) / 10


def daily_loss_exceeded(start_cash: float, current_cash: float,
                        open_pnl: float, max_loss: float) -> bool:
    if start_cash <= 0:
        return False
    drawdown = (start_cash - (current_cash + open_pnl)) / start_cash
    return drawdown >= max_loss


def open_position_count(portfolio: list[dict], tracked_tickers: list[str]) -> int:
    """How many of our tracked symbols currently have open positions."""
    held = {p["ticker"] for p in portfolio}
    return sum(1 for t in tracked_tickers if t in held)


def check_position_safety(ticker: str, portfolio: list[dict]) -> str | None:
    """
    Checks if an individual tracked position has crossed risk protection thresholds.
    Returns 'STOP_LOSS', 'TAKE_PROFIT', or None.
    """
    pos = next((p for p in portfolio if p.get("ticker") == ticker), None)
    if not pos:
        return None

    # T212 API standard keys or map to whatever your broker adapter provides
    initial = float(pos.get("averagePrice", 0))
    current = float(pos.get("currentPrice", 0))

    if initial <= 0:
        return None

    trade_return = (current / initial) - 1.0

    if trade_return <= -STOP_LOSS_PCT:
        return "STOP_LOSS"
    if trade_return >= TAKE_PROFIT_PCT:
        return "TAKE_PROFIT"

    return None
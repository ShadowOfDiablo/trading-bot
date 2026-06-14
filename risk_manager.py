import math
from datetime import datetime, time
import zoneinfo

NY           = zoneinfo.ZoneInfo("America/New_York")
MARKET_OPEN  = time(9, 35)   # 5 min buffer after open to skip the noise spike
MARKET_CLOSE = time(15, 50)  # 10 min before close to avoid end-of-day spread


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

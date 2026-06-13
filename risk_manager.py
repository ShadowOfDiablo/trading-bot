import math
from datetime import datetime, time
import zoneinfo
from config import cfg


NY = zoneinfo.ZoneInfo("America/New_York")
MARKET_OPEN  = time(9, 35)   # slight buffer after open to avoid the noise spike
MARKET_CLOSE = time(15, 50)  # exit before close to avoid end-of-day spread


def is_market_open() -> bool:
    now_ny = datetime.now(NY)
    if now_ny.weekday() >= 5:   # Saturday=5, Sunday=6
        return False
    return MARKET_OPEN <= now_ny.time() <= MARKET_CLOSE


def position_size(cash: float, price: float) -> float:
    """
    Allocates RISK_PER_TRADE fraction of available cash.
    Returns whole shares (floored) — T212 Invest supports fractional shares
    but whole numbers are safer to start with.
    """
    if price <= 0:
        return 0.0
    affordable = (cash * cfg.RISK_PER_TRADE) / price
    return math.floor(affordable * 10) / 10   # 1 decimal place


def daily_loss_exceeded(start_cash: float, current_cash: float, open_pnl: float) -> bool:
    """
    Returns True if the bot should stop for the day.
    Compares starting equity to current equity (cash + unrealised P&L).
    """
    if start_cash <= 0:
        return False
    current_equity = current_cash + open_pnl
    drawdown = (start_cash - current_equity) / start_cash
    return drawdown >= cfg.MAX_DAILY_LOSS

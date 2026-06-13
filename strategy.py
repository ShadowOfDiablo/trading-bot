from enum import Enum
import pandas as pd
from config import cfg


class Signal(Enum):
    LONG = "LONG"
    FLAT = "FLAT"


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_signal(df: pd.DataFrame) -> tuple[Signal, dict]:
    """
    EMA crossover + RSI filter.
    Long when fast EMA is above slow EMA and RSI is above threshold.
    Returns (Signal, dict of latest indicator values for logging).
    """
    close = df["close"]

    ema_fast = _ema(close, cfg.EMA_FAST)
    ema_slow = _ema(close, cfg.EMA_SLOW)
    rsi      = _rsi(close, cfg.RSI_PERIOD)

    latest = {
        "close":    round(float(close.iloc[-1]),    4),
        "ema_fast": round(float(ema_fast.iloc[-1]), 4),
        "ema_slow": round(float(ema_slow.iloc[-1]), 4),
        "rsi":      round(float(rsi.iloc[-1]),      2),
    }

    long_condition = (
        ema_fast.iloc[-1] > ema_slow.iloc[-1]
        and rsi.iloc[-1] > cfg.RSI_THRESHOLD
    )

    signal = Signal.LONG if long_condition else Signal.FLAT
    return signal, latest

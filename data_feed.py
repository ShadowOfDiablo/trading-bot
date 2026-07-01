import pandas as pd
import yfinance as yf
from config import cfg


def get_ohlcv(symbol: str | None = None, interval: str | None = None, bars: int = 100) -> pd.DataFrame:
    """
    Fetch the last `bars` candles from Yahoo Finance.
    interval examples: '1h', '30m', '1d'
    Returns DataFrame with lowercase columns: open, high, low, close, volume.
    """
    symbol   = symbol   or cfg.SYMBOL_YF
    interval = interval or cfg.INTERVAL

    # yfinance needs a period wide enough to cover bars + indicators warmup
    period_map = {"1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d",
                  "1h": "730d", "1d": "5y"}
    period = period_map.get(interval, "730d")

    df = yf.download(symbol, period=period, interval=interval,
                     auto_adjust=True, progress=False)

    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol} ({interval})")

    # ─── FIX: Flatten MultiIndex columns if yfinance returns tuples ───
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()

    # Drop the in-progress (incomplete) current candle
    df = df.iloc[:-1]

    return df.tail(bars)

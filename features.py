import numpy as np
import pandas as pd


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


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the ML feature matrix from an OHLCV DataFrame.
    All features are normalised or ratio-based so they generalise
    across different price levels and instruments.
    Returns a DataFrame with NaN rows dropped.
    """
    close  = df["close"]
    volume = df["volume"]

    feat = pd.DataFrame(index=df.index)

    # ── EMA ratios ────────────────────────────────────────────────────────────
    ema20  = _ema(close, 20)
    ema50  = _ema(close, 50)
    ema200 = _ema(close, 200)

    feat["close_to_ema20"]  = close / ema20  - 1
    feat["ema20_to_ema50"]  = ema20  / ema50  - 1
    feat["close_to_ema200"] = close / ema200 - 1

    # ── RSI (normalised 0–1) ──────────────────────────────────────────────────
    feat["rsi_7"]  = _rsi(close, 7)  / 100
    feat["rsi_14"] = _rsi(close, 14) / 100

    # ── Volatility ────────────────────────────────────────────────────────────
    feat["atr_14"] = _atr(df, 14) / close   # ATR as fraction of price

    # ── Volume ────────────────────────────────────────────────────────────────
    vol_ma = volume.rolling(20).mean()
    feat["volume_ratio"] = volume / vol_ma   # >1 = above average volume

    # ── Momentum ──────────────────────────────────────────────────────────────
    feat["mom_1h"]  = close / close.shift(1)  - 1
    feat["mom_4h"]  = close / close.shift(4)  - 1
    feat["mom_12h"] = close / close.shift(12) - 1
    feat["mom_24h"] = close / close.shift(24) - 1

    # ── Bollinger Band position (0 = lower band, 1 = upper band) ─────────────
    sma20    = close.rolling(20).mean()
    std20    = close.rolling(20).std()
    bb_range = (sma20 + 2 * std20) - (sma20 - 2 * std20)
    feat["bb_pos"] = (close - (sma20 - 2 * std20)) / bb_range.replace(0, np.nan)

    # ── Time features (cyclical sin/cos so midnight ≈ 23:00) ─────────────────
    if hasattr(df.index, "hour"):
        hour = df.index.hour
        dow  = df.index.dayofweek
        feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        feat["dow_sin"]  = np.sin(2 * np.pi * dow  / 5)
        feat["dow_cos"]  = np.cos(2 * np.pi * dow  / 5)

    return feat.dropna()

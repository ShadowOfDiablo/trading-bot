from __future__ import annotations
from enum import Enum
import pandas as pd
from model import load, predict
from config import cfg


class Signal(Enum):
    LONG = "LONG"
    FLAT = "FLAT"


def compute_signal(df: pd.DataFrame, symbol_yf: str) -> tuple[Signal, dict]:
    """
    Uses the trained ML model for symbol_yf to generate a trading signal.
    Falls back to FLAT with a warning if the model hasn't been trained yet
    (run train.py first).
    """
    model_data = load(symbol_yf)

    if model_data is None:
        return Signal.FLAT, {
            "close": round(float(df["close"].iloc[-1]), 4),
            "warning": f"No model for {symbol_yf} — run: python train.py",
        }

    ml_signal, confidence = predict(
        model_data, df, threshold=cfg.ML_CONFIDENCE_THRESHOLD
    )

    indicators = {
        "close":      round(float(df["close"].iloc[-1]), 4),
        "confidence": confidence,
        "threshold":  cfg.ML_CONFIDENCE_THRESHOLD,
    }

    return Signal.LONG if ml_signal == 1 else Signal.FLAT, indicators

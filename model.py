"""
Per-symbol RandomForest classifier.

Training target: will the price rise by a volatility-adjusted threshold 
(e.g., >= 0.5 * ATR) within the next FORWARD_HOURS candles? 
(binary: 1 = yes / 0 = no)

Walk-forward split is used for validation so no future data leaks into training.
"""

import os
import pickle
import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from features import build_features

log = logging.getLogger(__name__)

MODELS_DIR     = os.path.join(os.path.dirname(__file__), "models")
FORWARD_HOURS  = 4      # how many candles ahead we're predicting
ATR_MULTIPLIER = 0.5    # Require a move >= 50% of current ATR to count as "long"


def _path(symbol_yf: str) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    return os.path.join(MODELS_DIR, f"{symbol_yf}.pkl")


# ── Training ──────────────────────────────────────────────────────────────────

def train(symbol_yf: str, df: pd.DataFrame) -> dict:
    """
    Trains and saves a tuned model for symbol_yf.
    Returns a metrics dict {accuracy, precision, recall}.
    """
    features = build_features(df)

    # Future return label — shift(-N) looks N candles ahead
    forward_ret = df["close"].shift(-FORWARD_HOURS) / df["close"] - 1
    
    # ─── FIX: Align indices BEFORE comparing ───
    # Drop rows where indicators are warming up, or where future returns are NaN
    idx = features.index.intersection(forward_ret.dropna().index)
    features = features.loc[idx]
    forward_ret = forward_ret.loc[idx]

    # Dynamic Target: Volatility-adjusted using normalized ATR
    if "atr_14" in features.columns:
        threshold = ATR_MULTIPLIER * features["atr_14"]
    else:
        # Fallback if atr_14 is missing
        threshold = 0.003
        
    target = (forward_ret > threshold).astype(int)

    # Now X and y are perfectly aligned
    X, y = features, target

    # Time-ordered 70/30 split — NEVER shuffle financial time series
    split       = int(len(X) * 0.70)
    X_tr, X_te  = X.iloc[:split],  X.iloc[split:]
    y_tr, y_te  = y.iloc[:split],  y.iloc[split:]

    # Setup cross-validation for time series
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Define a tighter, more nimble hyperparameter grid
    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [6, 8, 12],                # Added deeper trees to capture finer regimes
        "min_samples_leaf": [3, 8, 15],          # Slightly lower bounds to capture patterns
        "max_features": ["sqrt"]
    }

    # Base classifier (Note: class_weight="balanced" removed to boost precision)
    # 3. Use 'balanced_subsample' to account for the volatility target scarcity
    base_clf = RandomForestClassifier(
        class_weight="balanced_subsample",       # Dynamically balances weights per tree split
        random_state=42, 
        n_jobs=-1
    )

    # Search for the best parameters, optimizing strictly for precision
    # Search for parameters using F1 scoring to ensure the bot actually finds trades
    search = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=param_grid,
        n_iter=10,             
        cv=tscv,
        scoring="average_precision",             # <── Optimizes the entire Precision/Recall tradeoff curve
        random_state=42,
        n_jobs=-1
    )
    
    # Fit the search
    search.fit(X_tr, y_tr)
    clf = search.best_estimator_
    
    log.info("%s best params: %s", symbol_yf, search.best_params_)

    # Test the winning model on the 30% holdout set
    y_pred  = clf.predict(X_te)
    report  = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    metrics = {
        "accuracy":  round(report["accuracy"], 3),
        "precision": round(report.get("1", {}).get("precision", 0), 3),
        "recall":    round(report.get("1", {}).get("recall",    0), 3),
    }

    log.info(
        "%s trained | acc=%.2f prec=%.2f recall=%.2f | train=%d test=%d",
        symbol_yf, metrics["accuracy"], metrics["precision"], metrics["recall"],
        len(X_tr), len(X_te),
    )

    with open(_path(symbol_yf), "wb") as f:
        pickle.dump({"model": clf, "feature_cols": list(X.columns)}, f)

    return metrics


# ── Inference ─────────────────────────────────────────────────────────────────

def load(symbol_yf: str) -> dict | None:
    """Returns the saved model dict or None if not trained yet."""
    p = _path(symbol_yf)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def predict(model_data: dict, df: pd.DataFrame, threshold: float = 0.55) -> tuple[int, float]:
    """
    Returns (signal, confidence).
    signal=1 → LONG, signal=0 → FLAT.
    confidence is the model's probability of the LONG class.
    """
    features = build_features(df)
    if features.empty:
        return 0, 0.0

    cols   = model_data["feature_cols"]
    latest = features[cols].iloc[[-1]]

    proba      = model_data["model"].predict_proba(latest)[0]
    confidence = float(proba[1]) if len(proba) > 1 else 0.0
    signal     = 1 if confidence >= threshold else 0

    return signal, round(confidence, 3)
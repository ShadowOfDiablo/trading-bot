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

    # 1. Compute the raw forward returns matching the raw DataFrame index
    forward_ret = df["close"].shift(-FORWARD_HOURS) / df["close"] - 1
    
    # 2. Extract the volatility metric matching the raw DataFrame index
    # (Since features.index is a subset of df.index, we match indices exactly)
    if "atr_14" in features.columns:
        vol_threshold = ATR_MULTIPLIER * features["atr_14"]
    else:
        vol_threshold = pd.Series(0.003, index=features.index)

    # 3. Align all three components to a perfectly matching intersection index
    intersect_idx = features.index.intersection(forward_ret.dropna().index)
    
    X = features.loc[intersect_idx]
    y_ret = forward_ret.loc[intersect_idx]
    y_thresh = vol_threshold.loc[intersect_idx]
        
    # 4. Perform the comparison on identical structural alignments
    target = (y_ret > y_thresh).astype(int)

    # Time-ordered 70/30 split — NEVER shuffle financial time series
    split       = int(len(X) * 0.70)
    X_tr, X_te  = X.iloc[:split],  X.iloc[split:]
    y_tr, y_te  = target.iloc[:split],  target.iloc[split:]

    # Setup cross-validation for time series
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Define a tighter, more nimble hyperparameter grid
    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [6, 8, 12],                
        "min_samples_leaf": [3, 8, 15],          
        "max_features": ["sqrt"]
    }

    # Use 'balanced_subsample' to handle volatility target scarcity
    base_clf = RandomForestClassifier(
        class_weight="balanced_subsample",       
        random_state=42, 
        n_jobs=-1
    )

    # Search using average_precision (PR-AUC) as the optimization score
    search = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=param_grid,
        n_iter=10,             
        cv=tscv,
        scoring="average_precision",             
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
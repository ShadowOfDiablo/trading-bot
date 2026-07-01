import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    T212_API_KEY  = os.getenv("T212_API_KEY", "")
    T212_API_SECRET = os.getenv("T212_API_SECRET", "")
    USE_MOCK_T212 = os.getenv("USE_MOCK_T212", "false").lower() in ("1", "true", "yes")
    # Keep Trading212 on demo mode for now to avoid accidental live trading.
    T212_MODE     = os.getenv("T212_MODE", "demo")
    T212_BASE_URL = "https://demo.trading212.com/api/v0"

    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

    # Symbols to trade — t212 ticker, yfinance symbol, per-trade risk fraction
    # ETFs get 1% risk, individual stocks get 0.5% (more volatile)
    SYMBOLS = [
        {"t212": "QQQ_US_EQ",  "yf": "QQQ",  "risk": 0.01},
        {"t212": "SPY_US_EQ",  "yf": "SPY",  "risk": 0.01},
        {"t212": "TSLA_US_EQ", "yf": "TSLA", "risk": 0.005},
        {"t212": "NVDA_US_EQ", "yf": "NVDA", "risk": 0.005},
    ]

    MAX_OPEN_POSITIONS     = 2     # never hold more than this many at once
    ML_CONFIDENCE_THRESHOLD = 0.55  # model must be ≥55% confident to go long
    RETRAIN_DAY            = 6     # 6 = Sunday — retrain models weekly

    INTERVAL       = os.getenv("INTERVAL", "1h")
    MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "0.03"))

    GITHUB_TOKEN              = os.getenv("GITHUB_TOKEN", "")
    GITHUB_REPO               = os.getenv("GITHUB_REPO", "")
    MODEL_SYNC_METHOD         = os.getenv("MODEL_SYNC_METHOD", "github")
    MODEL_SYNC_AUTO_FETCH     = os.getenv("MODEL_SYNC_AUTO_FETCH", "false").lower() in ("1", "true", "yes")
    MODEL_SYNC_VERSION_SUFFIX = os.getenv("MODEL_SYNC_VERSION_SUFFIX", "weekend")

cfg = Config()

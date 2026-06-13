import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    T212_API_KEY   = os.getenv("T212_API_KEY", "")
    T212_MODE      = os.getenv("T212_MODE", "demo")
    T212_BASE_URL  = (
        "https://demo.trading212.com/api/v0"
        if T212_MODE == "demo"
        else "https://live.trading212.com/api/v0"
    )

    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

    SYMBOL_T212 = os.getenv("SYMBOL_T212", "QQQ_US_EQ")
    SYMBOL_YF   = os.getenv("SYMBOL_YF",   "QQQ")
    INTERVAL    = os.getenv("INTERVAL",    "1h")

    EMA_FAST      = int(os.getenv("EMA_FAST",      "20"))
    EMA_SLOW      = int(os.getenv("EMA_SLOW",      "50"))
    RSI_PERIOD    = int(os.getenv("RSI_PERIOD",    "14"))
    RSI_THRESHOLD = float(os.getenv("RSI_THRESHOLD", "50"))

    RISK_PER_TRADE  = float(os.getenv("RISK_PER_TRADE",  "0.01"))
    MAX_DAILY_LOSS  = float(os.getenv("MAX_DAILY_LOSS",  "0.03"))

cfg = Config()

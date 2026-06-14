"""
Standalone training script — run this once before starting the bot,
then again weekly to keep models fresh on new market data.

Usage:
    python train.py
    python train.py --symbol NVDA   # retrain one symbol only
"""

import sys
import logging
import argparse

from config import cfg
from data_feed import get_ohlcv
from model import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def train_symbol(sym: dict):
    yf = sym["yf"]
    log.info("── Training %s ──────────────────────────", yf)
    try:
        # 2 years of hourly bars (yfinance max for 1h interval)
        df = get_ohlcv(symbol=yf, interval=cfg.INTERVAL, bars=4000)
        metrics = train(yf, df)
        log.info(
            "%s complete | accuracy=%.2f | precision=%.2f | recall=%.2f",
            yf, metrics["accuracy"], metrics["precision"], metrics["recall"],
        )
        return True
    except Exception as e:
        log.error("%s failed: %s", yf, e)
        return False


def main(symbol_filter: str | None = None):
    symbols = cfg.SYMBOLS
    if symbol_filter:
        symbols = [s for s in symbols if s["yf"].upper() == symbol_filter.upper()]
        if not symbols:
            log.error("Symbol '%s' not found in config", symbol_filter)
            sys.exit(1)

    log.info("Training %d model(s): %s", len(symbols), [s["yf"] for s in symbols])
    results = [train_symbol(s) for s in symbols]

    passed = sum(results)
    log.info("Done — %d/%d models trained successfully. Models saved to models/",
             passed, len(results))

    if passed < len(results):
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", help="Train a single symbol (e.g. NVDA)")
    args = parser.parse_args()
    main(args.symbol)

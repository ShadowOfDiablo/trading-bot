"""
Standalone training script — run this once before starting the bot,
then again weekly to keep models fresh on new market data.

Usage:
    python train.py                 # Train all symbols in config
    python train.py NVDA            # Train one symbol only
    python train.py NVDA TSLA QQQ   # Train multiple specific symbols
    python train.py --upload        # Train all and upload to GitHub
"""

import sys
import logging
import argparse
from __future__ import annotations
from config import cfg
from data_feed import get_ohlcv
from model import train
from model_sync import upload_models, current_version_name

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


def main(tickers: list[str] | None = None) -> bool:
    """
    Main training entry point. 
    Accepts an optional list of ticker strings. If empty/None, trains all config symbols.
    Returns True if all requested models trained successfully, False otherwise.
    """
    symbols = cfg.SYMBOLS
    
    if tickers:
        # Normalize inputs to uppercase strings
        target_tickers = [t.upper() for t in tickers]
        symbols = [s for s in symbols if s["yf"].upper() in target_tickers]
        
        if not symbols:
            log.error("None of the requested tickers %s were found in config", tickers)
            return False

    log.info("Training %d model(s): %s", len(symbols), [s["yf"] for s in symbols])
    results = [train_symbol(s) for s in symbols]

    passed = sum(results)
    log.info("Done — %d/%d models trained successfully. Models saved to models/",
             passed, len(results))

    return passed == len(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Using nargs="*" lets us pass 0, 1, or multiple space-separated tickers
    parser.add_argument("tickers", nargs="*", help="Optional ticker(s) to train (e.g. NVDA TSLA)")
    parser.add_argument("--upload", action="store_true",
                        help="Upload trained models to GitHub as a packaged release")
    parser.add_argument("--version-suffix", default=None,
                        help="Suffix for the uploaded model version name (e.g. weekend)")
    args = parser.parse_args()
    
    success = main(args.tickers)
    
    if args.upload and success:
        version = current_version_name(args.version_suffix or cfg.MODEL_SYNC_VERSION_SUFFIX)
        upload_models(version)
        
    sys.exit(0 if success else 1)
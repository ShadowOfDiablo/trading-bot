"""
Entry point. Aligns to hourly candle closes and loops forever.

Usage:
    cp .env.example .env
    # edit .env with your keys
    pip install -r requirements.txt
    python run.py
"""

import logging
import time
from datetime import datetime, timedelta

from bot import Bot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log"),
    ],
)
log = logging.getLogger(__name__)


def seconds_until_next_candle(interval_minutes: int = 60) -> float:
    """Returns seconds to sleep until the next candle close (plus a 5s buffer)."""
    now = datetime.utcnow()
    minutes_past = now.minute % interval_minutes
    seconds_past = minutes_past * 60 + now.second
    remaining    = interval_minutes * 60 - seconds_past + 5  # 5s buffer
    return remaining


def main():
    log.info("Starting trading bot")
    bot = Bot()

    # Run once immediately on startup (catches up if bot was offline)
    bot.run_cycle()

    while True:
        sleep_for = seconds_until_next_candle(60)
        log.info("Next cycle in %.0fs", sleep_for)
        time.sleep(sleep_for)
        bot.run_cycle()


if __name__ == "__main__":
    main()

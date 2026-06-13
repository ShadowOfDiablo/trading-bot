"""
Sends alerts to Telegram via the Bot API.
If no token/chat_id is configured, messages are silently skipped.
"""

import logging
import requests
from config import cfg

log = logging.getLogger(__name__)


def _send(text: str) -> None:
    if not cfg.TELEGRAM_BOT_TOKEN or not cfg.TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{cfg.TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={
            "chat_id": cfg.TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown",
        }, timeout=5)
    except Exception as e:
        log.warning("Telegram send failed: %s", e)


def alert_trade(action: str, symbol: str, quantity: float, price: float, reason: str) -> None:
    emoji = "🟢" if action == "BUY" else "🔴"
    _send(
        f"{emoji} *{action}* `{symbol}`\n"
        f"Qty: `{quantity}` @ ~`{price}`\n"
        f"Reason: {reason}"
    )


def alert_kill_switch(drawdown_pct: float) -> None:
    _send(f"⛔ *Kill switch triggered* — daily drawdown `{drawdown_pct:.1%}` exceeded limit. Bot stopped for today.")


def alert_error(message: str) -> None:
    _send(f"⚠️ *Bot error*\n```{message}```")


def alert_startup(mode: str, symbol: str) -> None:
    _send(f"🤖 *Trading bot started*\nMode: `{mode}` | Symbol: `{symbol}`")

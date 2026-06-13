"""
Trading212 REST API client (Invest/ISA account — long-only equity).

T212 API docs: https://t212public-api-docs.redoc.ly/
Instrument tickers follow the format SYMBOL_EXCHANGE_EQ, e.g. QQQ_US_EQ.
"""

import requests
from config import cfg


class T212Error(Exception):
    pass


class T212Client:
    def __init__(self):
        self.base = cfg.T212_BASE_URL
        self.headers = {
            "Authorization": cfg.T212_API_KEY,
            "Content-Type": "application/json",
        }

    def _get(self, path: str) -> dict | list:
        r = requests.get(f"{self.base}{path}", headers=self.headers, timeout=10)
        if not r.ok:
            raise T212Error(f"GET {path} → {r.status_code}: {r.text}")
        return r.json()

    def _post(self, path: str, body: dict) -> dict:
        r = requests.post(f"{self.base}{path}", json=body, headers=self.headers, timeout=10)
        if not r.ok:
            raise T212Error(f"POST {path} → {r.status_code}: {r.text}")
        return r.json()

    def _delete(self, path: str) -> None:
        r = requests.delete(f"{self.base}{path}", headers=self.headers, timeout=10)
        if not r.ok:
            raise T212Error(f"DELETE {path} → {r.status_code}: {r.text}")

    # ── Account ──────────────────────────────────────────────────────────────

    def get_cash(self) -> float:
        """Returns free cash available for trading."""
        data = self._get("/equity/account/cash")
        return float(data["free"])

    def get_account_info(self) -> dict:
        return self._get("/equity/account/info")

    # ── Portfolio ─────────────────────────────────────────────────────────────

    def get_portfolio(self) -> list[dict]:
        """Returns all open positions."""
        return self._get("/equity/portfolio")

    def get_position(self, ticker: str) -> dict | None:
        """Returns the open position for ticker, or None if not held."""
        positions = self.get_portfolio()
        for p in positions:
            if p["ticker"] == ticker:
                return p
        return None

    # ── Orders ────────────────────────────────────────────────────────────────

    def place_market_buy(self, ticker: str, quantity: float) -> dict:
        return self._post("/equity/orders/market", {
            "ticker": ticker,
            "quantity": round(quantity, 4),
        })

    def place_market_sell(self, ticker: str, quantity: float) -> dict:
        # Negative quantity = sell on T212 market orders
        return self._post("/equity/orders/market", {
            "ticker": ticker,
            "quantity": -round(abs(quantity), 4),
        })

    def close_position(self, ticker: str) -> dict | None:
        """Sell the entire open position for ticker if one exists."""
        pos = self.get_position(ticker)
        if pos is None:
            return None
        qty = float(pos["quantity"])
        return self.place_market_sell(ticker, qty)

    def get_open_orders(self) -> list[dict]:
        return self._get("/equity/orders")

    def cancel_order(self, order_id: int) -> None:
        self._delete(f"/equity/orders/{order_id}")

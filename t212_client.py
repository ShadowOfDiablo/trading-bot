"""
Trading212 REST API client (Invest/ISA account — long-only equity).

T212 API docs: https://t212public-api-docs.redoc.ly/
Instrument tickers follow the format SYMBOL_EXCHANGE_EQ, e.g. QQQ_US_EQ.
"""

import base64

import requests

from config import cfg


class T212Error(Exception):
    pass


class T212Client:
    def __init__(self):
        self.base = cfg.T212_BASE_URL
        self.api_key = (cfg.T212_API_KEY or "").strip()
        self.api_secret = (getattr(cfg, "T212_API_SECRET", "") or "").strip()
        self.use_mock = bool(getattr(cfg, "USE_MOCK_T212", False))
        self.headers = {"Content-Type": "application/json"}
        self._configure_headers()

    def _configure_headers(self) -> None:
        if self.api_key and self.api_secret:
            creds = f"{self.api_key}:{self.api_secret}".encode("utf-8")
            token = base64.b64encode(creds).decode("utf-8")
            self.headers = {
                "Authorization": f"Basic {token}",
                "Content-Type": "application/json",
            }
            return

        self.headers = {"Content-Type": "application/json"}

    def _get(self, path: str) -> dict | list:
        r = requests.get(f"{self.base}{path}", headers=self.headers, timeout=10)
        if not r.ok:
            raise T212Error(f"GET {path} → {r.status_code}: {r.text}")
        return r.json()

    def check_auth(self) -> tuple[bool, int, str]:
        """Return whether the Trading212 API is reachable, or whether we are in mock mode."""
        if self.use_mock:
            return True, 200, "mock mode active"

        try:
            r = requests.get(
                f"{self.base}/equity/account/info",
                headers=self.headers,
                timeout=10,
            )
            return (r.ok, r.status_code, r.text)
        except Exception as e:
            return (False, 0, str(e))

    def ensure_auth(self) -> bool:
        self._configure_headers()
        return self.check_auth()[0]

    def _post(self, path: str, body: dict) -> dict:
        r = requests.post(f"{self.base}{path}", json=body, headers=self.headers, timeout=10)
        if not r.ok:
            raise T212Error(f"POST {path} → {r.status_code}: {r.text}")
        return r.json()

    def _delete(self, path: str) -> None:
        r = requests.delete(f"{self.base}{path}", headers=self.headers, timeout=10)
        if not r.ok:
            raise T212Error(f"DELETE {path} → {r.status_code}: {r.text}")

    def get_cash(self) -> float:
        """Fetch the live available trading cash balance from the Trading 212 API."""
        r = requests.get(f"{self.base}/equity/account/cash", headers=self.headers, timeout=10)
        
        if not r.ok:
            raise T212Error(f"Failed to fetch cash balance. HTTP Status: {r.status_code}")
            
        data = r.json()
        
        # Handle Trading 212's standard nested object structure safely
        if isinstance(data, dict):
            if "cash" in data and isinstance(data["cash"], dict):
                return float(data["cash"].get("availableToTrade", 0.0))
            if "free" in data:
                return float(data["free"])
            if "totalValue" in data:
                return float(data["totalValue"])
                
        raise T212Error(f"Unexpected response payload format from T212: {data}")

    def get_account_info(self) -> dict:
        return self._get("/equity/account/info")

    def get_portfolio(self) -> list[dict]:
        """Return open positions from Trading212 when available."""
        try:
            data = self._get("/equity/portfolio")
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

    def get_position(self, ticker: str) -> dict | None:
        positions = self.get_portfolio()
        for p in positions:
            if p.get("ticker") == ticker:
                return p
        return None

    def place_market_buy(self, ticker: str, quantity: float) -> dict:
        return self._post("/equity/orders/market", {
            "ticker": ticker,
            "quantity": round(quantity, 4),
        })

    def place_market_sell(self, ticker: str, quantity: float) -> dict:
        return self._post("/equity/orders/market", {
            "ticker": ticker,
            "quantity": -round(abs(quantity), 4),
        })

    def close_position(self, ticker: str) -> dict | None:
        pos = self.get_position(ticker)
        if pos is None:
            return None
        qty = float(pos.get("quantity", 0.0))
        return self.place_market_sell(ticker, qty)

    def get_open_orders(self) -> list[dict]:
        try:
            data = self._get("/equity/orders")
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

    def cancel_order(self, order_id: int) -> None:
        self._delete(f"/equity/orders/{order_id}")

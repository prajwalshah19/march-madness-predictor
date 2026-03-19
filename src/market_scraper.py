"""Market data collection from Kalshi and Polymarket prediction markets."""

from __future__ import annotations

import base64
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from src.config import CFG


def market_data_available() -> bool:
    """Check if any market API credentials are configured."""
    return bool(
        os.environ.get("KALSHI_ACCESS_KEY")
        or os.environ.get("POLYMARKET_API_KEY")
    )


class KalshiClient:
    """Client for the Kalshi prediction market API.

    Uses RSA-PSS key signing per https://docs.kalshi.com/getting_started/api_keys
    """

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self) -> None:
        self.access_key = os.environ.get("KALSHI_ACCESS_KEY", "")
        key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")

        if not self.access_key:
            raise ValueError("KALSHI_ACCESS_KEY not set in environment")
        if not key_path:
            raise ValueError("KALSHI_PRIVATE_KEY_PATH not set in environment")

        key_file = Path(key_path)
        if not key_file.exists():
            raise FileNotFoundError(f"Kalshi private key not found: {key_file}")

        with open(key_file, "rb") as f:
            self._private_key = serialization.load_pem_private_key(f.read(), password=None)

    def _sign_request(self, method: str, path: str) -> dict[str, str]:
        """Generate signed headers for a Kalshi API request.

        Headers:
            KALSHI-ACCESS-KEY: Key ID
            KALSHI-ACCESS-TIMESTAMP: millisecond timestamp
            KALSHI-ACCESS-SIGNATURE: RSA-PSS signed (timestamp + method + path)
        """
        timestamp_ms = str(int(time.time() * 1000))

        # Strip query params from path for signing
        path_no_query = urlparse(path).path

        # Message: timestamp + METHOD + path (no query params)
        message = f"{timestamp_ms}{method.upper()}{path_no_query}"

        signature = self._private_key.sign(
            message.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )

        return {
            "KALSHI-ACCESS-KEY": self.access_key,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
            "Content-Type": "application/json",
        }

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make an authenticated GET request."""
        url = f"{self.BASE_URL}{endpoint}"
        # Sign against the API path (not full URL)
        headers = self._sign_request("GET", f"/trade-api/v2{endpoint}")
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # Series tickers for NCAA basketball markets on Kalshi
    NCAA_SERIES = [
        "KXMARMAD",        # Men's College Basketball Champion
        "KXWMARMAD",       # Women's College Basketball Champion
        "KXNCAAMBGAME",    # Men's College Basketball matchup games
        "KXMARMADSEED",    # Seed to win men's championship
        "KXMARMADCONFWIN", # Conference to win men's championship
        "KXMARMADPTS",     # Tournament player points
        "KXNCAAMBMOP",     # Men's tournament MOP
        "KXNCAAWBMOP",     # Women's tournament MOP
    ]

    def get_ncaa_markets(self) -> list[dict[str, Any]]:
        """Fetch NCAA basketball markets across all relevant series."""
        markets = []

        for series in self.NCAA_SERIES:
            cursor: Optional[str] = None
            for _ in range(10):  # Max 10 pages per series
                params: dict[str, Any] = {
                    "limit": 100,
                    "series_ticker": series,
                }
                if cursor:
                    params["cursor"] = cursor

                try:
                    data = self._get("/markets", params)
                except requests.HTTPError:
                    break

                batch = data.get("markets", [])
                markets.extend(batch)

                cursor = data.get("cursor")
                if not cursor or not batch:
                    break

        return markets

    def get_orderbook(self, ticker: str) -> dict[str, Any]:
        """Fetch order book for a specific market."""
        data = self._get(f"/markets/{ticker}/orderbook")
        return data.get("orderbook", {})

    def get_trades(self, ticker: str, limit: int = 100) -> list[dict[str, Any]]:
        """Fetch recent trades for a market."""
        data = self._get("/markets/trades", {"ticker": ticker, "limit": limit})
        return data.get("trades", [])

    def fetch_all_ncaa_data(self) -> dict[str, Any]:
        """Fetch all NCAA market data including order books and trades.

        Only fetches orderbook/trades for active markets to avoid
        wasting API calls on finalized/settled markets.
        """
        markets = self.get_ncaa_markets()
        active = [m for m in markets if m.get("status") == "active"]
        print(f"    Found {len(markets)} Kalshi NCAA markets ({len(active)} active)")

        enriched_markets = []
        for market in active:
            ticker = market.get("ticker", "")
            try:
                market["orderbook"] = self.get_orderbook(ticker)
                market["recent_trades"] = self.get_trades(ticker)
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                market["orderbook"] = {}
                market["recent_trades"] = []
                print(f"    WARNING: Failed to fetch details for {ticker}: {e}")

            enriched_markets.append(market)

        # Also include non-active markets (with basic data, no orderbook)
        # so we have last_price data from settled/finalized markets
        for market in markets:
            if market.get("status") != "active":
                market["orderbook"] = {}
                market["recent_trades"] = []
                enriched_markets.append(market)

        return {
            "source": "kalshi",
            "timestamp": datetime.utcnow().isoformat(),
            "markets": enriched_markets,
        }


class PolymarketClient:
    """Client for the Polymarket CLOB API."""

    BASE_URL = "https://clob.polymarket.com"

    def __init__(self) -> None:
        self.api_key = os.environ.get("POLYMARKET_API_KEY", "")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get_ncaa_markets(self) -> list[dict[str, Any]]:
        """Fetch NCAA basketball markets."""
        markets = []
        for query in ["NCAA", "March Madness", "college basketball"]:
            try:
                resp = requests.get(
                    f"{self.BASE_URL}/markets",
                    headers=self._headers(),
                    params={"tag": query, "active": True},
                    timeout=30,
                )
                resp.raise_for_status()
                batch = resp.json()
                if isinstance(batch, list):
                    markets.extend(batch)
            except Exception as e:
                print(f"    WARNING: Polymarket search for '{query}' failed: {e}")

        return markets

    def get_orderbook(self, token_id: str) -> dict[str, Any]:
        """Fetch order book for a specific market."""
        resp = requests.get(
            f"{self.BASE_URL}/book",
            headers=self._headers(),
            params={"token_id": token_id},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def fetch_all_ncaa_data(self) -> dict[str, Any]:
        """Fetch all NCAA market data."""
        markets = self.get_ncaa_markets()
        print(f"    Found {len(markets)} Polymarket NCAA markets")

        for market in markets:
            token_id = market.get("condition_id", "")
            if token_id:
                try:
                    market["orderbook"] = self.get_orderbook(token_id)
                    time.sleep(0.2)
                except Exception:
                    market["orderbook"] = {}

        return {
            "source": "polymarket",
            "timestamp": datetime.utcnow().isoformat(),
            "markets": markets,
        }


def fetch_market_data() -> dict[str, Any]:
    """Fetch market data from all available sources.

    Returns combined market data dict.
    """
    combined: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "kalshi": None,
        "polymarket": None,
    }

    # Kalshi
    if os.environ.get("KALSHI_ACCESS_KEY"):
        try:
            print("  Fetching Kalshi data...")
            client = KalshiClient()
            combined["kalshi"] = client.fetch_all_ncaa_data()
        except Exception as e:
            print(f"  WARNING: Kalshi fetch failed: {e}")

    # Polymarket
    if os.environ.get("POLYMARKET_API_KEY"):
        try:
            print("  Fetching Polymarket data...")
            client = PolymarketClient()
            combined["polymarket"] = client.fetch_all_ncaa_data()
        except Exception as e:
            print(f"  WARNING: Polymarket fetch failed: {e}")

    # Save raw data
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    market_dir = CFG.MARKET_DIR
    market_dir.mkdir(parents=True, exist_ok=True)

    # Save timestamped snapshot
    snapshot_path = market_dir / f"snapshot_{timestamp}.json"
    with open(snapshot_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)

    # Save latest
    latest_path = market_dir / "latest.json"
    with open(latest_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)

    print(f"  Market data saved to {snapshot_path}")
    return combined

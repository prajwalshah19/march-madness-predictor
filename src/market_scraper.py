"""Market data collection from Kalshi and Polymarket prediction markets."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

from src.config import CFG


def market_data_available() -> bool:
    """Check if any market API credentials are configured."""
    return bool(
        os.environ.get("KALSHI_API_KEY")
        or os.environ.get("POLYMARKET_API_KEY")
    )


class KalshiClient:
    """Client for the Kalshi prediction market API."""

    BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"

    def __init__(self) -> None:
        self.api_key = os.environ.get("KALSHI_API_KEY", "")
        self.api_secret = os.environ.get("KALSHI_API_SECRET", "")
        self.token: Optional[str] = None

        if not self.api_key:
            raise ValueError("KALSHI_API_KEY not set in environment")

    def _authenticate(self) -> None:
        """Authenticate and get session token."""
        resp = requests.post(
            f"{self.BASE_URL}/login",
            json={"email": self.api_key, "password": self.api_secret},
            timeout=30,
        )
        resp.raise_for_status()
        self.token = resp.json().get("token")

    def _headers(self) -> dict[str, str]:
        if not self.token:
            self._authenticate()
        return {"Authorization": f"Bearer {self.token}"}

    def get_ncaa_markets(self) -> list[dict[str, Any]]:
        """Fetch NCAA basketball markets."""
        markets = []
        cursor: Optional[str] = None

        for _ in range(10):  # Max 10 pages
            params: dict[str, Any] = {
                "limit": 100,
                "status": "open",
                "series_ticker": "NCAA",
            }
            if cursor:
                params["cursor"] = cursor

            resp = requests.get(
                f"{self.BASE_URL}/markets",
                headers=self._headers(),
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            batch = data.get("markets", [])
            markets.extend(batch)

            cursor = data.get("cursor")
            if not cursor or not batch:
                break

        return markets

    def get_orderbook(self, ticker: str) -> dict[str, Any]:
        """Fetch order book for a specific market."""
        resp = requests.get(
            f"{self.BASE_URL}/orderbook/{ticker}",
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("orderbook", {})

    def get_trades(self, ticker: str, limit: int = 100) -> list[dict[str, Any]]:
        """Fetch recent trades for a market."""
        resp = requests.get(
            f"{self.BASE_URL}/markets/{ticker}/trades",
            headers=self._headers(),
            params={"limit": limit},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("trades", [])

    def fetch_all_ncaa_data(self) -> dict[str, Any]:
        """Fetch all NCAA market data including order books and trades."""
        markets = self.get_ncaa_markets()
        print(f"    Found {len(markets)} Kalshi NCAA markets")

        enriched_markets = []
        for market in markets:
            ticker = market.get("ticker", "")
            try:
                market["orderbook"] = self.get_orderbook(ticker)
                market["recent_trades"] = self.get_trades(ticker)
                time.sleep(0.2)  # Rate limiting
            except Exception as e:
                market["orderbook"] = {}
                market["recent_trades"] = []
                print(f"    WARNING: Failed to fetch details for {ticker}: {e}")

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
        # Search for NCAA/March Madness markets
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
    if os.environ.get("KALSHI_API_KEY"):
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

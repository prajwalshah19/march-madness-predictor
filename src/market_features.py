"""Market feature extraction: whale detection, order book analysis, implied probabilities."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.config import CFG


def analyze_order_books(market_data: dict[str, Any]) -> pd.DataFrame:
    """Extract market signals from raw market data.

    Produces per-matchup features:
        - MarketProb: mid-price implied probability
        - WhaleSignal: directional whale flow [-1, +1]
        - DepthScore: normalized order book depth [0, 1]
        - Momentum: price momentum score
        - MarketConfidence: overall confidence in market signal [0, 1]

    Args:
        market_data: Raw market data from fetch_market_data().

    Returns:
        DataFrame with market signals per matchup.
    """
    signals: list[dict] = []

    # Process Kalshi markets
    kalshi_data = market_data.get("kalshi")
    if kalshi_data and kalshi_data.get("markets"):
        for market in kalshi_data["markets"]:
            signal = _process_kalshi_market(market)
            if signal:
                signals.append(signal)

    # Process Polymarket markets
    poly_data = market_data.get("polymarket")
    if poly_data and poly_data.get("markets"):
        for market in poly_data["markets"]:
            signal = _process_polymarket_market(market)
            if signal:
                signals.append(signal)

    if not signals:
        print("  No market signals extracted")
        return pd.DataFrame(columns=[
            "TeamA", "TeamB", "MarketProb", "WhaleSignal",
            "DepthScore", "Momentum", "MarketConfidence", "Source",
        ])

    df = pd.DataFrame(signals)

    # Normalize depth scores to [0, 1]
    if "DepthScore" in df.columns and df["DepthScore"].max() > 0:
        df["DepthScore"] = df["DepthScore"] / df["DepthScore"].max()

    print(f"  Extracted {len(df)} market signals")
    return df


def _process_kalshi_market(market: dict[str, Any]) -> Optional[dict]:
    """Extract signal from a single Kalshi market."""
    ticker = market.get("ticker", "")
    orderbook = market.get("orderbook", {})
    trades = market.get("recent_trades", [])

    # Get mid price
    yes_bid = orderbook.get("yes", {}).get("best_bid", 0) or 0
    yes_ask = orderbook.get("yes", {}).get("best_ask", 0) or 0
    if yes_bid > 0 and yes_ask > 0:
        mid_price = (yes_bid + yes_ask) / 2.0 / 100.0  # Convert cents to probability
    elif market.get("last_price"):
        mid_price = market["last_price"] / 100.0
    else:
        return None

    # Whale detection
    whale_signal = _detect_whales(trades)

    # Order book depth
    depth_score, book_imbalance = _compute_book_depth(orderbook)

    # Price momentum (would need time series; use available data)
    momentum = 0.0
    if market.get("previous_yes_price") and market.get("last_price"):
        prev = market["previous_yes_price"] / 100.0
        curr = market["last_price"] / 100.0
        momentum = curr - prev

    # Confidence
    volume = market.get("volume", 0) or 0
    confidence = _compute_confidence(depth_score, volume)

    return {
        "Ticker": ticker,
        "MarketProb": round(mid_price, 4),
        "WhaleSignal": round(whale_signal, 4),
        "DepthScore": round(depth_score, 4),
        "BookImbalance": round(book_imbalance, 4),
        "Momentum": round(momentum, 4),
        "MarketConfidence": round(confidence, 4),
        "Volume": volume,
        "Source": "kalshi",
        "TeamA": 0,  # Will be mapped by team name matching
        "TeamB": 0,
    }


def _process_polymarket_market(market: dict[str, Any]) -> Optional[dict]:
    """Extract signal from a single Polymarket market."""
    orderbook = market.get("orderbook", {})

    # Get price from tokens
    tokens = market.get("tokens", [])
    mid_price = None
    for token in tokens:
        if token.get("outcome") == "Yes":
            mid_price = token.get("price")
            break

    if mid_price is None:
        return None

    # Book depth
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    bid_depth = sum(float(b.get("size", 0)) for b in bids) if bids else 0
    ask_depth = sum(float(a.get("size", 0)) for a in asks) if asks else 0
    depth_score = bid_depth + ask_depth
    book_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0

    volume = float(market.get("volume", 0) or 0)
    confidence = _compute_confidence(depth_score, volume)

    return {
        "Ticker": market.get("condition_id", ""),
        "MarketProb": round(float(mid_price), 4),
        "WhaleSignal": 0.0,  # Need trade history for whale detection
        "DepthScore": round(depth_score, 4),
        "BookImbalance": round(book_imbalance, 4),
        "Momentum": 0.0,
        "MarketConfidence": round(confidence, 4),
        "Volume": volume,
        "Source": "polymarket",
        "TeamA": 0,
        "TeamB": 0,
    }


def _detect_whales(trades: list[dict[str, Any]]) -> float:
    """Detect whale activity in recent trades.

    Returns whale_signal in [-1, +1]:
        +1 = all whales buying YES
        -1 = all whales buying NO
    """
    if not trades:
        return 0.0

    sizes = [float(t.get("count", t.get("size", 0)) or 0) for t in trades]
    if not sizes or max(sizes) == 0:
        return 0.0

    median_size = float(np.median(sizes))
    if median_size == 0:
        return 0.0

    whale_threshold = 5.0 * median_size

    whale_yes_volume = 0.0
    whale_no_volume = 0.0

    for trade in trades:
        size = float(trade.get("count", trade.get("size", 0)) or 0)
        if size >= whale_threshold:
            side = trade.get("taker_side", trade.get("side", ""))
            if side in ("yes", "buy"):
                whale_yes_volume += size
            elif side in ("no", "sell"):
                whale_no_volume += size

    total_whale = whale_yes_volume + whale_no_volume
    if total_whale == 0:
        return 0.0

    return (whale_yes_volume - whale_no_volume) / total_whale


def _compute_book_depth(orderbook: dict[str, Any]) -> tuple[float, float]:
    """Compute order book depth score and imbalance.

    Returns (depth_score, book_imbalance).
    """
    # Kalshi format
    yes_data = orderbook.get("yes", {})
    no_data = orderbook.get("no", {})

    if isinstance(yes_data, dict):
        bid_depth = float(yes_data.get("total_bid_size", 0) or 0)
        ask_depth = float(yes_data.get("total_ask_size", 0) or 0)
    elif isinstance(yes_data, list):
        bid_depth = sum(float(o.get("size", 0)) for o in yes_data)
        ask_depth = sum(float(o.get("size", 0)) for o in (no_data if isinstance(no_data, list) else []))
    else:
        bid_depth = 0
        ask_depth = 0

    depth_score = bid_depth + ask_depth
    imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0

    return depth_score, imbalance


def _compute_confidence(depth_score: float, volume: float) -> float:
    """Compute overall market confidence score [0, 1].

    Higher depth and volume = higher confidence.
    """
    # Sigmoid-like scaling
    depth_conf = 1.0 - 1.0 / (1.0 + depth_score / 1000.0)
    volume_conf = 1.0 - 1.0 / (1.0 + volume / 10000.0)

    # Weighted combination
    return round(0.5 * depth_conf + 0.5 * volume_conf, 4)

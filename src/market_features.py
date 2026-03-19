"""Market feature extraction: whale detection, order book analysis, implied probabilities."""

from __future__ import annotations

import re
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.config import CFG
from src.team_matcher import TeamMatcher, _clean_team_name

# Columns for matchup signals (direct game probabilities)
MATCHUP_COLUMNS = [
    "TeamA", "TeamB", "MarketProb", "WhaleSignal",
    "DepthScore", "Momentum", "MarketConfidence", "Source",
]

# Columns for futures signals (championship / outright odds)
FUTURES_COLUMNS = [
    "TeamID", "Gender", "ChampionshipProb", "LogOdds",
    "MarketConfidence", "Volume", "Source",
]


def analyze_order_books(
    market_data: dict[str, Any],
    matcher: TeamMatcher,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract market signals from raw market data.

    Returns:
        (matchup_signals, futures_signals)
        matchup_signals: per-matchup features with resolved TeamIDs
        futures_signals: per-team championship odds with log-odds strength
    """
    matchup_signals: list[dict] = []
    futures_signals: list[dict] = []

    # Process Kalshi markets (active only — settled markets have stale prices)
    kalshi_data = market_data.get("kalshi")
    if kalshi_data and kalshi_data.get("markets"):
        # Group matchup game markets by event (Kalshi has one ticker per team side)
        game_markets: dict[str, list] = {}

        for market in kalshi_data["markets"]:
            if market.get("status") not in ("active", "open", None):
                continue

            ticker = market.get("ticker", "")
            parsed = matcher.parse_kalshi_market(market)

            if parsed["market_type"] == "matchup":
                # Group by event ticker (strip team suffix)
                event_key = market.get("event_ticker", ticker.rsplit("-", 1)[0])
                if event_key not in game_markets:
                    game_markets[event_key] = []
                game_markets[event_key].append((market, parsed))
            else:
                signal = _process_kalshi_futures(market, parsed)
                if signal:
                    futures_signals.append(signal)

        # Process paired matchup markets
        for event_key, sides in game_markets.items():
            signal = _process_kalshi_game_pair(sides, matcher)
            if signal:
                matchup_signals.append(signal)

    # Process Polymarket markets (if available, lower priority than Kalshi)
    poly_data = market_data.get("polymarket")
    if poly_data and poly_data.get("markets"):
        for market in poly_data["markets"]:
            signal = _process_polymarket_market(market, matcher)
            if signal:
                if signal.get("_market_type") == "futures":
                    del signal["_market_type"]
                    futures_signals.append(signal)
                else:
                    if "_market_type" in signal:
                        del signal["_market_type"]
                    matchup_signals.append(signal)

    # Build matchup DataFrame
    if matchup_signals:
        matchup_df = pd.DataFrame(matchup_signals)
        # Normalize depth scores to [0, 1]
        if "DepthScore" in matchup_df.columns and matchup_df["DepthScore"].max() > 0:
            matchup_df["DepthScore"] = matchup_df["DepthScore"] / matchup_df["DepthScore"].max()
        matched = (matchup_df["TeamA"] != 0).sum()
        print(f"  Matchup signals: {len(matchup_df)} markets, {matched} with resolved teams")
    else:
        matchup_df = pd.DataFrame(columns=MATCHUP_COLUMNS)
        print("  No matchup market signals extracted")

    # Build futures DataFrame
    if futures_signals:
        futures_df = pd.DataFrame(futures_signals)
        matched = (futures_df["TeamID"] != 0).sum()
        print(f"  Futures signals: {len(futures_df)} markets, {matched} with resolved teams")
    else:
        futures_df = pd.DataFrame(columns=FUTURES_COLUMNS)
        print("  No futures market signals extracted")

    return matchup_df, futures_df


def futures_to_matchup_prob(
    futures_df: pd.DataFrame,
    team_a: int,
    team_b: int,
) -> Optional[tuple[float, float]]:
    """Derive a matchup probability from futures (championship odds).

    Uses log-odds differential: if team A has higher championship odds,
    they're implied to be stronger. Converts the differential through
    a logistic function to get P(A beats B).

    Returns:
        (market_prob, confidence) or None if either team not in futures.
    """
    if futures_df.empty:
        return None

    row_a = futures_df[futures_df["TeamID"] == team_a]
    row_b = futures_df[futures_df["TeamID"] == team_b]

    if row_a.empty or row_b.empty:
        return None

    lo_a = float(row_a.iloc[0]["LogOdds"])
    lo_b = float(row_b.iloc[0]["LogOdds"])

    # Log-odds differential → logistic → probability
    # Scale factor: championship odds are concentrated (top teams ~20%, most < 1%),
    # so the raw log-odds diff is quite large. We compress it to get a game-level
    # probability. A factor of 0.3 maps well: diff of 2 → ~65% win probability.
    diff = lo_a - lo_b
    prob = 1.0 / (1.0 + np.exp(-0.3 * diff))

    # Confidence is lower than direct matchup markets: futures give team strength,
    # not head-to-head probability. Use the average of both teams' confidence.
    conf_a = float(row_a.iloc[0]["MarketConfidence"])
    conf_b = float(row_b.iloc[0]["MarketConfidence"])
    confidence = 0.5 * (conf_a + conf_b) * 0.6  # Discount by 0.6 vs direct markets

    return float(np.clip(prob, 0.05, 0.95)), float(confidence)


def _process_kalshi_game_pair(
    sides: list[tuple[dict[str, Any], dict[str, Any]]],
    matcher: "TeamMatcher",
) -> Optional[dict]:
    """Process a Kalshi matchup game with one or two side markets.

    Kalshi lists separate YES markets per team in a matchup game.
    We pair them to get both team IDs and the implied probability.
    """
    if not sides:
        return None

    # Collect team IDs and prices from each side
    team_ids = []
    prices = []
    best_market = None
    best_volume = 0

    for market, parsed in sides:
        # Each side's yes_sub_title is the team name for that side
        team_name = market.get("yes_sub_title", "")
        gender = parsed.get("gender", "M")
        team_id = matcher.match(team_name, gender) if team_name else None

        if team_id and team_id not in team_ids:
            team_ids.append(team_id)

        mid_price = _get_mid_price(market, market.get("orderbook", {}))
        if mid_price is not None:
            prices.append((team_id or 0, mid_price))

        vol = _to_float(market.get("volume_fp")) or 0
        if vol > best_volume:
            best_volume = vol
            best_market = market

    if len(team_ids) < 2 or not best_market:
        # Can't form a pair — try to get both teams from title parsing
        if sides:
            market, parsed = sides[0]
            ids = parsed.get("team_ids", [])
            team_ids = [i for i in ids if i != 0]
        if len(team_ids) < 2:
            return None

    team_a = min(team_ids[0], team_ids[1])
    team_b = max(team_ids[0], team_ids[1])

    # Find the price for team_a (lower ID)
    market_prob = 0.5
    for tid, price in prices:
        if tid == team_a:
            market_prob = price
            break
        elif tid == team_b:
            market_prob = 1.0 - price
            break

    orderbook = best_market.get("orderbook", {})
    trades = best_market.get("recent_trades", [])
    whale_signal = _detect_whales(trades)
    depth_score, book_imbalance = _compute_book_depth_from_market(best_market, orderbook)
    momentum = _compute_momentum(best_market)
    volume = _to_float(best_market.get("volume_fp")) or 0
    confidence = _compute_confidence(depth_score, volume)

    return {
        "Ticker": best_market.get("ticker", ""),
        "TeamA": team_a,
        "TeamB": team_b,
        "MarketProb": round(market_prob, 4),
        "WhaleSignal": round(whale_signal, 4),
        "DepthScore": round(depth_score, 4),
        "BookImbalance": round(book_imbalance, 4),
        "Momentum": round(momentum, 4),
        "MarketConfidence": round(confidence, 4),
        "Volume": volume,
        "Source": "kalshi",
    }


def _process_kalshi_matchup(
    market: dict[str, Any],
    parsed: dict[str, Any],
) -> Optional[dict]:
    """Extract signal from a Kalshi matchup market (direct game odds)."""
    team_ids = parsed.get("team_ids", [0, 0])
    if len(team_ids) < 2:
        return None

    team_a = team_ids[0]
    team_b = team_ids[1]

    orderbook = market.get("orderbook", {})
    trades = market.get("recent_trades", [])

    # Get mid price → P(team_a wins)
    mid_price = _get_mid_price(market, orderbook)
    if mid_price is None:
        return None

    whale_signal = _detect_whales(trades)
    depth_score, book_imbalance = _compute_book_depth_from_market(market, orderbook)
    momentum = _compute_momentum(market)
    volume = _to_float(market.get("volume_fp")) or _to_float(market.get("volume")) or 0
    confidence = _compute_confidence(depth_score, volume)

    # Ensure TeamA < TeamB (Kaggle convention: lower ID first)
    if team_a > team_b and team_a != 0 and team_b != 0:
        team_a, team_b = team_b, team_a
        mid_price = 1.0 - mid_price  # Flip probability
        whale_signal = -whale_signal

    return {
        "Ticker": market.get("ticker", ""),
        "TeamA": team_a,
        "TeamB": team_b,
        "MarketProb": round(mid_price, 4),
        "WhaleSignal": round(whale_signal, 4),
        "DepthScore": round(depth_score, 4),
        "BookImbalance": round(book_imbalance, 4),
        "Momentum": round(momentum, 4),
        "MarketConfidence": round(confidence, 4),
        "Volume": volume,
        "Source": "kalshi",
    }


def _process_kalshi_futures(
    market: dict[str, Any],
    parsed: dict[str, Any],
) -> Optional[dict]:
    """Extract signal from a Kalshi outright/futures market (championship odds)."""
    team_ids = parsed.get("team_ids", [0])
    if not team_ids:
        return None

    team_id = team_ids[0]
    gender = parsed.get("gender", "M")

    orderbook = market.get("orderbook", {})

    # Championship probability from mid-price
    mid_price = _get_mid_price(market, orderbook)
    if mid_price is None:
        return None

    # Log-odds: log(p / (1-p)). Clip to avoid log(0).
    prob_clipped = np.clip(mid_price, 0.001, 0.999)
    log_odds = float(np.log(prob_clipped / (1.0 - prob_clipped)))

    volume = _to_float(market.get("volume_fp")) or _to_float(market.get("volume")) or 0
    depth_score, _ = _compute_book_depth_from_market(market, orderbook)
    confidence = _compute_confidence(depth_score, volume)

    return {
        "Ticker": market.get("ticker", ""),
        "TeamID": team_id,
        "Gender": gender,
        "ChampionshipProb": round(mid_price, 4),
        "LogOdds": round(log_odds, 4),
        "MarketConfidence": round(confidence, 4),
        "Volume": volume,
        "Source": "kalshi",
    }


def _process_polymarket_market(
    market: dict[str, Any],
    matcher: TeamMatcher,
) -> Optional[dict]:
    """Extract signal from a Polymarket market.

    Polymarket markets could be matchup or outright — detect from title.
    """
    orderbook = market.get("orderbook", {})
    title = market.get("question", market.get("title", ""))

    # Detect gender
    gender = "M"
    if any(w in title.lower() for w in ["women", "wbb"]):
        gender = "W"

    # Get token price
    tokens = market.get("tokens", [])
    mid_price = None
    for token in tokens:
        if token.get("outcome") == "Yes":
            mid_price = token.get("price")
            break
    if mid_price is None:
        return None
    mid_price = float(mid_price)

    # Book depth
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    bid_depth = sum(float(b.get("size", 0)) for b in bids) if bids else 0
    ask_depth = sum(float(a.get("size", 0)) for a in asks) if asks else 0
    depth_score = bid_depth + ask_depth
    volume = float(market.get("volume", 0) or 0)
    confidence = _compute_confidence(depth_score, volume)

    # Determine if matchup or outright
    vs_match = re.search(
        r"(.+?)\s+(?:vs?\.?|versus)\s+(.+?)(?:\s*[-–—:|]|\s*$)",
        title, re.IGNORECASE,
    ) if title else None

    if vs_match:
        team_a_raw = _clean_team_name(vs_match.group(1))
        team_b_raw = _clean_team_name(vs_match.group(2))
        id_a = matcher.match(team_a_raw, gender) or 0
        id_b = matcher.match(team_b_raw, gender) or 0

        if id_a > id_b and id_a != 0 and id_b != 0:
            id_a, id_b = id_b, id_a
            mid_price = 1.0 - mid_price

        return {
            "Ticker": market.get("condition_id", ""),
            "TeamA": id_a,
            "TeamB": id_b,
            "MarketProb": round(mid_price, 4),
            "WhaleSignal": 0.0,
            "DepthScore": round(depth_score, 4),
            "BookImbalance": 0.0,
            "Momentum": 0.0,
            "MarketConfidence": round(confidence, 4),
            "Volume": volume,
            "Source": "polymarket",
            "_market_type": "matchup",
        }
    else:
        # Outright market
        to_win = re.search(r"(.+?)\s+to\s+win", title, re.IGNORECASE)
        team_name = _clean_team_name(to_win.group(1)) if to_win else None
        team_id = matcher.match(team_name, gender) if team_name else None

        prob_clipped = np.clip(mid_price, 0.001, 0.999)
        log_odds = float(np.log(prob_clipped / (1.0 - prob_clipped)))

        return {
            "Ticker": market.get("condition_id", ""),
            "TeamID": team_id or 0,
            "Gender": gender,
            "ChampionshipProb": round(mid_price, 4),
            "LogOdds": round(log_odds, 4),
            "MarketConfidence": round(confidence, 4),
            "Volume": volume,
            "Source": "polymarket",
            "_market_type": "futures",
        }


# ── Shared helpers ──────────────────────────────────────────────


def _get_mid_price(market: dict[str, Any], orderbook: dict[str, Any]) -> Optional[float]:
    """Extract mid-price probability from a market.

    Handles both dollar-denominated fields (current API) and cent-based fields.
    """
    # Current Kalshi API: dollar-denominated fields directly on the market
    yes_bid = _to_float(market.get("yes_bid_dollars"))
    yes_ask = _to_float(market.get("yes_ask_dollars"))
    if yes_bid is not None and yes_ask is not None and yes_bid > 0 and yes_ask > 0:
        return (yes_bid + yes_ask) / 2.0  # Already in [0, 1] range

    # Fallback: last trade price
    last = _to_float(market.get("last_price_dollars"))
    if last is None:
        last = _to_float(market.get("last_price"))
    if last is not None and last > 0:
        return last if last <= 1.0 else last / 100.0

    # Fallback: legacy orderbook
    ob_bid = _to_float(orderbook.get("yes", {}).get("best_bid"))
    ob_ask = _to_float(orderbook.get("yes", {}).get("best_ask"))
    if ob_bid is not None and ob_ask is not None and ob_bid > 0 and ob_ask > 0:
        mid = (ob_bid + ob_ask) / 2.0
        return mid if mid <= 1.0 else mid / 100.0

    return None


def _to_float(val: Any) -> Optional[float]:
    """Safely convert a value to float. Returns None for missing/empty values."""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _detect_whales(trades: list[dict[str, Any]]) -> float:
    """Detect whale activity in recent trades.

    Returns whale_signal in [-1, +1]:
        +1 = all whales buying YES
        -1 = all whales buying NO
    """
    if not trades:
        return 0.0

    sizes = [float(t.get("count_fp", t.get("count", t.get("size", 0))) or 0) for t in trades]
    if not sizes or max(sizes) == 0:
        return 0.0

    median_size = float(np.median(sizes))
    if median_size == 0:
        return 0.0

    whale_threshold = 5.0 * median_size

    whale_yes_volume = 0.0
    whale_no_volume = 0.0

    for trade in trades:
        size = float(trade.get("count_fp", trade.get("count", trade.get("size", 0))) or 0)
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


def _compute_book_depth_from_market(
    market: dict[str, Any], orderbook: dict[str, Any]
) -> tuple[float, float]:
    """Compute order book depth score and imbalance.

    Uses market-level size fields (current API) or falls back to orderbook dict.
    """
    # Current API: sizes on the market object itself
    bid_depth = _to_float(market.get("yes_bid_size_fp")) or 0
    ask_depth = _to_float(market.get("yes_ask_size_fp")) or 0

    # Fallback: legacy orderbook structure
    if bid_depth == 0 and ask_depth == 0:
        yes_data = orderbook.get("yes", {})
        no_data = orderbook.get("no", {})
        if isinstance(yes_data, dict):
            bid_depth = float(yes_data.get("total_bid_size", 0) or 0)
            ask_depth = float(yes_data.get("total_ask_size", 0) or 0)
        elif isinstance(yes_data, list):
            bid_depth = sum(float(o.get("size", 0)) for o in yes_data)
            ask_depth = sum(float(o.get("size", 0)) for o in (no_data if isinstance(no_data, list) else []))

    depth_score = bid_depth + ask_depth
    imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0

    return depth_score, imbalance


def _compute_momentum(market: dict[str, Any]) -> float:
    """Compute price momentum from previous close."""
    prev = _to_float(market.get("previous_price_dollars")) or _to_float(market.get("previous_yes_price"))
    curr = _to_float(market.get("last_price_dollars")) or _to_float(market.get("last_price"))
    if prev and curr:
        # Normalize to [0,1] if needed
        if prev > 1:
            prev /= 100.0
        if curr > 1:
            curr /= 100.0
        return curr - prev
    return 0.0


def _compute_confidence(depth_score: float, volume: float) -> float:
    """Compute overall market confidence score [0, 1].

    Higher depth and volume = higher confidence.
    """
    depth_conf = 1.0 - 1.0 / (1.0 + depth_score / 1000.0)
    volume_conf = 1.0 - 1.0 / (1.0 + volume / 10000.0)
    return round(0.5 * depth_conf + 0.5 * volume_conf, 4)



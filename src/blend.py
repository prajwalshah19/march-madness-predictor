"""Prediction blending layer: combine base model with market signals."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.config import CFG
from src.market_features import futures_to_matchup_prob


def blend_predictions(
    base_predictions: pd.DataFrame,
    market_signals: Optional[pd.DataFrame] = None,
    futures_signals: Optional[pd.DataFrame] = None,
    base_weight: float = CFG.BLEND_BASE_WEIGHT,
    market_weight: float = CFG.BLEND_MARKET_WEIGHT,
) -> pd.DataFrame:
    """Blend base model predictions with market signals.

    Blending strategy:
        - Direct matchup market: confidence-weighted blend (default 0.6/0.4)
        - Futures-derived signal: lower-confidence blend (~0.85/0.15)
        - No market data: use base prediction only
        - Whale signal adjusts weight toward market when strong

    Args:
        base_predictions: DataFrame with columns ID, Pred (or BasePred).
        market_signals: Matchup-level market features (direct game odds).
        futures_signals: Team-level futures (championship odds with LogOdds).
        base_weight: Default weight for base model predictions.
        market_weight: Default weight for market predictions.

    Returns:
        DataFrame with columns: ID, BasePred, MarketPred, BlendedPred,
        MarketConfidence, MarketSource
    """
    # Normalize column names
    result = base_predictions.copy()
    if "Pred" in result.columns and "BasePred" not in result.columns:
        result = result.rename(columns={"Pred": "BasePred"})

    # Initialize output columns
    result["MarketPred"] = np.nan
    result["BlendedPred"] = result["BasePred"]
    result["MarketConfidence"] = 0.0
    result["MarketSource"] = ""

    has_matchup = market_signals is not None and not market_signals.empty
    has_futures = futures_signals is not None and not futures_signals.empty

    if not has_matchup and not has_futures:
        print("  No market signals available, using base predictions only")
        result["Pred"] = result["BlendedPred"].clip(CFG.CLIP_LOW, CFG.CLIP_HIGH)
        return result

    # Build matchup lookup by (lower_team_id, higher_team_id)
    matchup_lookup = _build_market_lookup(market_signals) if has_matchup else {}

    direct_count = 0
    futures_count = 0

    for idx, row in result.iterrows():
        match_id = row["ID"]
        parts = match_id.split("_")
        if len(parts) != 3:
            continue

        team_a = int(parts[1])
        team_b = int(parts[2])
        key = (min(team_a, team_b), max(team_a, team_b))

        base_pred = row["BasePred"]
        market_prob = None
        confidence = 0.0
        whale_signal = 0.0
        source = ""

        # Priority 1: Direct matchup market
        if key in matchup_lookup:
            market = matchup_lookup[key]
            market_prob = market["MarketProb"]
            confidence = market["MarketConfidence"]
            whale_signal = market.get("WhaleSignal", 0)
            source = "matchup"
            direct_count += 1

        # Priority 2: Futures-derived probability (if no direct market)
        elif has_futures:
            derived = futures_to_matchup_prob(futures_signals, team_a, team_b)
            if derived is not None:
                market_prob, confidence = derived
                source = "futures"
                futures_count += 1

        if market_prob is None:
            continue

        result.at[idx, "MarketPred"] = market_prob
        result.at[idx, "MarketConfidence"] = confidence
        result.at[idx, "MarketSource"] = source

        # Determine blend weights based on confidence and source
        if source == "futures":
            # Futures-derived: always conservative blend
            w_base = CFG.BLEND_LOW_CONFIDENCE_BASE
            w_market = CFG.BLEND_LOW_CONFIDENCE_MARKET
        elif confidence >= 0.5:
            # High confidence direct market: use standard weights
            w_base = base_weight - 0.1 * (confidence - 0.5)
            w_market = 1.0 - w_base
        else:
            # Low confidence direct market: conservative blend
            w_base = CFG.BLEND_LOW_CONFIDENCE_BASE
            w_market = CFG.BLEND_LOW_CONFIDENCE_MARKET

        # Whale signal adjustment (matchup markets only)
        if source == "matchup" and abs(whale_signal) > 0.5:
            whale_boost = 0.05 * abs(whale_signal)
            w_market += whale_boost
            w_base -= whale_boost

        # Ensure weights are valid
        w_base = max(0.1, min(0.95, w_base))
        w_market = 1.0 - w_base

        # Blend
        blended = w_base * base_pred + w_market * market_prob
        result.at[idx, "BlendedPred"] = blended

    # Clip all predictions
    result["BlendedPred"] = result["BlendedPred"].clip(CFG.CLIP_LOW, CFG.CLIP_HIGH)
    result["Pred"] = result["BlendedPred"]

    total = direct_count + futures_count
    print(f"  Blended {total}/{len(result)} matchups with market data "
          f"({direct_count} direct, {futures_count} futures-derived)")
    return result


def _build_market_lookup(market_signals: pd.DataFrame) -> dict[tuple[int, int], dict]:
    """Build a lookup dictionary from matchup market signals.

    Returns dict keyed by (lower_team_id, higher_team_id).
    Only includes entries where both teams were resolved.
    """
    lookup: dict[tuple[int, int], dict] = {}

    for _, row in market_signals.iterrows():
        team_a = int(row.get("TeamA", 0))
        team_b = int(row.get("TeamB", 0))

        if team_a == 0 or team_b == 0:
            continue

        key = (min(team_a, team_b), max(team_a, team_b))

        # If multiple signals for same matchup, prefer higher confidence
        existing = lookup.get(key)
        if existing and existing.get("MarketConfidence", 0) >= row.get("MarketConfidence", 0):
            continue

        lookup[key] = row.to_dict()

    return lookup

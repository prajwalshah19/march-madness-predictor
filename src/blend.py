"""Prediction blending layer: combine base model with market signals."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.config import CFG


def blend_predictions(
    base_predictions: pd.DataFrame,
    market_signals: Optional[pd.DataFrame] = None,
    futures_signals: Optional[pd.DataFrame] = None,
    base_weight: float = CFG.BLEND_BASE_WEIGHT,
    market_weight: float = CFG.BLEND_MARKET_WEIGHT,
) -> pd.DataFrame:
    """Blend base model predictions with market signals.

    Vectorized implementation — no row-level Python loops.
    """
    result = base_predictions.copy()
    if "Pred" in result.columns and "BasePred" not in result.columns:
        result = result.rename(columns={"Pred": "BasePred"})

    result["MarketPred"] = np.nan
    result["BlendedPred"] = result["BasePred"].values.copy()
    result["MarketConfidence"] = 0.0
    result["MarketSource"] = ""

    has_matchup = market_signals is not None and not market_signals.empty
    has_futures = futures_signals is not None and not futures_signals.empty

    if not has_matchup and not has_futures:
        print("  No market signals available, using base predictions only")
        result["Pred"] = result["BlendedPred"].clip(CFG.CLIP_LOW, CFG.CLIP_HIGH)
        return result

    # Parse team IDs from match IDs vectorially
    split = result["ID"].str.split("_", expand=True)
    if split.shape[1] < 3:
        result["Pred"] = result["BlendedPred"].clip(CFG.CLIP_LOW, CFG.CLIP_HIGH)
        return result
    team_a = split[1].astype(int).values
    team_b = split[2].astype(int).values

    n = len(result)
    market_prob = np.full(n, np.nan)
    confidence = np.zeros(n)
    whale_signal = np.zeros(n)
    source = np.empty(n, dtype=object)
    source[:] = ""

    direct_count = 0

    # Priority 1: Direct matchup markets
    if has_matchup:
        matchup_lookup = _build_market_lookup(market_signals)
        if matchup_lookup:
            lo = np.minimum(team_a, team_b)
            hi = np.maximum(team_a, team_b)
            for i in range(n):
                key = (int(lo[i]), int(hi[i]))
                if key in matchup_lookup:
                    m = matchup_lookup[key]
                    market_prob[i] = m["MarketProb"]
                    confidence[i] = m["MarketConfidence"]
                    whale_signal[i] = m.get("WhaleSignal", 0)
                    source[i] = "matchup"
                    direct_count += 1

    # Priority 2: Futures-derived probabilities (vectorized)
    futures_count = 0
    if has_futures:
        futures_mask = np.isnan(market_prob)  # Only where no direct matchup
        if futures_mask.any():
            # Build log-odds and confidence lookup dicts from futures
            resolved = futures_signals[futures_signals["TeamID"] != 0]
            if not resolved.empty:
                # Keep best confidence per team
                best_idx = resolved.groupby("TeamID")["MarketConfidence"].idxmax()
                best = resolved.loc[best_idx]
                lo_dict = dict(zip(best["TeamID"].astype(int), best["LogOdds"].astype(float)))
                conf_dict = dict(zip(best["TeamID"].astype(int), best["MarketConfidence"].astype(float)))

                # Vectorized lookup
                lo_a = np.array([lo_dict.get(int(t), np.nan) for t in team_a])
                lo_b = np.array([lo_dict.get(int(t), np.nan) for t in team_b])
                conf_a = np.array([conf_dict.get(int(t), 0.0) for t in team_a])
                conf_b = np.array([conf_dict.get(int(t), 0.0) for t in team_b])

                both_exist = futures_mask & ~np.isnan(lo_a) & ~np.isnan(lo_b)
                if both_exist.any():
                    diff = lo_a[both_exist] - lo_b[both_exist]
                    prob = 1.0 / (1.0 + np.exp(-0.3 * diff))
                    market_prob[both_exist] = np.clip(prob, 0.05, 0.95)
                    confidence[both_exist] = 0.5 * (conf_a[both_exist] + conf_b[both_exist]) * 0.6
                    source[both_exist] = "futures"
                    futures_count = int(both_exist.sum())

    # Compute blend weights vectorially
    has_market = ~np.isnan(market_prob)
    is_futures = source == "futures"
    is_matchup = source == "matchup"
    hi_conf = confidence >= 0.5

    w_base = np.full(n, 1.0)
    # Futures: always conservative
    w_base[is_futures] = CFG.BLEND_LOW_CONFIDENCE_BASE
    # High-confidence matchup
    matchup_hi = is_matchup & hi_conf
    w_base[matchup_hi] = base_weight - 0.1 * (confidence[matchup_hi] - 0.5)
    # Low-confidence matchup
    matchup_lo = is_matchup & ~hi_conf
    w_base[matchup_lo] = CFG.BLEND_LOW_CONFIDENCE_BASE

    # Whale boost (matchup only)
    whale_active = is_matchup & (np.abs(whale_signal) > 0.5)
    whale_boost = 0.05 * np.abs(whale_signal[whale_active])
    w_base[whale_active] -= whale_boost

    # Clip weights
    w_base = np.clip(w_base, 0.1, 0.95)
    w_market = 1.0 - w_base

    # Blend where we have market data
    blended = result["BasePred"].values.copy()
    blended[has_market] = (
        w_base[has_market] * blended[has_market]
        + w_market[has_market] * market_prob[has_market]
    )

    result["MarketPred"] = market_prob
    result["MarketConfidence"] = confidence
    result["MarketSource"] = source
    result["BlendedPred"] = np.clip(blended, CFG.CLIP_LOW, CFG.CLIP_HIGH)
    result["Pred"] = result["BlendedPred"]

    total = direct_count + futures_count
    print(f"  Blended {total}/{n} matchups with market data "
          f"({direct_count} direct, {futures_count} futures-derived)")
    return result


def _build_market_lookup(market_signals: pd.DataFrame) -> dict[tuple[int, int], dict]:
    """Build lookup from matchup signals. Only resolved pairs."""
    valid = market_signals[(market_signals["TeamA"] != 0) & (market_signals["TeamB"] != 0)]
    if valid.empty:
        return {}

    # Sort by confidence descending, keep first (highest) per pair
    valid = valid.sort_values("MarketConfidence", ascending=False)
    lookup: dict[tuple[int, int], dict] = {}
    for rec in valid.to_dict("records"):
        key = (min(int(rec["TeamA"]), int(rec["TeamB"])),
               max(int(rec["TeamA"]), int(rec["TeamB"])))
        if key not in lookup:
            lookup[key] = rec
    return lookup

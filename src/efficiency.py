"""Adjusted offensive and defensive efficiency ratings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CFG
from src.data_loader import DataLoader


def _compute_possessions(fga: int, ora: int, to: int, fta: int) -> float:
    """Estimate possessions for one team in a game.

    Formula: FGA - OR + TO + 0.475 * FTA
    """
    return fga - ora + to + CFG.FTA_COEFFICIENT * fta


def _compute_game_stats(games: pd.DataFrame) -> pd.DataFrame:
    """Compute per-game efficiency stats from detailed results.

    Returns one row per team per game with offensive/defensive efficiency.
    Fully vectorized — no Python loops.
    """
    # Possession estimates for both sides
    w_poss = games["WFGA"] - games["WOR"] + games["WTO"] + CFG.FTA_COEFFICIENT * games["WFTA"]
    l_poss = games["LFGA"] - games["LOR"] + games["LTO"] + CFG.FTA_COEFFICIENT * games["LFTA"]
    avg_poss = (w_poss + l_poss) / 2.0

    # Filter out zero-possession games
    valid = avg_poss > 0
    g = games[valid]
    ap = avg_poss[valid]

    gender = g["Gender"] if "Gender" in g.columns else pd.Series("M", index=g.index)

    # Winner rows
    winners = pd.DataFrame({
        "Season": g["Season"].values,
        "DayNum": g["DayNum"].values,
        "Gender": gender.values,
        "TeamID": g["WTeamID"].values,
        "OppID": g["LTeamID"].values,
        "OffEff": (g["WScore"].values / ap.values * 100),
        "DefEff": (g["LScore"].values / ap.values * 100),
        "Possessions": ap.values,
    })

    # Loser rows
    losers = pd.DataFrame({
        "Season": g["Season"].values,
        "DayNum": g["DayNum"].values,
        "Gender": gender.values,
        "TeamID": g["LTeamID"].values,
        "OppID": g["WTeamID"].values,
        "OffEff": (g["LScore"].values / ap.values * 100),
        "DefEff": (g["WScore"].values / ap.values * 100),
        "Possessions": ap.values,
    })

    return pd.concat([winners, losers], ignore_index=True)


def _get_recency_weights(day_nums: pd.Series) -> pd.Series:
    """Weight recent games 1.5x (last 30% of season by DayNum)."""
    if day_nums.empty:
        return day_nums
    max_day = day_nums.max()
    min_day = day_nums.min()
    day_range = max_day - min_day
    if day_range == 0:
        return pd.Series(1.0, index=day_nums.index)
    threshold = min_day + 0.7 * day_range
    return pd.Series(np.where(day_nums >= threshold, 1.5, 1.0), index=day_nums.index)


def compute_efficiency(loader: DataLoader, elo_ratings: pd.DataFrame) -> pd.DataFrame:
    """Compute adjusted offensive and defensive efficiency for all teams.

    Args:
        loader: Data loader with Kaggle CSVs.
        elo_ratings: DataFrame with Elo ratings (Season, TeamID, Gender, EloPreTourney, EloPostSeason).

    Returns:
        DataFrame with columns: Season, TeamID, Gender, EloPreTourney, EloPostSeason,
        AdjOff, AdjDef, NetEff, Tempo
    """
    detailed = loader.get_regular_season_detailed()
    if detailed.empty:
        print("  WARNING: No detailed results available, skipping efficiency")
        return elo_ratings.copy()

    # Compute per-game stats
    game_stats = _compute_game_stats(detailed)

    # Try to load Massey ordinals for SOS adjustment
    try:
        massey = loader.get_massey()
        has_massey = True
    except KeyError:
        massey = pd.DataFrame()
        has_massey = False

    results = []

    for (season, gender), group in game_stats.groupby(["Season", "Gender"]):
        # Compute recency-weighted averages per team
        team_stats = []
        for team_id, team_games in group.groupby("TeamID"):
            weights = _get_recency_weights(team_games["DayNum"])
            total_w = weights.sum()

            avg_off = (team_games["OffEff"] * weights).sum() / total_w
            avg_def = (team_games["DefEff"] * weights).sum() / total_w
            avg_tempo = (team_games["Possessions"] * weights).sum() / total_w

            # SOS adjustment using Massey ordinals
            adj_off = avg_off
            adj_def = avg_def

            if has_massey and not massey.empty:
                season_massey = massey[massey["Season"] == season]
                if not season_massey.empty:
                    # Get latest Massey rank for each opponent
                    latest_massey = (
                        season_massey
                        .sort_values("RankingDayNum")
                        .groupby("TeamID")
                        .last()
                        .reset_index()
                    )
                    # Use composite ranking (average across systems)
                    avg_ranks = latest_massey.groupby("TeamID")["OrdinalRank"].mean()
                    opp_ids = team_games["OppID"]
                    opp_ranks = opp_ids.map(avg_ranks).dropna()

                    if len(opp_ranks) > 0:
                        avg_opp_rank = opp_ranks.mean()
                        # Median rank is ~175 for ~350 teams
                        # Positive adjustment if opponents are tough (low rank = good)
                        median_rank = 175.0
                        sos_factor = 1.0 + (median_rank - avg_opp_rank) / median_rank * 0.1
                        adj_off = avg_off * sos_factor
                        adj_def = avg_def / sos_factor  # Lower is better for defense

            team_stats.append({
                "Season": season,
                "TeamID": team_id,
                "Gender": gender,
                "AdjOff": round(adj_off, 2),
                "AdjDef": round(adj_def, 2),
                "NetEff": round(adj_off - adj_def, 2),
                "Tempo": round(avg_tempo, 1),
            })

        results.extend(team_stats)

    eff_df = pd.DataFrame(results)

    # Merge with Elo ratings
    merged = elo_ratings.merge(
        eff_df,
        on=["Season", "TeamID", "Gender"],
        how="left",
    )

    # Fill missing efficiency (teams without detailed data)
    for col in ["AdjOff", "AdjDef", "NetEff", "Tempo"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(merged[col].median() if not merged[col].isna().all() else 0)

    print(f"  Computed efficiency for {len(eff_df)} team-seasons")
    return merged

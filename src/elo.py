"""Elo rating system for NCAA basketball teams."""

from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd

from src.config import CFG
from src.data_loader import DataLoader


def _expected_score(rating_a: float, rating_b: float) -> float:
    """Compute expected score for team A against team B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _mov_multiplier(score_diff: int, elo_diff: float) -> float:
    """Margin of victory multiplier that dampens blowouts against weak teams."""
    return math.log(abs(score_diff) + 1) * (2.2 / (elo_diff * 0.001 + 2.2))


def compute_elo(loader: DataLoader) -> pd.DataFrame:
    """Compute Elo ratings for all teams across all available seasons.

    Returns a DataFrame with columns:
        Season, TeamID, Gender, EloPreTourney, EloPostSeason
    """
    K = CFG.ELO_K
    INITIAL = CFG.ELO_INITIAL
    REGRESS = CFG.ELO_SEASON_REGRESS
    HOME_ADV = CFG.ELO_HOME_ADVANTAGE

    # Track current Elo for each (gender, team)
    elos: dict[str, dict[int, float]] = {"M": defaultdict(lambda: INITIAL), "W": defaultdict(lambda: INITIAL)}

    # Collect snapshots
    snapshots: list[dict] = []

    for gender in ["M", "W"]:
        # Get all regular season + tourney games
        reg = loader.get_regular_season_compact(gender=gender)
        tourney = loader.get_tourney_compact(gender=gender)

        if reg.empty:
            continue

        all_seasons = sorted(reg["Season"].unique())

        for season in all_seasons:
            # Season-start regression toward mean
            for team_id in list(elos[gender].keys()):
                old = elos[gender][team_id]
                elos[gender][team_id] = INITIAL + REGRESS * (old - INITIAL)

            # Process regular season games chronologically
            season_reg = reg[reg["Season"] == season].sort_values("DayNum")
            _process_games(season_reg, elos[gender], K, HOME_ADV)

            # Snapshot pre-tournament Elo
            pre_tourney_elos = dict(elos[gender])

            # Process tournament games
            season_tourney = tourney[tourney["Season"] == season].sort_values("DayNum")
            if not season_tourney.empty:
                _process_games(season_tourney, elos[gender], K, home_advantage=0)

            # Record snapshots for all teams that played this season
            teams_this_season = set(season_reg["WTeamID"]) | set(season_reg["LTeamID"])
            for team_id in teams_this_season:
                snapshots.append({
                    "Season": season,
                    "TeamID": team_id,
                    "Gender": gender,
                    "EloPreTourney": pre_tourney_elos.get(team_id, INITIAL),
                    "EloPostSeason": elos[gender].get(team_id, INITIAL),
                })

    df = pd.DataFrame(snapshots)
    print(f"  Computed Elo for {len(df)} team-seasons")
    return df


def _process_games(
    games: pd.DataFrame,
    elos: dict[int, float],
    k: int,
    home_advantage: int,
) -> None:
    """Process a set of games and update Elo ratings in-place."""
    for _, row in games.iterrows():
        w_id = int(row["WTeamID"])
        l_id = int(row["LTeamID"])
        w_score = int(row["WScore"])
        l_score = int(row["LScore"])
        w_loc = row.get("WLoc", "N")

        # Home court adjustment
        if home_advantage > 0:
            if w_loc == "H":
                w_elo_adj = elos[w_id] + home_advantage
                l_elo_adj = elos[l_id]
            elif w_loc == "A":
                w_elo_adj = elos[w_id]
                l_elo_adj = elos[l_id] + home_advantage
            else:  # Neutral
                w_elo_adj = elos[w_id]
                l_elo_adj = elos[l_id]
        else:
            w_elo_adj = elos[w_id]
            l_elo_adj = elos[l_id]

        # Expected scores
        exp_w = _expected_score(w_elo_adj, l_elo_adj)

        # Margin of victory multiplier
        score_diff = w_score - l_score
        elo_diff = w_elo_adj - l_elo_adj
        mov_mult = _mov_multiplier(score_diff, elo_diff)

        # Update ratings
        update = k * mov_mult * (1 - exp_w)
        elos[w_id] += update
        elos[l_id] -= update

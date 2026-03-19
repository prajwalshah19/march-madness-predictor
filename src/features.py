"""Tournament-specific features computed from detailed box scores."""

from __future__ import annotations

import pandas as pd

from src.config import CFG
from src.data_loader import DataLoader


def compute_tournament_features(
    loader: DataLoader,
    team_ratings: pd.DataFrame,
) -> pd.DataFrame:
    """Compute tournament-relevant features per team per season.

    Features:
        1. Opp3PtPct: Opponent 3pt shooting % (lower = better 3pt defense)
        2. CloseGameTOMargin: Turnover margin in close games (≤5 pts)
        3. OffRebRate: Offensive rebounding rate
        4. LateFTPct: Free throw % in last 25% of season

    Args:
        loader: Data loader.
        team_ratings: Existing ratings DataFrame to merge into.

    Returns:
        Updated team_ratings with new feature columns.
    """
    detailed = loader.get_regular_season_detailed()
    if detailed.empty:
        print("  WARNING: No detailed results, skipping tournament features")
        return team_ratings.copy()

    all_features: list[dict] = []

    for (season, gender), season_games in detailed.groupby(["Season", "Gender"]):
        teams_in_season = set(season_games["WTeamID"]) | set(season_games["LTeamID"])

        for team_id in teams_in_season:
            feats = _compute_team_features(team_id, season_games)
            feats["Season"] = season
            feats["TeamID"] = team_id
            feats["Gender"] = gender
            all_features.append(feats)

    feat_df = pd.DataFrame(all_features)

    # Merge with existing ratings
    merged = team_ratings.merge(
        feat_df[["Season", "TeamID", "Gender", "Opp3PtPct", "CloseGameTOMargin", "OffRebRate", "LateFTPct"]],
        on=["Season", "TeamID", "Gender"],
        how="left",
    )

    # Fill missing values with column medians
    for col in ["Opp3PtPct", "CloseGameTOMargin", "OffRebRate", "LateFTPct"]:
        if col in merged.columns:
            median_val = merged[col].median()
            merged[col] = merged[col].fillna(median_val if pd.notna(median_val) else 0)

    print(f"  Computed tournament features for {len(feat_df)} team-seasons")
    return merged


def _compute_team_features(team_id: int, season_games: pd.DataFrame) -> dict:
    """Compute all tournament features for one team in one season."""
    # Separate games where team won vs lost
    wins = season_games[season_games["WTeamID"] == team_id]
    losses = season_games[season_games["LTeamID"] == team_id]

    # === 1. Opponent 3pt% (Opp3PtPct) ===
    # When team won: opponent is losing team (L prefix)
    opp_fgm3_w = wins["LFGM3"].sum() if not wins.empty else 0
    opp_fga3_w = wins["LFGA3"].sum() if not wins.empty else 0
    # When team lost: opponent is winning team (W prefix)
    opp_fgm3_l = losses["WFGM3"].sum() if not losses.empty else 0
    opp_fga3_l = losses["WFGA3"].sum() if not losses.empty else 0

    total_opp_fgm3 = opp_fgm3_w + opp_fgm3_l
    total_opp_fga3 = opp_fga3_w + opp_fga3_l
    opp_3pt_pct = total_opp_fgm3 / total_opp_fga3 if total_opp_fga3 > 0 else 0.33

    # === 2. Turnover margin in close games (CloseGameTOMargin) ===
    close_wins = wins[(wins["WScore"] - wins["LScore"]).abs() <= 5] if not wins.empty else wins
    close_losses = losses[(losses["WScore"] - losses["LScore"]).abs() <= 5] if not losses.empty else losses

    close_to_vals = pd.concat([
        close_wins["LTO"] - close_wins["WTO"],    # opp TO - team TO
        close_losses["WTO"] - close_losses["LTO"],
    ]) if not (close_wins.empty and close_losses.empty) else pd.Series(dtype=float)

    if len(close_to_vals) >= 5:
        close_to_margin = close_to_vals.mean()
    else:
        team_to_total = wins["WTO"].sum() + losses["LTO"].sum()
        opp_to_total = wins["LTO"].sum() + losses["WTO"].sum()
        n_games = len(wins) + len(losses)
        close_to_margin = (opp_to_total - team_to_total) / n_games if n_games > 0 else 0

    # === 3. Offensive rebounding rate (OffRebRate) ===
    # Team OR / (Team OR + Opp DR)
    team_or_w = wins["WOR"].sum() if not wins.empty else 0
    opp_dr_w = wins["LDR"].sum() if not wins.empty else 0
    team_or_l = losses["LOR"].sum() if not losses.empty else 0
    opp_dr_l = losses["WDR"].sum() if not losses.empty else 0

    total_team_or = team_or_w + team_or_l
    total_opp_dr = opp_dr_w + opp_dr_l
    off_reb_rate = total_team_or / (total_team_or + total_opp_dr) if (total_team_or + total_opp_dr) > 0 else 0.3

    # === 4. Late-season FT% (LateFTPct) ===
    all_games = pd.concat([wins, losses]) if not losses.empty else wins
    if all_games.empty:
        late_ft_pct = 0.7
    else:
        max_day = all_games["DayNum"].max()
        min_day = all_games["DayNum"].min()
        day_range = max_day - min_day
        late_threshold = min_day + 0.75 * day_range if day_range > 0 else min_day

        late_wins = wins[wins["DayNum"] >= late_threshold] if not wins.empty else pd.DataFrame()
        late_losses = losses[losses["DayNum"] >= late_threshold] if not losses.empty else pd.DataFrame()

        ftm = (late_wins["WFTM"].sum() if not late_wins.empty else 0) + \
              (late_losses["LFTM"].sum() if not late_losses.empty else 0)
        fta = (late_wins["WFTA"].sum() if not late_wins.empty else 0) + \
              (late_losses["LFTA"].sum() if not late_losses.empty else 0)

        late_ft_pct = ftm / fta if fta > 0 else 0.7

    return {
        "Opp3PtPct": round(opp_3pt_pct, 4),
        "CloseGameTOMargin": round(close_to_margin, 4),
        "OffRebRate": round(off_reb_rate, 4),
        "LateFTPct": round(late_ft_pct, 4),
    }

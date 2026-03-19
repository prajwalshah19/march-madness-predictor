"""Mock bracket simulator with ASCII output and Monte Carlo advancement probabilities."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import CFG
from src.data_loader import DataLoader


# Standard bracket matchup order within a region (seed numbers)
FIRST_ROUND_MATCHUPS = [
    (1, 16), (8, 9), (5, 12), (4, 13),
    (2, 15), (7, 10), (6, 11), (3, 14),
]

ROUND_NAMES = ["R64", "R32", "S16", "E8", "F4", "Championship", "Winner"]


def _build_prediction_lookup(predictions: pd.DataFrame) -> dict[tuple[int, int], float]:
    """Build a fast lookup dict from predictions DataFrame. Key: (low_id, high_id) -> P(low wins)."""
    lookup: dict[tuple[int, int], float] = {}
    for _, row in predictions.iterrows():
        parts = str(row["ID"]).split("_")
        if len(parts) == 3:
            low, high = int(parts[1]), int(parts[2])
            lookup[(low, high)] = float(row["Pred"])
    return lookup


def _get_prediction_fast(
    lookup: dict[tuple[int, int], float],
    team_a_id: int,
    team_b_id: int,
) -> float:
    """Look up P(team_a wins) from precomputed lookup dict."""
    low = min(team_a_id, team_b_id)
    high = max(team_a_id, team_b_id)
    pred = lookup.get((low, high), 0.5)
    return pred if team_a_id == low else 1.0 - pred


def simulate_bracket(
    loader: DataLoader,
    predictions: pd.DataFrame,
    season: int = CFG.TARGET_SEASON,
    gender: str = "M",
    n_simulations: int = CFG.MONTE_CARLO_SIMULATIONS,
) -> dict:
    """Simulate tournament bracket using Monte Carlo.

    Returns dict with:
        - chalk_bracket: list of (round, winner_id, loser_id, prob) for chalk
        - advancement_probs: dict of team_id -> {round_name: probability}
        - champion_probs: dict of team_id -> championship probability
    """
    seeds = loader.get_seeds(season=season, gender=gender)
    if seeds.empty:
        return {"chalk_bracket": [], "advancement_probs": {}, "champion_probs": {}}

    # Build region brackets
    regions = sorted(seeds["Region"].unique())

    # Track advancement counts across simulations
    advancement_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # Also track chalk bracket (most probable outcome)
    chalk_results: list[tuple[str, int, int, float]] = []

    # Get team names for display
    try:
        teams = loader.get_teams()
        team_names = dict(zip(teams["TeamID"], teams["TeamName"]))
    except Exception:
        team_names = {}

    # Build fast prediction lookup
    lookup = _build_prediction_lookup(predictions)

    # Run Monte Carlo simulations
    rng = np.random.default_rng(42)

    for sim in range(n_simulations):
        is_chalk = sim == 0  # First simulation tracks chalk

        # Simulate each region to get Final Four
        final_four = []

        for region in regions[:4]:  # Max 4 regions
            region_seeds = seeds[seeds["Region"] == region].sort_values("SeedNum")
            if region_seeds.empty:
                continue

            # Build seed->team mapping
            seed_teams: dict[int, int] = {}
            for _, row in region_seeds.iterrows():
                seed_num = int(row["SeedNum"])
                if seed_num not in seed_teams:  # Skip play-in duplicates
                    seed_teams[seed_num] = int(row["TeamID"])

            # First round
            round_name = "R64"
            winners = []
            for s1, s2 in FIRST_ROUND_MATCHUPS:
                t1 = seed_teams.get(s1)
                t2 = seed_teams.get(s2)
                if t1 is None or t2 is None:
                    winners.append(t1 or t2)
                    continue

                # Both teams made R64
                advancement_counts[t1]["R64"] += 1
                advancement_counts[t2]["R64"] += 1

                prob_t1 = _get_prediction_fast(lookup,t1, t2)
                if is_chalk:
                    winner = t1 if prob_t1 >= 0.5 else t2
                    chalk_results.append(("R64", winner, t2 if winner == t1 else t1, max(prob_t1, 1 - prob_t1)))
                else:
                    winner = t1 if rng.random() < prob_t1 else t2

                advancement_counts[winner]["R32"] += 1
                winners.append(winner)

            # Subsequent rounds within region
            current = winners
            for round_idx, round_name in enumerate(["R32", "S16", "E8"]):
                next_round = []
                for i in range(0, len(current), 2):
                    if i + 1 >= len(current):
                        next_round.append(current[i])
                        continue
                    t1, t2 = current[i], current[i + 1]
                    if t1 is None or t2 is None:
                        next_round.append(t1 or t2)
                        continue

                    prob_t1 = _get_prediction_fast(lookup,t1, t2)
                    if is_chalk:
                        winner = t1 if prob_t1 >= 0.5 else t2
                        chalk_results.append((round_name, winner, t2 if winner == t1 else t1, max(prob_t1, 1 - prob_t1)))
                    else:
                        winner = t1 if rng.random() < prob_t1 else t2

                    next_name = {"R32": "S16", "S16": "E8", "E8": "F4"}[round_name]
                    advancement_counts[winner][next_name] += 1
                    next_round.append(winner)

                current = next_round

            if current:
                final_four.append(current[0])

        # Final Four: semi-finals
        if len(final_four) >= 4:
            # Semis: region 0 vs region 1, region 2 vs region 3
            semis = [(final_four[0], final_four[1]), (final_four[2], final_four[3])]
            finalists = []
            for t1, t2 in semis:
                if t1 is None or t2 is None:
                    finalists.append(t1 or t2)
                    continue
                prob_t1 = _get_prediction_fast(lookup,t1, t2)
                if is_chalk:
                    winner = t1 if prob_t1 >= 0.5 else t2
                    chalk_results.append(("F4", winner, t2 if winner == t1 else t1, max(prob_t1, 1 - prob_t1)))
                else:
                    winner = t1 if rng.random() < prob_t1 else t2
                advancement_counts[winner]["Championship"] += 1
                finalists.append(winner)

            # Championship game
            if len(finalists) >= 2 and finalists[0] and finalists[1]:
                t1, t2 = finalists[0], finalists[1]
                prob_t1 = _get_prediction_fast(lookup,t1, t2)
                if is_chalk:
                    winner = t1 if prob_t1 >= 0.5 else t2
                    chalk_results.append(("Championship", winner, t2 if winner == t1 else t1, max(prob_t1, 1 - prob_t1)))
                else:
                    winner = t1 if rng.random() < prob_t1 else t2
                advancement_counts[winner]["Winner"] += 1

    # Convert counts to probabilities
    advancement_probs: dict[int, dict[str, float]] = {}
    for team_id, counts in advancement_counts.items():
        advancement_probs[team_id] = {
            rnd: round(count / n_simulations, 4)
            for rnd, count in counts.items()
        }

    # Champion probabilities
    champion_probs = {
        tid: probs.get("Winner", 0)
        for tid, probs in advancement_probs.items()
    }

    return {
        "chalk_bracket": chalk_results,
        "advancement_probs": advancement_probs,
        "champion_probs": champion_probs,
        "team_names": team_names,
        "seeds": seeds,
    }


def generate_bracket(
    loader: DataLoader,
    predictions: pd.DataFrame,
    gender: str = "M",
    output: Optional[str] = None,
    season: int = CFG.TARGET_SEASON,
) -> str:
    """Generate a readable ASCII bracket and save to file.

    Returns the bracket string.
    """
    result = simulate_bracket(loader, predictions, season=season, gender=gender)
    team_names = result["team_names"]
    seeds_df = result["seeds"]
    advancement = result["advancement_probs"]
    champion_probs = result["champion_probs"]

    gender_label = "Men's" if gender == "M" else "Women's"
    lines = [
        f"{'=' * 70}",
        f"  {gender_label} NCAA Tournament Bracket — {season}",
        f"{'=' * 70}",
        "",
    ]

    # Print chalk bracket results
    if result["chalk_bracket"]:
        lines.append("CHALK BRACKET (most likely outcome at each matchup):")
        lines.append("-" * 50)
        current_round = ""
        for round_name, winner, loser, prob in result["chalk_bracket"]:
            if round_name != current_round:
                lines.append(f"\n  {round_name}:")
                current_round = round_name
            w_name = team_names.get(winner, str(winner))
            l_name = team_names.get(loser, str(loser))
            # Get seeds
            w_seed = _get_seed_num(seeds_df, winner)
            l_seed = _get_seed_num(seeds_df, loser)
            lines.append(
                f"    ({w_seed:>2}) {w_name:<25} def ({l_seed:>2}) {l_name:<25} ({prob:.1%})"
            )

    # Print top championship contenders
    lines.append("")
    lines.append("TOP CHAMPIONSHIP CONTENDERS:")
    lines.append("-" * 50)
    top = sorted(champion_probs.items(), key=lambda x: x[1], reverse=True)[:15]
    for team_id, prob in top:
        if prob > 0:
            name = team_names.get(team_id, str(team_id))
            seed = _get_seed_num(seeds_df, team_id)
            lines.append(f"  ({seed:>2}) {name:<30} {prob:>6.1%}")

    # Print advancement probability table for top teams
    lines.append("")
    lines.append("ADVANCEMENT PROBABILITIES (top 20 teams):")
    lines.append("-" * 80)
    header = f"  {'Team':<25} {'R64':>6} {'R32':>6} {'S16':>6} {'E8':>6} {'F4':>6} {'Champ':>6} {'Win':>6}"
    lines.append(header)
    lines.append("  " + "-" * 73)

    top_teams = sorted(advancement.items(), key=lambda x: x[1].get("Winner", 0), reverse=True)[:20]
    for team_id, probs in top_teams:
        name = team_names.get(team_id, str(team_id))[:24]
        seed = _get_seed_num(seeds_df, team_id)
        display_name = f"({seed:>2}) {name}"
        row = f"  {display_name:<25}"
        for rnd in ROUND_NAMES:
            val = probs.get(rnd, 0)
            row += f" {val:>5.0%}" if val > 0 else f" {'---':>5}"
        lines.append(row)

    bracket_str = "\n".join(lines)

    # Save to file
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(bracket_str)
        print(f"  Bracket saved to {output_path}")

    return bracket_str


def generate_advancement_csv(
    loader: DataLoader,
    predictions: pd.DataFrame,
    season: int = CFG.TARGET_SEASON,
    output: Optional[str] = None,
) -> pd.DataFrame:
    """Generate CSV with per-team advancement probabilities for both genders."""
    all_rows = []

    for gender in ["M", "W"]:
        result = simulate_bracket(loader, predictions, season=season, gender=gender)

        for team_id, probs in result["advancement_probs"].items():
            row = {
                "Season": season,
                "TeamID": team_id,
                "Gender": gender,
            }
            for rnd in ROUND_NAMES:
                row[f"{rnd}_prob"] = probs.get(rnd, 0)
            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"  Advancement probabilities saved to {output_path}")

    return df


def _get_seed_num(seeds_df: pd.DataFrame, team_id: int) -> int:
    """Get seed number for a team."""
    row = seeds_df[seeds_df["TeamID"] == team_id]
    if not row.empty:
        return int(row.iloc[0]["SeedNum"])
    return 0

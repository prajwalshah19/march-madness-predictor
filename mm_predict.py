#!/usr/bin/env python3
"""Quick matchup predictor CLI for bracket building.

Usage:
    mm-predict "Duke" "Arizona"
    mm-predict "UConn" "South Carolina" -g W
    mm-predict "Houston" "Florida" -v
    mm-predict upsets             # Show likely upsets by round
    mm-predict upsets 12          # Show upsets for 12-seeds specifically
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CFG
from src.team_matcher import TeamMatcher


def load_data() -> tuple[TeamMatcher, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all cached pipeline artifacts. Exits if not available."""
    import io
    import warnings
    warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

    ratings_path = CFG.OUTPUT_DIR / "team_ratings.csv"
    preds_path = CFG.OUTPUT_DIR / "blended_predictions.csv"

    if not ratings_path.exists() or not preds_path.exists():
        print("ERROR: Run `python main.py` first to generate predictions.")
        sys.exit(1)

    # Suppress loader/matcher print output
    _stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        matcher = TeamMatcher(CFG.DATA_DIR)
        from src.data_loader import DataLoader
        loader = DataLoader(CFG.DATA_DIR)
        seeds = loader.get_seeds()
    finally:
        sys.stdout = _stdout

    ratings = pd.read_csv(ratings_path)
    preds = pd.read_csv(preds_path)
    seeds = seeds[seeds["Season"] == CFG.TARGET_SEASON]

    # Load market signals if available
    market_path = CFG.OUTPUT_DIR / "market_signals.csv"
    futures_path = CFG.OUTPUT_DIR / "futures_signals.csv"
    market_signals = pd.read_csv(market_path) if market_path.exists() else pd.DataFrame()
    futures_signals = pd.read_csv(futures_path) if futures_path.exists() else pd.DataFrame()

    return matcher, ratings, preds, seeds, market_signals, futures_signals


def resolve_team(matcher: TeamMatcher, name: str, gender: str) -> int | None:
    """Resolve a team name to ID, with helpful error on failure."""
    team_id = matcher.match(name, gender)
    if team_id is None:
        other = "W" if gender == "M" else "M"
        team_id = matcher.match(name, other)
        if team_id is not None:
            print(f"  Note: '{name}' matched as {'women' if other == 'W' else 'men'}'s team")
    return team_id


def get_seed(seeds: pd.DataFrame, team_id: int, gender: str) -> int | None:
    """Get numeric seed for a team."""
    row = seeds[(seeds["TeamID"] == team_id) & (seeds["Gender"] == gender)]
    if row.empty:
        return None
    return int(row.iloc[0]["SeedNum"])


def get_seed_str(seeds: pd.DataFrame, team_id: int, gender: str) -> str:
    """Get seed string like '(1)' or '(12)' for a team."""
    s = get_seed(seeds, team_id, gender)
    return f"({s})" if s is not None else ""


def get_futures(futures_signals: pd.DataFrame, team_id: int) -> dict | None:
    """Look up futures data for a team."""
    if futures_signals.empty:
        return None
    row = futures_signals[futures_signals["TeamID"] == team_id]
    if row.empty:
        return None
    return row.sort_values("MarketConfidence", ascending=False).iloc[0].to_dict()


def predict_matchup(
    matcher: TeamMatcher,
    ratings: pd.DataFrame,
    preds: pd.DataFrame,
    seeds: pd.DataFrame,
    market_signals: pd.DataFrame,
    futures_signals: pd.DataFrame,
    team1_name: str,
    team2_name: str,
    gender: str,
    verbose: bool = False,
) -> None:
    """Look up and display matchup prediction."""
    id1 = resolve_team(matcher, team1_name, gender)
    id2 = resolve_team(matcher, team2_name, gender)

    if id1 is None:
        print(f"  Could not find team: '{team1_name}'")
        print(f"  Try a different spelling or check data/kaggle/MTeamSpellings.csv")
        sys.exit(1)
    if id2 is None:
        print(f"  Could not find team: '{team2_name}'")
        print(f"  Try a different spelling or check data/kaggle/MTeamSpellings.csv")
        sys.exit(1)

    name1 = matcher.get_name(id1)
    name2 = matcher.get_name(id2)

    # Kaggle convention: lower ID is TeamA
    team_a, team_b = min(id1, id2), max(id1, id2)
    flipped = id1 != team_a

    match_id = f"{CFG.TARGET_SEASON}_{team_a}_{team_b}"
    row = preds[preds["ID"] == match_id]

    if row.empty:
        print(f"  No prediction found for {name1} vs {name2}")
        print(f"  (Looked for ID: {match_id})")
        sys.exit(1)

    row = row.iloc[0]
    prob_a = float(row["Pred"])
    base_prob_a = float(row["BasePred"])
    prob_1 = (1 - prob_a) if flipped else prob_a
    base_prob_1 = (1 - base_prob_a) if flipped else base_prob_a

    r1 = ratings[(ratings["Season"] == CFG.TARGET_SEASON) & (ratings["TeamID"] == id1) & (ratings["Gender"] == gender)]
    r2 = ratings[(ratings["Season"] == CFG.TARGET_SEASON) & (ratings["TeamID"] == id2) & (ratings["Gender"] == gender)]

    seed1 = get_seed_str(seeds, id1, gender)
    seed2 = get_seed_str(seeds, id2, gender)

    winner_name = name1 if prob_1 > 0.5 else name2
    winner_prob = prob_1 if prob_1 > 0.5 else (1 - prob_1)
    variance = prob_1 * (1 - prob_1)

    if winner_prob >= 0.80:
        confidence = "Very High"
    elif winner_prob >= 0.65:
        confidence = "High"
    elif winner_prob >= 0.55:
        confidence = "Moderate"
    else:
        confidence = "Toss-up"

    print()
    print(f"  {'=' * 56}")
    print(f"  {seed1 + ' ' if seed1 else ''}{name1}  vs  {seed2 + ' ' if seed2 else ''}{name2}")
    print(f"  {'=' * 56}")
    print()
    print(f"  Winner:      {winner_name}")
    print(f"  Win Prob:    {winner_prob:.1%}")
    print(f"  Confidence:  {confidence}")
    print(f"  Variance:    {variance:.4f}")
    print()

    print(f"  ── Probability Breakdown ──")
    print(f"  {name1:<25} {prob_1:>6.1%}")
    print(f"  {name2:<25} {1 - prob_1:>6.1%}")
    print()

    market_source = str(row.get("MarketSource", ""))
    market_conf = float(row.get("MarketConfidence", 0))
    if market_source and market_conf > 0:
        market_prob_a = float(row.get("MarketPred", prob_a))
        market_prob_1 = (1 - market_prob_a) if flipped else market_prob_a
        print(f"  ── Model vs Market ──")
        print(f"  {'':25} {'Model':>8} {'Market':>8} {'Blended':>8}")
        print(f"  {name1:<25} {base_prob_1:>7.1%} {market_prob_1:>7.1%} {prob_1:>7.1%}")
        print(f"  {name2:<25} {1 - base_prob_1:>7.1%} {1 - market_prob_1:>7.1%} {1 - prob_1:>7.1%}")
        print(f"  Market source: {market_source} (confidence: {market_conf:.2f})")

        disagreement = abs(base_prob_1 - market_prob_1)
        if disagreement > 0.10:
            print(f"  ** Model/market disagree by {disagreement:.0%} — market may know something **")
        print()

    if verbose and not r1.empty and not r2.empty:
        r1v = r1.iloc[0]
        r2v = r2.iloc[0]
        print(f"  ── Team Profiles ──")
        print(f"  {'':25} {name1:>12} {name2:>12}")
        print(f"  {'Elo':25} {r1v.get('EloPreTourney', 0):>12.0f} {r2v.get('EloPreTourney', 0):>12.0f}")
        print(f"  {'Net Efficiency':25} {r1v.get('NetEff', 0):>12.1f} {r2v.get('NetEff', 0):>12.1f}")
        print(f"  {'Adj Offense':25} {r1v.get('AdjOff', 0):>12.1f} {r2v.get('AdjOff', 0):>12.1f}")
        print(f"  {'Adj Defense':25} {r1v.get('AdjDef', 0):>12.1f} {r2v.get('AdjDef', 0):>12.1f}")
        print(f"  {'Tempo':25} {r1v.get('Tempo', 0):>12.1f} {r2v.get('Tempo', 0):>12.1f}")
        print(f"  {'Opp 3pt%':25} {r1v.get('Opp3PtPct', 0):>11.1%} {r2v.get('Opp3PtPct', 0):>11.1%}")
        print(f"  {'Off Reb Rate':25} {r1v.get('OffRebRate', 0):>11.1%} {r2v.get('OffRebRate', 0):>11.1%}")
        print(f"  {'Late FT%':25} {r1v.get('LateFTPct', 0):>11.1%} {r2v.get('LateFTPct', 0):>11.1%}")
        massey1 = r1v.get("MasseyLogOdds", 0)
        massey2 = r2v.get("MasseyLogOdds", 0)
        if massey1 != 0 or massey2 != 0:
            print(f"  {'Massey Log-Odds':25} {massey1:>12.2f} {massey2:>12.2f}")
        print()

    f1 = get_futures(futures_signals, id1)
    f2 = get_futures(futures_signals, id2)
    if f1 or f2:
        print(f"  ── Championship Futures ──")
        if f1:
            print(f"  {name1:<25} {f1['ChampionshipProb']:>6.1%} to win title  (vol: ${f1['Volume']:,.0f})")
        else:
            print(f"  {name1:<25} No futures market")
        if f2:
            print(f"  {name2:<25} {f2['ChampionshipProb']:>6.1%} to win title  (vol: ${f2['Volume']:,.0f})")
        else:
            print(f"  {name2:<25} No futures market")
        print()

    if seed1 and seed2:
        s1 = int(seed1.strip("()"))
        s2 = int(seed2.strip("()"))
        if s1 != s2:
            higher_seed_prob = prob_1 if s1 < s2 else (1 - prob_1)
            if higher_seed_prob < 0.50:
                lower_seed = name1 if s1 > s2 else name2
                print(f"  ** UPSET ALERT: {lower_seed} favored despite lower seed **")
                print()

    print(f"  {'=' * 56}")
    print()


# ── Standard seed matchups by round ──────────────────────────
# Maps (higher_seed, lower_seed) for typical tournament bracket pairings
ROUND_MATCHUPS: dict[str, list[tuple[int, int]]] = {
    "R64": [(1, 16), (2, 15), (3, 14), (4, 13), (5, 12), (6, 11), (7, 10), (8, 9)],
    "R32": [(1, 8), (2, 7), (3, 6), (4, 5)],
}


def find_upsets(
    matcher: TeamMatcher,
    ratings: pd.DataFrame,
    preds: pd.DataFrame,
    seeds: pd.DataFrame,
    futures_signals: pd.DataFrame,
    gender: str,
    filter_seed: int | None = None,
) -> None:
    """Find and display the most likely upsets across tournament matchups."""
    # Build seed lookup: (gender, seed_num) -> list of (team_id, region)
    gender_seeds = seeds[seeds["Gender"] == gender]
    if gender_seeds.empty:
        print(f"  No seeds found for {'men' if gender == 'M' else 'women'}'s tournament")
        return

    seed_teams: dict[int, list[tuple[int, str]]] = {}
    for _, row in gender_seeds.iterrows():
        sn = int(row["SeedNum"])
        tid = int(row["TeamID"])
        region = str(row.get("Region", "?"))
        seed_teams.setdefault(sn, []).append((tid, region))

    upsets: list[dict] = []

    # Check all seed-vs-seed matchups
    for round_name, matchup_pairs in ROUND_MATCHUPS.items():
        for fav_seed, dog_seed in matchup_pairs:
            if filter_seed is not None and dog_seed != filter_seed:
                continue

            fav_teams = seed_teams.get(fav_seed, [])
            dog_teams = seed_teams.get(dog_seed, [])

            for fav_id, fav_region in fav_teams:
                for dog_id, dog_region in dog_teams:
                    # In real brackets, only same-region matchups happen in R64/R32
                    # but we show all possible for bracket planning
                    ta, tb = min(fav_id, dog_id), max(fav_id, dog_id)
                    match_id = f"{CFG.TARGET_SEASON}_{ta}_{tb}"
                    row = preds[preds["ID"] == match_id]
                    if row.empty:
                        continue

                    pred = float(row.iloc[0]["Pred"])
                    base_pred = float(row.iloc[0]["BasePred"])
                    # prob of lower-ID team winning
                    # We need prob of the underdog (dog) winning
                    if dog_id == ta:
                        dog_prob = pred
                        dog_base = base_pred
                    else:
                        dog_prob = 1 - pred
                        dog_base = 1 - base_pred

                    # Only show if underdog has meaningful chance (>15%)
                    if dog_prob < 0.15:
                        continue

                    fav_name = matcher.get_name(fav_id)
                    dog_name = matcher.get_name(dog_id)

                    # Market boost: how much does market like the underdog vs model?
                    market_boost = dog_prob - dog_base

                    upsets.append({
                        "round": round_name,
                        "fav_seed": fav_seed,
                        "dog_seed": dog_seed,
                        "fav_name": fav_name,
                        "dog_name": dog_name,
                        "dog_prob": dog_prob,
                        "dog_base": dog_base,
                        "market_boost": market_boost,
                        "fav_region": fav_region,
                        "dog_region": dog_region,
                    })

    if not upsets:
        print(f"  No upset candidates found" +
              (f" for {filter_seed}-seeds" if filter_seed else ""))
        return

    # Sort by underdog probability descending (most likely upsets first)
    upsets.sort(key=lambda x: x["dog_prob"], reverse=True)

    label = "Men's" if gender == "M" else "Women's"
    title = f"{label} Upset Candidates"
    if filter_seed:
        title += f" — {filter_seed}-seeds"

    print()
    print(f"  {'=' * 72}")
    print(f"  {title}")
    print(f"  {'=' * 72}")
    print()
    print(f"  {'Matchup':<40} {'Upset%':>7} {'Model':>7} {'Mkt Boost':>9} {'Round':>5}")
    print(f"  {'-' * 72}")

    for u in upsets:
        matchup_str = f"({u['dog_seed']:>2}) {u['dog_name']:<16} > ({u['fav_seed']:>2}) {u['fav_name']:<14}"
        mkt = f"+{u['market_boost']:.0%}" if u["market_boost"] > 0.005 else (
              f"{u['market_boost']:.0%}" if u["market_boost"] < -0.005 else "  --")

        # Color coding via markers
        if u["dog_prob"] >= 0.50:
            marker = " ***"  # Favored upset
        elif u["dog_prob"] >= 0.35:
            marker = "  **"  # Strong upset candidate
        elif u["dog_prob"] >= 0.25:
            marker = "   *"  # Worth considering
        else:
            marker = "    "

        print(f"  {matchup_str:<40} {u['dog_prob']:>6.1%} {u['dog_base']:>6.1%} {mkt:>9} {u['round']:>5}{marker}")

    print()
    print(f"  *** = model favors upset  ** = strong candidate  * = worth considering")
    print(f"  Mkt Boost = how much market data shifts probability vs base model")
    print(f"  {'=' * 72}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="March Madness matchup predictor",
        usage="mm-predict [-h] {TEAM1 TEAM2 | upsets [SEED]}",
    )
    # Check if first arg is "upsets" subcommand
    if len(sys.argv) > 1 and sys.argv[1] == "upsets":
        # Parse upsets subcommand
        sub = argparse.ArgumentParser(
            description="Show likely tournament upsets",
            usage="mm-predict upsets [SEED] [-g M|W]",
        )
        sub.add_argument("_cmd", help=argparse.SUPPRESS)  # consume "upsets"
        sub.add_argument("seed", nargs="?", type=int, default=None,
                         help="Filter to a specific seed (e.g., 12)")
        sub.add_argument("-g", "--gender", default="M", choices=["M", "W"],
                         help="M for men's (default), W for women's")
        args = sub.parse_args()

        matcher, ratings, preds, seeds, market_signals, futures_signals = load_data()
        find_upsets(matcher, ratings, preds, seeds, futures_signals, args.gender, args.seed)
    else:
        # Parse matchup subcommand
        parser.add_argument("team1", help="First team name")
        parser.add_argument("team2", help="Second team name")
        parser.add_argument("-g", "--gender", default="M", choices=["M", "W"],
                            help="M for men's (default), W for women's")
        parser.add_argument("-v", "--verbose", action="store_true",
                            help="Show detailed team profiles")
        args = parser.parse_args()

        matcher, ratings, preds, seeds, market_signals, futures_signals = load_data()
        predict_matchup(
            matcher, ratings, preds, seeds, market_signals, futures_signals,
            args.team1, args.team2, args.gender, args.verbose,
        )


if __name__ == "__main__":
    main()

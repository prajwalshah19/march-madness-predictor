#!/usr/bin/env python3
"""Quick matchup predictor CLI for bracket building.

Usage:
    mm-predict "Duke" "Arizona"
    mm-predict "Duke" "Arizona" -v
    mm-predict path "Duke"                # Show bracket path with probabilities
    mm-predict upsets                     # Actual bracket upset candidates
    mm-predict upsets 12                  # 12-seed upset candidates
    mm-predict upsets --all               # All possible cross-region matchups
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CFG
from src.team_matcher import TeamMatcher


# ── Bracket structure ────────────────────────────────────────────
# Per-region bracket tree: R64 seed pairings and how they feed into later rounds.
# R64 games indexed 0-7, then grouped into R32, S16, E8.
R64_PAIRINGS = [(1, 16), (8, 9), (4, 13), (5, 12), (6, 11), (3, 14), (7, 10), (2, 15)]

# R32: which R64 game winners play each other (indices into R64_PAIRINGS)
R32_PAIRINGS = [(0, 1), (3, 2), (4, 5), (7, 6)]
# i.e., R32 game 0: winner(1v16) vs winner(8v9)
#       R32 game 1: winner(5v12) vs winner(4v13)
#       R32 game 2: winner(6v11) vs winner(3v14)
#       R32 game 3: winner(2v15) vs winner(7v10)
# NOTE: ordering follows actual bracket (R2x1=0,1 R2x4=3,2 R2x3=4,5 R2x2=7,6)

# S16: which R32 game winners play each other (indices into R32 results)
S16_PAIRINGS = [(0, 1), (3, 2)]

# E8: which S16 winners play each other
E8_PAIRINGS = [(0, 1)]

# Final Four region pairings (from tournament slots: R5WX, R5YZ)
# Will be parsed dynamically from slot data.

ROUND_NAMES = ["R64", "R32", "Sweet 16", "Elite 8", "Final Four", "Championship"]

# Which seeds are in each "bracket section" for opponent lookup
# Maps seed -> opponent seeds at each round within the same region
OPPONENT_SEEDS: dict[int, dict[str, list[int]]] = {}
_top_quarter_a = [1, 16]
_top_quarter_b = [8, 9]
_mid_quarter_a = [4, 13]
_mid_quarter_b = [5, 12]
_bot_quarter_a = [6, 11]
_bot_quarter_b = [3, 14]
_low_quarter_a = [7, 10]
_low_quarter_b = [2, 15]

_top_half = _top_quarter_a + _top_quarter_b + _mid_quarter_a + _mid_quarter_b
_bot_half = _bot_quarter_a + _bot_quarter_b + _low_quarter_a + _low_quarter_b

for s in _top_quarter_a:
    OPPONENT_SEEDS[s] = {"R32": _top_quarter_b, "S16": _mid_quarter_a + _mid_quarter_b, "E8": _bot_half}
for s in _top_quarter_b:
    OPPONENT_SEEDS[s] = {"R32": _top_quarter_a, "S16": _mid_quarter_a + _mid_quarter_b, "E8": _bot_half}
for s in _mid_quarter_a:
    OPPONENT_SEEDS[s] = {"R32": _mid_quarter_b, "S16": _top_quarter_a + _top_quarter_b, "E8": _bot_half}
for s in _mid_quarter_b:
    OPPONENT_SEEDS[s] = {"R32": _mid_quarter_a, "S16": _top_quarter_a + _top_quarter_b, "E8": _bot_half}
for s in _bot_quarter_a:
    OPPONENT_SEEDS[s] = {"R32": _bot_quarter_b, "S16": _low_quarter_a + _low_quarter_b, "E8": _top_half}
for s in _bot_quarter_b:
    OPPONENT_SEEDS[s] = {"R32": _bot_quarter_a, "S16": _low_quarter_a + _low_quarter_b, "E8": _top_half}
for s in _low_quarter_a:
    OPPONENT_SEEDS[s] = {"R32": _low_quarter_b, "S16": _bot_quarter_a + _bot_quarter_b, "E8": _top_half}
for s in _low_quarter_b:
    OPPONENT_SEEDS[s] = {"R32": _low_quarter_a, "S16": _bot_quarter_a + _bot_quarter_b, "E8": _top_half}


# ── Data loading ─────────────────────────────────────────────────

def load_data():
    """Load all cached pipeline artifacts. Exits if not available."""
    import io
    import warnings
    warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

    ratings_path = CFG.OUTPUT_DIR / "team_ratings.csv"
    preds_path = CFG.OUTPUT_DIR / "blended_predictions.csv"

    if not ratings_path.exists() or not preds_path.exists():
        print("ERROR: Run `python main.py` first to generate predictions.")
        sys.exit(1)

    _stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        matcher = TeamMatcher(CFG.DATA_DIR)
        from src.data_loader import DataLoader
        loader = DataLoader(CFG.DATA_DIR)
        seeds = loader.get_seeds()
        try:
            slots = loader.get_tourney_slots(season=CFG.TARGET_SEASON)
        except Exception:
            slots = pd.DataFrame()
    finally:
        sys.stdout = _stdout

    ratings = pd.read_csv(ratings_path)
    preds = pd.read_csv(preds_path)
    seeds = seeds[seeds["Season"] == CFG.TARGET_SEASON]

    market_path = CFG.OUTPUT_DIR / "market_signals.csv"
    futures_path = CFG.OUTPUT_DIR / "futures_signals.csv"
    market_signals = pd.read_csv(market_path) if market_path.exists() else pd.DataFrame()
    futures_signals = pd.read_csv(futures_path) if futures_path.exists() else pd.DataFrame()

    return matcher, ratings, preds, seeds, slots, market_signals, futures_signals


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
    row = seeds[(seeds["TeamID"] == team_id) & (seeds["Gender"] == gender)]
    return int(row.iloc[0]["SeedNum"]) if not row.empty else None


def get_seed_str(seeds: pd.DataFrame, team_id: int, gender: str) -> str:
    s = get_seed(seeds, team_id, gender)
    return f"({s})" if s is not None else ""


def get_futures(futures_signals: pd.DataFrame, team_id: int) -> dict | None:
    if futures_signals.empty:
        return None
    row = futures_signals[futures_signals["TeamID"] == team_id]
    if row.empty:
        return None
    return row.sort_values("MarketConfidence", ascending=False).iloc[0].to_dict()


def _get_pred(preds: pd.DataFrame, team_a: int, team_b: int) -> float:
    """Get P(team_a wins) from predictions. Returns 0.5 if not found."""
    lo, hi = min(team_a, team_b), max(team_a, team_b)
    match_id = f"{CFG.TARGET_SEASON}_{lo}_{hi}"
    row = preds[preds["ID"] == match_id]
    if row.empty:
        return 0.5
    p = float(row.iloc[0]["Pred"])
    return p if team_a == lo else (1 - p)


def _build_bracket(seeds: pd.DataFrame, gender: str) -> dict[str, dict[int, int]]:
    """Build region -> {seed_num: team_id} mapping."""
    gender_seeds = seeds[seeds["Gender"] == gender]
    bracket: dict[str, dict[int, int]] = {}
    for _, row in gender_seeds.iterrows():
        region = str(row["Region"])
        sn = int(row["SeedNum"])
        tid = int(row["TeamID"])
        bracket.setdefault(region, {})[sn] = tid
    return bracket


def _get_f4_pairings(slots: pd.DataFrame, gender: str) -> list[tuple[str, str]]:
    """Parse Final Four region pairings from tournament slots."""
    if slots.empty:
        # Default NCAA bracket structure
        return [("W", "X"), ("Y", "Z")]

    prefix = "M" if gender == "M" else "W"
    seen = set()
    pairings = []
    for _, row in slots.iterrows():
        slot = str(row.get("Slot", ""))
        if slot.startswith("R5"):
            strong = str(row.get("StrongSeed", ""))
            weak = str(row.get("WeakSeed", ""))
            r1 = strong[-2] if len(strong) >= 2 else ""
            r2 = weak[-2] if len(weak) >= 2 else ""
            pair = (r1, r2)
            if r1 and r2 and pair not in seen:
                seen.add(pair)
                pairings.append(pair)

    return pairings if pairings else [("W", "X"), ("Y", "Z")]


# ── Bracket probability computation ──────────────────────────────

def compute_region_probs(
    preds: pd.DataFrame,
    region_teams: dict[int, int],
    matcher: TeamMatcher,
) -> dict[int, dict[str, float]]:
    """Compute P(each seed reaches each round) for a single region.

    Uses exact bracket tree structure with forward probability propagation.

    Returns: {team_id: {"R64": 1.0, "R32": p, "S16": p, "E8": p, "F4": p}}
    """
    def pred(sa: int, sb: int) -> float:
        ta = region_teams.get(sa, 0)
        tb = region_teams.get(sb, 0)
        if ta == 0 or tb == 0:
            return 0.5
        return _get_pred(preds, ta, tb)

    all_seeds = set(region_teams.keys())

    # P(reach R64) = 1.0 for all
    p = {s: {"R64": 1.0} for s in all_seeds}

    # R64: compute P(reach R32) = P(win R64 game)
    for sa, sb in R64_PAIRINGS:
        if sa in all_seeds and sb in all_seeds:
            w = pred(sa, sb)
            p[sa]["R32"] = w
            p[sb]["R32"] = 1 - w

    # Fill missing seeds
    for s in all_seeds:
        p[s].setdefault("R32", 0.0)

    # R32 -> S16: for each seed, weighted probability across possible opponents
    for s in all_seeds:
        opp_seeds = OPPONENT_SEEDS.get(s, {}).get("R32", [])
        prob_advance = 0.0
        for opp in opp_seeds:
            if opp in all_seeds:
                prob_advance += p[s]["R32"] * p[opp]["R32"] * pred(s, opp)
        p[s]["S16"] = prob_advance

    # S16 -> E8
    for s in all_seeds:
        opp_seeds = OPPONENT_SEEDS.get(s, {}).get("S16", [])
        prob_advance = 0.0
        for opp in opp_seeds:
            if opp in all_seeds:
                prob_advance += p[s]["S16"] * p[opp]["S16"] * pred(s, opp)
        p[s]["E8"] = prob_advance

    # E8 -> F4 (region final)
    for s in all_seeds:
        opp_seeds = OPPONENT_SEEDS.get(s, {}).get("E8", [])
        prob_advance = 0.0
        for opp in opp_seeds:
            if opp in all_seeds:
                prob_advance += p[s]["E8"] * p[opp]["E8"] * pred(s, opp)
        p[s]["F4"] = prob_advance

    # Convert to team_id keys
    result: dict[int, dict[str, float]] = {}
    for s, probs in p.items():
        tid = region_teams.get(s, 0)
        if tid:
            result[tid] = probs

    return result


# ── Path command ─────────────────────────────────────────────────

def show_path(
    matcher: TeamMatcher,
    preds: pd.DataFrame,
    seeds: pd.DataFrame,
    slots: pd.DataFrame,
    futures_signals: pd.DataFrame,
    team_name: str,
    gender: str,
) -> None:
    """Show a team's bracket path with round-by-round probabilities."""
    team_id = resolve_team(matcher, team_name, gender)
    if team_id is None:
        print(f"  Could not find team: '{team_name}'")
        sys.exit(1)

    name = matcher.get_name(team_id)
    team_seed_row = seeds[(seeds["TeamID"] == team_id) & (seeds["Gender"] == gender)]
    if team_seed_row.empty:
        print(f"  {name} is not in the {CFG.TARGET_SEASON} tournament bracket")
        sys.exit(1)

    team_seed = int(team_seed_row.iloc[0]["SeedNum"])
    team_region = str(team_seed_row.iloc[0]["Region"])

    # Build full bracket
    bracket = _build_bracket(seeds, gender)
    region_teams = bracket.get(team_region, {})

    # Compute advancement probabilities for this region
    region_probs = compute_region_probs(preds, region_teams, matcher)

    # Also compute all other regions for F4/Championship
    all_region_probs: dict[str, dict[int, dict[str, float]]] = {}
    for reg, rteams in bracket.items():
        all_region_probs[reg] = compute_region_probs(preds, rteams, matcher)

    # Get F4 pairings
    f4_pairings = _get_f4_pairings(slots, gender)
    opposing_f4_region = None
    championship_regions = []
    for r1, r2 in f4_pairings:
        if team_region == r1:
            opposing_f4_region = r2
        elif team_region == r2:
            opposing_f4_region = r1
        else:
            championship_regions.extend([r1, r2])

    my_probs = region_probs.get(team_id, {})

    print()
    print(f"  {'=' * 66}")
    print(f"  ({team_seed}) {name} — Region {team_region} Bracket Path")
    print(f"  {'=' * 66}")
    print()

    cumulative = 1.0
    rounds_data = [
        ("R64", "R32", "Round of 64"),
        ("R32", "S16", "Round of 32"),
        ("S16", "E8", "Sweet 16"),
        ("E8", "F4", "Elite 8"),
    ]

    for from_round, to_round, display_name in rounds_data:
        opp_seed_list = OPPONENT_SEEDS.get(team_seed, {}).get(from_round, []) if from_round != "R64" else []
        r64_opp = None

        if from_round == "R64":
            # Direct opponent from R64 pairings
            for sa, sb in R64_PAIRINGS:
                if team_seed == sa:
                    r64_opp = sb
                    break
                elif team_seed == sb:
                    r64_opp = sa
                    break

        # Win probability for this round
        win_prob = 0.0
        if from_round == "R64" and r64_opp is not None:
            opp_tid = region_teams.get(r64_opp, 0)
            opp_name = matcher.get_name(opp_tid) if opp_tid else f"Seed {r64_opp}"
            win_prob = _get_pred(preds, team_id, opp_tid) if opp_tid else 0.5
            print(f"  {display_name:<14} vs ({r64_opp:>2}) {opp_name}")
            print(f"  {'':14} Win: {win_prob:.1%}")
        else:
            # Multiple possible opponents — show top candidates
            opponents = []
            for opp_s in opp_seed_list:
                opp_tid = region_teams.get(opp_s, 0)
                if opp_tid == 0:
                    continue
                opp_probs = region_probs.get(opp_tid, {})
                opp_reach = opp_probs.get(from_round, 0.0)
                if opp_reach < 0.01:
                    continue
                wp = _get_pred(preds, team_id, opp_tid)
                opponents.append({
                    "seed": opp_s,
                    "name": matcher.get_name(opp_tid),
                    "reach_prob": opp_reach,
                    "win_prob": wp,
                })

            opponents.sort(key=lambda x: x["reach_prob"], reverse=True)

            if opponents:
                # Weighted win probability
                total_opp_prob = sum(o["reach_prob"] for o in opponents)
                if total_opp_prob > 0:
                    win_prob = sum(
                        o["reach_prob"] * o["win_prob"] for o in opponents
                    ) / total_opp_prob

                print(f"  {display_name:<14} Likely opponents:")
                for o in opponents[:4]:
                    bar = "█" * int(o["reach_prob"] * 20) + "░" * (20 - int(o["reach_prob"] * 20))
                    print(f"  {'':14}  ({o['seed']:>2}) {o['name']:<18} {o['reach_prob']:>5.0%} chance  |  beat: {o['win_prob']:.0%}")
                if len(opponents) > 4:
                    print(f"  {'':14}  ... and {len(opponents) - 4} others")
                print(f"  {'':14} Weighted win: {win_prob:.1%}")

        cumulative *= win_prob
        advance_prob = my_probs.get(to_round, cumulative)
        print(f"  {'':14} Advance prob: {advance_prob:.1%}")
        print()

    # Final Four
    if opposing_f4_region and opposing_f4_region in all_region_probs:
        f4_reach = my_probs.get("F4", cumulative)
        opp_region_probs = all_region_probs[opposing_f4_region]
        opp_region_teams = bracket.get(opposing_f4_region, {})

        f4_opponents = []
        for opp_tid, oprobs in opp_region_probs.items():
            opp_f4 = oprobs.get("F4", 0.0)
            if opp_f4 < 0.01:
                continue
            wp = _get_pred(preds, team_id, opp_tid)
            opp_s = get_seed(seeds, opp_tid, gender) or 0
            f4_opponents.append({
                "seed": opp_s,
                "name": matcher.get_name(opp_tid),
                "reach_prob": opp_f4,
                "win_prob": wp,
            })
        f4_opponents.sort(key=lambda x: x["reach_prob"], reverse=True)

        print(f"  {'Final Four':<14} vs Region {opposing_f4_region} champion:")
        for o in f4_opponents[:4]:
            print(f"  {'':14}  ({o['seed']:>2}) {o['name']:<18} {o['reach_prob']:>5.0%} chance  |  beat: {o['win_prob']:.0%}")

        total_f4 = sum(o["reach_prob"] for o in f4_opponents)
        if total_f4 > 0:
            f4_win = sum(o["reach_prob"] * o["win_prob"] for o in f4_opponents) / total_f4
            cumulative *= f4_win
            print(f"  {'':14} Weighted win: {f4_win:.1%}")
            print(f"  {'':14} Reach F4: {f4_reach:.1%} | Win F4: {f4_reach * f4_win:.1%}")
            print()

    # Championship
    if championship_regions:
        print(f"  {'Championship':<14} vs Region {'/'.join(championship_regions)} champion")
        champ_opponents = []
        for creg in championship_regions:
            if creg in all_region_probs:
                for opp_tid, oprobs in all_region_probs[creg].items():
                    opp_f4 = oprobs.get("F4", 0.0)
                    if opp_f4 < 0.01:
                        continue
                    wp = _get_pred(preds, team_id, opp_tid)
                    opp_s = get_seed(seeds, opp_tid, gender) or 0
                    champ_opponents.append({
                        "seed": opp_s,
                        "name": matcher.get_name(opp_tid),
                        "reach_prob": opp_f4,
                        "win_prob": wp,
                        "region": creg,
                    })
        champ_opponents.sort(key=lambda x: x["reach_prob"], reverse=True)
        for o in champ_opponents[:4]:
            print(f"  {'':14}  ({o['seed']:>2}) {o['name']:<18} {o['reach_prob']:>5.0%} chance  |  beat: {o['win_prob']:.0%}")
        total_ch = sum(o["reach_prob"] for o in champ_opponents)
        if total_ch > 0:
            ch_win = sum(o["reach_prob"] * o["win_prob"] for o in champ_opponents) / total_ch
            print(f"  {'':14} Weighted win: {ch_win:.1%}")
            cumulative *= ch_win
        print()

    # Futures comparison
    fut = get_futures(futures_signals, team_id)
    print(f"  ── Summary ──")
    print(f"  Model championship prob:  {cumulative:.1%}")
    if fut:
        print(f"  Market championship prob: {fut['ChampionshipProb']:.1%}  (vol: ${fut['Volume']:,.0f})")
    print(f"  {'=' * 66}")
    print()


# ── Matchup command ──────────────────────────────────────────────

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

    team_a, team_b = min(id1, id2), max(id1, id2)
    flipped = id1 != team_a

    match_id = f"{CFG.TARGET_SEASON}_{team_a}_{team_b}"
    row = preds[preds["ID"] == match_id]

    if row.empty:
        print(f"  No prediction found for {name1} vs {name2}")
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
        r1v, r2v = r1.iloc[0], r2.iloc[0]
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
        massey1, massey2 = r1v.get("MasseyLogOdds", 0), r2v.get("MasseyLogOdds", 0)
        if massey1 != 0 or massey2 != 0:
            print(f"  {'Massey Log-Odds':25} {massey1:>12.2f} {massey2:>12.2f}")
        print()

    f1, f2 = get_futures(futures_signals, id1), get_futures(futures_signals, id2)
    if f1 or f2:
        print(f"  ── Championship Futures ──")
        for nm, f in [(name1, f1), (name2, f2)]:
            if f:
                print(f"  {nm:<25} {f['ChampionshipProb']:>6.1%} to win title  (vol: ${f['Volume']:,.0f})")
            else:
                print(f"  {nm:<25} No futures market")
        print()

    if seed1 and seed2:
        s1, s2 = int(seed1.strip("()")), int(seed2.strip("()"))
        if s1 != s2:
            higher_seed_prob = prob_1 if s1 < s2 else (1 - prob_1)
            if higher_seed_prob < 0.50:
                lower_seed = name1 if s1 > s2 else name2
                print(f"  ** UPSET ALERT: {lower_seed} favored despite lower seed **")
                print()

    print(f"  {'=' * 56}")
    print()


# ── Upsets command ───────────────────────────────────────────────

def find_upsets(
    matcher: TeamMatcher,
    preds: pd.DataFrame,
    seeds: pd.DataFrame,
    gender: str,
    filter_seed: int | None = None,
    show_all: bool = False,
) -> None:
    """Find and display likely upsets — actual bracket matchups only (unless --all)."""
    bracket = _build_bracket(seeds, gender)
    if not bracket:
        print(f"  No bracket found for {'men' if gender == 'M' else 'women'}'s tournament")
        return

    upsets: list[dict] = []

    for region, region_teams in bracket.items():
        # Only R64 matchups are guaranteed — the actual pairings
        for fav_seed, dog_seed in R64_PAIRINGS:
            if filter_seed is not None and filter_seed not in (fav_seed, dog_seed):
                continue
            fav_tid = region_teams.get(fav_seed)
            dog_tid = region_teams.get(dog_seed)
            if not fav_tid or not dog_tid:
                continue

            # Ensure fav is actually the higher seed (lower number)
            if fav_seed > dog_seed:
                fav_seed, dog_seed = dog_seed, fav_seed
                fav_tid, dog_tid = dog_tid, fav_tid

            dog_prob = _get_pred(preds, dog_tid, fav_tid)
            if dog_prob < 0.02:
                continue

            upsets.append({
                "round": "R64",
                "region": region,
                "fav_seed": fav_seed,
                "dog_seed": dog_seed,
                "fav_name": matcher.get_name(fav_tid),
                "dog_name": matcher.get_name(dog_tid),
                "dog_prob": dog_prob,
            })

        if show_all:
            continue  # Skip R32 bracket logic for --all mode

        # Also show R32 "expected" matchups (chalk opponents)
        # R32 matchups: winner of each R64 pair plays the paired R64 winner
        r32_seed_pairs = [
            ((1, 16), (8, 9)),
            ((4, 13), (5, 12)),
            ((6, 11), (3, 14)),
            ((7, 10), (2, 15)),
        ]
        for (fa, fb), (da, db) in r32_seed_pairs:
            # Most likely R32 matchup is chalk (lower seed wins R64)
            fav_seed_r32 = min(fa, fb)
            dog_seed_r32 = min(da, db)
            if fav_seed_r32 > dog_seed_r32:
                fav_seed_r32, dog_seed_r32 = dog_seed_r32, fav_seed_r32
                fa, fb, da, db = da, db, fa, fb

            if filter_seed is not None and filter_seed not in (fav_seed_r32, dog_seed_r32):
                continue

            fav_tid = region_teams.get(min(fa, fb))
            dog_tid = region_teams.get(min(da, db))
            if not fav_tid or not dog_tid:
                continue

            dog_prob = _get_pred(preds, dog_tid, fav_tid)
            if dog_prob < 0.15:
                continue

            upsets.append({
                "round": "R32",
                "region": region,
                "fav_seed": min(fa, fb),
                "dog_seed": min(da, db),
                "fav_name": matcher.get_name(fav_tid),
                "dog_name": matcher.get_name(dog_tid),
                "dog_prob": dog_prob,
            })

    if not upsets:
        seed_str = f" for {filter_seed}-seeds" if filter_seed else ""
        print(f"  No upset candidates found{seed_str}")
        return

    upsets.sort(key=lambda x: x["dog_prob"], reverse=True)

    label = "Men's" if gender == "M" else "Women's"
    title = f"{label} Upset Candidates — {'Actual Bracket' if not show_all else 'All Matchups'}"
    if filter_seed:
        title += f" — {filter_seed}-seeds"

    print()
    print(f"  {'=' * 74}")
    print(f"  {title}")
    print(f"  {'=' * 74}")
    print()
    print(f"  {'Matchup':<44} {'Upset%':>7} {'Region':>6} {'Round':>5}")
    print(f"  {'-' * 74}")

    for u in upsets:
        matchup_str = f"({u['dog_seed']:>2}) {u['dog_name']:<18} > ({u['fav_seed']:>2}) {u['fav_name']:<16}"
        if u["dog_prob"] >= 0.50:
            marker = " ***"
        elif u["dog_prob"] >= 0.35:
            marker = "  **"
        elif u["dog_prob"] >= 0.25:
            marker = "   *"
        else:
            marker = "    "
        print(f"  {matchup_str:<44} {u['dog_prob']:>6.1%} {u['region']:>6} {u['round']:>5}{marker}")

    print()
    print(f"  *** = model favors upset  ** = strong candidate  * = worth considering")
    print(f"  {'=' * 74}")
    print()


# ── Bracket command ──────────────────────────────────────────────

def _simulate_region_chalk(
    preds: pd.DataFrame,
    region_teams: dict[int, int],
    matcher: TeamMatcher,
) -> list[list[tuple[int, int, str, float]]]:
    """Simulate a region bracket picking the favorite at each round.

    Returns list of rounds, each a list of (seed, team_id, name, win_prob) tuples.
    """
    # R64
    r64_results = []
    for sa, sb in R64_PAIRINGS:
        ta = region_teams.get(sa, 0)
        tb = region_teams.get(sb, 0)
        if ta == 0 or tb == 0:
            winner_seed = sa if ta else sb
            winner_id = ta or tb
            r64_results.append((winner_seed, winner_id, matcher.get_name(winner_id), 0.5))
            continue
        p = _get_pred(preds, ta, tb)
        if p >= 0.5:
            r64_results.append((sa, ta, matcher.get_name(ta), p))
        else:
            r64_results.append((sb, tb, matcher.get_name(tb), 1 - p))

    rounds = [
        [(s, tid, name, wp) for s, tid, name, wp in r64_results]
    ]

    # R32, S16, E8 — pair adjacent winners
    current = r64_results
    while len(current) > 1:
        next_round = []
        for i in range(0, len(current), 2):
            sa, ta, na, _ = current[i]
            sb, tb, nb, _ = current[i + 1]
            p = _get_pred(preds, ta, tb) if ta and tb else 0.5
            if p >= 0.5:
                next_round.append((sa, ta, na, p))
            else:
                next_round.append((sb, tb, nb, 1 - p))
        rounds.append(next_round)
        current = next_round

    return rounds


def _fmt_team(seed: int, name: str, width: int = 20) -> str:
    """Format team as '(1) Duke' padded to width."""
    s = f"({seed:>2}) {name}"
    return s[:width].ljust(width)


def show_bracket(
    matcher: TeamMatcher,
    preds: pd.DataFrame,
    seeds: pd.DataFrame,
    slots: pd.DataFrame,
    futures_signals: pd.DataFrame,
    gender: str,
) -> None:
    """Generate and display a visual ASCII bracket."""
    bracket = _build_bracket(seeds, gender)
    if not bracket:
        print(f"  No bracket found")
        return

    f4_pairings = _get_f4_pairings(slots, gender)
    label = "Men's" if gender == "M" else "Women's"

    print()
    print(f"  {'=' * 78}")
    print(f"  {label} NCAA Tournament Bracket — {CFG.TARGET_SEASON} (Model Picks)")
    print(f"  {'=' * 78}")

    # Simulate each region
    region_champions: dict[str, tuple[int, int, str]] = {}  # region -> (seed, tid, name)
    region_order = sorted(bracket.keys())

    for region in region_order:
        region_teams = bracket[region]
        rounds = _simulate_region_chalk(preds, region_teams, matcher)

        # R64 results (8 games), R32 (4), S16 (2), E8 (1)
        r64 = rounds[0]  # 8 winners
        r32 = rounds[1] if len(rounds) > 1 else []  # 4 winners
        s16 = rounds[2] if len(rounds) > 2 else []  # 2 winners
        e8 = rounds[3] if len(rounds) > 3 else []   # 1 winner (region champ)

        if e8:
            cs, ct, cn, _ = e8[0]
            region_champions[region] = (cs, ct, cn)

        W = 20  # team name column width

        print()
        print(f"  ┌─── Region {region} ────────────────────────────────────────────────────┐")
        print(f"  │{'R64':<{W+2}}  {'R32':<{W+2}}  {'Sweet 16':<{W+2}}  {'Elite 8':<{W+2}} │")
        print(f"  │{'':{W+2}}  {'':{W+2}}  {'':{W+2}}  {'':{W+2}} │")

        # Build 8-line display: each line shows R64 winner, and where applicable R32/S16/E8
        for i in range(8):
            r64_str = _fmt_team(r64[i][0], r64[i][2], W)
            wp64 = f"{r64[i][3]:.0%}"

            r32_str = ""
            wp32 = ""
            if i % 2 == 0 and i // 2 < len(r32):
                r32_str = _fmt_team(r32[i // 2][0], r32[i // 2][2], W)
                wp32 = f"{r32[i // 2][3]:.0%}"

            s16_str = ""
            wp16 = ""
            if i % 4 == 0 and i // 4 < len(s16):
                s16_str = _fmt_team(s16[i // 4][0], s16[i // 4][2], W)
                wp16 = f"{s16[i // 4][3]:.0%}"

            e8_str = ""
            wp_e8 = ""
            if i == 0 and e8:
                e8_str = _fmt_team(e8[0][0], e8[0][2], W)
                wp_e8 = f"{e8[0][3]:.0%}"

            # Connectors
            r64_col = f"{r64_str} {wp64:>4}"
            r32_col = f"{r32_str} {wp32:>4}" if r32_str else " " * (W + 5)
            s16_col = f"{s16_str} {wp16:>4}" if s16_str else " " * (W + 5)
            e8_col = f"{e8_str} {wp_e8:>4}" if e8_str else " " * (W + 5)

            # Draw bracket lines
            r64_conn = "─┐" if i % 2 == 0 else " │"
            r32_conn = "─┐" if i % 2 == 0 and r32_str else ("─┘" if i % 2 == 1 and i // 2 < len(r32) else "  ")
            s16_conn = "─┐" if i % 4 == 0 and s16_str else ("─┘" if i == 3 and len(s16) > 0 else ("─┘" if i == 7 and len(s16) > 1 else "  "))
            e8_conn = "──" if i == 0 and e8_str else "  "

            # Simplified: just show the grid
            line = f"  │ {r64_str} {wp64:>3}"
            if i % 2 == 0 and i // 2 < len(r32):
                line += f"  {r32_str} {wp32:>3}"
            else:
                line += f"  {'':>{W + 4}}"
            if i % 4 == 0 and i // 4 < len(s16):
                line += f"  {s16_str} {wp16:>3}"
            else:
                line += f"  {'':>{W + 4}}"
            if i == 0 and e8:
                line += f"  {e8_str} {wp_e8:>3}"

            # Trim trailing spaces and add box border
            print(f"{line.rstrip()}")

        print(f"  └{'─' * 77}┘")

    # Final Four
    print()
    print(f"  ┌─── Final Four ──────────────────────────────────────────────────────────┐")

    for r1, r2 in f4_pairings:
        c1 = region_champions.get(r1)
        c2 = region_champions.get(r2)
        if c1 and c2:
            p = _get_pred(preds, c1[1], c2[1])
            if p >= 0.5:
                winner = c1
                wp = p
            else:
                winner = c2
                wp = 1 - p
            print(f"  │  {_fmt_team(c1[0], c1[2])} [{r1}]  vs  {_fmt_team(c2[0], c2[2])} [{r2}]")
            print(f"  │  Winner: {_fmt_team(winner[0], winner[2])} ({wp:.0%})")
            print(f"  │")

    # Championship
    f4_winners = []
    for r1, r2 in f4_pairings:
        c1 = region_champions.get(r1)
        c2 = region_champions.get(r2)
        if c1 and c2:
            p = _get_pred(preds, c1[1], c2[1])
            f4_winners.append(c1 if p >= 0.5 else c2)

    if len(f4_winners) == 2:
        a, b = f4_winners
        p = _get_pred(preds, a[1], b[1])
        if p >= 0.5:
            champ, wp = a, p
        else:
            champ, wp = b, 1 - p
        print(f"  │  Championship:")
        print(f"  │  {_fmt_team(a[0], a[2])}  vs  {_fmt_team(b[0], b[2])}")
        print(f"  │")
        fut = get_futures(futures_signals, champ[1])
        market_str = f"  (Kalshi: {fut['ChampionshipProb']:.0%})" if fut else ""
        print(f"  │  Champion: {_fmt_team(champ[0], champ[2])} ({wp:.0%}){market_str}")

    print(f"  └{'─' * 77}┘")

    # Advancement table
    print()
    print(f"  ┌─── Advancement Probabilities (Top 16) ──────────────────────────────────┐")
    print(f"  │  {'Team':<25} {'Rgn':>3}  {'R32':>5} {'S16':>5} {'E8':>5} {'F4':>5} {'Chmp':>5} │")
    print(f"  │  {'─' * 55}  │")

    # Compute all region probs
    all_probs: list[tuple[str, int, int, str, dict]] = []
    for region in region_order:
        rteams = bracket[region]
        rprobs = compute_region_probs(preds, rteams, matcher)
        for tid, probs in rprobs.items():
            s = get_seed(seeds, tid, gender) or 99
            all_probs.append((region, s, tid, matcher.get_name(tid), probs))

    all_probs.sort(key=lambda x: x[4].get("F4", 0), reverse=True)

    for region, s, tid, name, probs in all_probs[:16]:
        team_str = f"({s:>2}) {name}"
        print(f"  │  {team_str:<25} {region:>3}"
              f"  {probs.get('R32', 0):>4.0%}"
              f"  {probs.get('S16', 0):>4.0%}"
              f"  {probs.get('E8', 0):>4.0%}"
              f"  {probs.get('F4', 0):>4.0%}"
              f"  {'':>5} │")

    print(f"  └{'─' * 77}┘")
    print()


# ── CLI entry point ──────────────────────────────────────────────

def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else None

    if cmd == "upsets":
        sub = argparse.ArgumentParser(usage="mm-predict upsets [SEED] [-g M|W] [--all]")
        sub.add_argument("_cmd", help=argparse.SUPPRESS)
        sub.add_argument("seed", nargs="?", type=int, default=None,
                         help="Filter to a specific seed (e.g., 12)")
        sub.add_argument("-g", "--gender", default="M", choices=["M", "W"])
        sub.add_argument("--all", action="store_true",
                         help="Show all possible matchups, not just actual bracket")
        args = sub.parse_args()

        matcher, ratings, preds, seeds, slots, market_signals, futures_signals = load_data()
        find_upsets(matcher, preds, seeds, args.gender, args.seed, args.all)

    elif cmd == "path":
        sub = argparse.ArgumentParser(usage="mm-predict path TEAM [-g M|W]")
        sub.add_argument("_cmd", help=argparse.SUPPRESS)
        sub.add_argument("team", help="Team name")
        sub.add_argument("-g", "--gender", default="M", choices=["M", "W"])
        args = sub.parse_args()

        matcher, ratings, preds, seeds, slots, market_signals, futures_signals = load_data()
        show_path(matcher, preds, seeds, slots, futures_signals, args.team, args.gender)

    elif cmd == "bracket":
        sub = argparse.ArgumentParser(usage="mm-predict bracket [-g M|W]")
        sub.add_argument("_cmd", help=argparse.SUPPRESS)
        sub.add_argument("-g", "--gender", default="M", choices=["M", "W"])
        args = sub.parse_args()

        matcher, ratings, preds, seeds, slots, market_signals, futures_signals = load_data()
        show_bracket(matcher, preds, seeds, slots, futures_signals, args.gender)

    else:
        parser = argparse.ArgumentParser(
            description="March Madness matchup predictor",
            usage="mm-predict {TEAM1 TEAM2 | path TEAM | upsets [SEED] | bracket}",
        )
        parser.add_argument("team1", help="First team name")
        parser.add_argument("team2", help="Second team name")
        parser.add_argument("-g", "--gender", default="M", choices=["M", "W"])
        parser.add_argument("-v", "--verbose", action="store_true")
        args = parser.parse_args()

        matcher, ratings, preds, seeds, slots, market_signals, futures_signals = load_data()
        predict_matchup(
            matcher, ratings, preds, seeds, market_signals, futures_signals,
            args.team1, args.team2, args.gender, args.verbose,
        )


if __name__ == "__main__":
    main()

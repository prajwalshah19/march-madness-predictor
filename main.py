#!/usr/bin/env python3
"""March Madness 2026 Prediction Pipeline — Main Orchestrator.

Artifact-aware pipeline: each stage saves outputs to disk and reuses them
on subsequent runs. Use --fresh to force recomputation of all stages,
or delete individual output files to recompute specific stages.

Usage:
    python main.py              # Use cached artifacts where available
    python main.py --fresh      # Recompute everything from scratch
"""

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from src.backtest import run_backtest
from src.base_model import MarchMadnessModel
from src.blend import blend_predictions
from src.bracket import generate_advancement_csv, generate_bracket
from src.config import CFG
from src.data_loader import DataLoader
from src.efficiency import compute_efficiency
from src.elo import compute_elo
from src.features import compute_tournament_features
from src.market_features import analyze_order_books
from src.market_scraper import fetch_market_data, market_data_available
from src.submission import generate_submission
from src.team_matcher import TeamMatcher

# ── Artifact paths ───────────────────────────────────────────────

TEAM_RATINGS_PATH = CFG.OUTPUT_DIR / "team_ratings.csv"
BACKTEST_RESULTS_PATH = CFG.OUTPUT_DIR / "backtest_results.json"
BASE_PREDICTIONS_PATH = CFG.OUTPUT_DIR / "base_predictions.csv"
MARKET_SIGNALS_PATH = CFG.OUTPUT_DIR / "market_signals.csv"
FUTURES_SIGNALS_PATH = CFG.OUTPUT_DIR / "futures_signals.csv"
BLENDED_PREDICTIONS_PATH = CFG.OUTPUT_DIR / "blended_predictions.csv"
MARKET_LATEST_PATH = CFG.MARKET_DIR / "latest.json"


def _cached(path: Path, fresh: bool) -> bool:
    """Check if an artifact exists and is usable."""
    return path.exists() and not fresh


def _stage_timer(name: str):
    """Context manager that prints stage timing."""
    class Timer:
        def __init__(self):
            self.start = 0.0
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, *args):
            elapsed = time.perf_counter() - self.start
            print(f"  [{elapsed:.1f}s]")
    return Timer()


def main() -> None:
    """Run the full prediction pipeline."""
    parser = argparse.ArgumentParser(description="March Madness prediction pipeline")
    parser.add_argument("--fresh", action="store_true",
                        help="Recompute all stages, ignoring cached artifacts")
    parser.add_argument("--skip-brackets", action="store_true",
                        help="Skip bracket generation (Stage 10) for faster iteration")
    args = parser.parse_args()
    fresh = args.fresh

    print("=" * 60)
    print("March Machine Learning Mania 2026 — Prediction Pipeline")
    if not fresh:
        print("  (using cached artifacts where available; --fresh to recompute)")
    print("=" * 60)

    CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── STAGE 1: Load data ──────────────────────────────────────
    print("\nSTAGE 1: Loading Kaggle data...")
    with _stage_timer("load"):
        try:
            loader = DataLoader(CFG.DATA_DIR)
        except FileNotFoundError as e:
            print(f"\n  ERROR: {e}")
            print(f"  Please download Kaggle CSVs to {CFG.DATA_DIR}/")
            print("  https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data")
            sys.exit(1)

        teams = loader.get_teams()
        print(f"  Loaded {len(teams)} teams, seasons {loader.min_season}-{loader.max_season}")

    # ── STAGES 2-4: Elo + Efficiency + Tournament Features ──────
    if _cached(TEAM_RATINGS_PATH, fresh):
        print(f"\nSTAGES 2-4: Loading cached team ratings from {TEAM_RATINGS_PATH}")
        team_ratings = pd.read_csv(TEAM_RATINGS_PATH)
        print(f"  {len(team_ratings)} team-season rows loaded")
    else:
        print("\nSTAGE 2: Computing Elo ratings...")
        with _stage_timer("elo"):
            elo_ratings = compute_elo(loader)

            latest_season = elo_ratings["Season"].max()
            for gender, label in [("M", "Men's"), ("W", "Women's")]:
                top = (
                    elo_ratings[(elo_ratings["Season"] == latest_season) & (elo_ratings["Gender"] == gender)]
                    .nlargest(5, "EloPreTourney")
                )
                if not top.empty:
                    top_names = top.merge(teams[["TeamID", "TeamName"]], on="TeamID", how="left")
                    names = ", ".join(
                        f"{r['TeamName']} ({r['EloPreTourney']:.0f})"
                        for _, r in top_names.iterrows()
                        if "TeamName" in r and not str(r.get("TeamName", "")) == "nan"
                    )
                    print(f"  Top 5 {label} Elo ({latest_season}): {names}")

        print("\nSTAGE 3: Computing adjusted efficiency...")
        with _stage_timer("efficiency"):
            team_ratings = compute_efficiency(loader, elo_ratings)

        print("\nSTAGE 4: Computing tournament-specific features...")
        with _stage_timer("features"):
            team_ratings = compute_tournament_features(loader, team_ratings)

        team_ratings.to_csv(TEAM_RATINGS_PATH, index=False)
        print(f"  Team ratings saved to {TEAM_RATINGS_PATH}")

    # ── STAGE 5: Backtest ───────────────────────────────────────
    if _cached(BACKTEST_RESULTS_PATH, fresh):
        print(f"\nSTAGE 5: Loading cached backtest from {BACKTEST_RESULTS_PATH}")
        with open(BACKTEST_RESULTS_PATH) as f:
            backtest_results = json.load(f)
        if "final_model" in backtest_results:
            print(f"  Brier: {backtest_results['final_model']['avg_brier']:.4f}, "
                  f"features: {backtest_results['final_model']['features_kept']}")
    else:
        print("\nSTAGE 5: Backtesting base model (2010-2025)...")
        with _stage_timer("backtest"):
            backtest_results = run_backtest(loader, team_ratings)

    # Determine best features from backtest
    final_features = ["elo_diff", "seed_diff"]
    if "final_model" in backtest_results:
        final_features = backtest_results["final_model"].get("features_kept", final_features)
        if not _cached(BACKTEST_RESULTS_PATH, fresh):
            print(f"  Final model Brier: {backtest_results['final_model']['avg_brier']:.4f}")
            print(f"  Features kept: {final_features}")

    # ── STAGE 6: Train final base model ─────────────────────────
    if _cached(BASE_PREDICTIONS_PATH, fresh):
        print(f"\nSTAGE 6: Loading cached base predictions from {BASE_PREDICTIONS_PATH}")
        base_preds = pd.read_csv(BASE_PREDICTIONS_PATH)
        print(f"  {len(base_preds)} matchups loaded")
    else:
        print("\nSTAGE 6: Training base model on all data...")
        with _stage_timer("train"):
            model = MarchMadnessModel(features=final_features)
            train_info = model.train(loader, team_ratings)
            print(f"  Trained on {train_info['n_games']} games, "
                  f"train Brier: {train_info['train_brier']:.4f}")

            base_preds = model.predict_all_matchups(
                loader, team_ratings, season=CFG.TARGET_SEASON
            )
            base_preds.to_csv(BASE_PREDICTIONS_PATH, index=False)
            print(f"  Base predictions: {len(base_preds)} matchups")

    # ── STAGES 7-9: Market data + blending ───────────────────────
    matchup_signals = None
    futures_signals = None

    # Stage 7: Get market data (API or cached)
    market_data = None
    if market_data_available():
        print("\nSTAGE 7: Fetching market data...")
        with _stage_timer("market_fetch"):
            market_data = fetch_market_data()
    elif _cached(MARKET_LATEST_PATH, fresh):
        print(f"\nSTAGE 7: Loading cached market snapshot from {MARKET_LATEST_PATH}")
        with open(MARKET_LATEST_PATH) as f:
            market_data = json.load(f)
        print(f"  Snapshot from {market_data.get('timestamp', 'unknown')}")
    else:
        print("\nSTAGE 7: No market data (no API keys, no cached snapshot)")

    # Stage 8: Analyze market data
    if market_data is not None:
        if (_cached(MARKET_SIGNALS_PATH, fresh)
                and not market_data_available()):
            # Use cached signals only when we didn't just fetch fresh data
            print(f"\nSTAGE 8: Loading cached market signals")
            matchup_signals = pd.read_csv(MARKET_SIGNALS_PATH)
            if _cached(FUTURES_SIGNALS_PATH, fresh):
                futures_signals = pd.read_csv(FUTURES_SIGNALS_PATH)
            print(f"  {len(matchup_signals)} matchup signals, "
                  f"{len(futures_signals) if futures_signals is not None else 0} futures signals")
        else:
            print("\nSTAGE 8: Analyzing order books...")
            with _stage_timer("analyze"):
                matcher = TeamMatcher(CFG.DATA_DIR)
                matchup_signals, futures_signals = analyze_order_books(market_data, matcher)
                matchup_signals.to_csv(MARKET_SIGNALS_PATH, index=False)
                if not futures_signals.empty:
                    futures_signals.to_csv(FUTURES_SIGNALS_PATH, index=False)

    # Stage 9: Blend
    print("\nSTAGE 9: Blending predictions...")
    with _stage_timer("blend"):
        final_preds = blend_predictions(base_preds, matchup_signals, futures_signals)

    final_preds.to_csv(BLENDED_PREDICTIONS_PATH, index=False)

    # ── STAGE 10: Generate brackets ─────────────────────────────
    if args.skip_brackets:
        print("\nSTAGE 10: Skipped (--skip-brackets)")
    else:
        print("\nSTAGE 10: Generating mock brackets...")
        with _stage_timer("brackets"):
            for gender, label in [("M", "men"), ("W", "women")]:
                try:
                    bracket_output = str(CFG.OUTPUT_DIR / f"bracket_{label}.txt")
                    bracket_str = generate_bracket(
                        loader, final_preds, gender=gender, output=bracket_output,
                        season=CFG.TARGET_SEASON,
                    )
                    lines = bracket_str.split("\n")
                    contender_section = False
                    count = 0
                    for line in lines:
                        if "TOP CHAMPIONSHIP" in line:
                            contender_section = True
                            print(f"\n  {label.title()} {line.strip()}")
                            continue
                        if contender_section and line.strip().startswith("("):
                            print(f"  {line}")
                            count += 1
                            if count >= 5:
                                break
                except Exception as e:
                    print(f"  WARNING: Could not generate {label}'s bracket: {e}")

            try:
                generate_advancement_csv(
                    loader, final_preds, season=CFG.TARGET_SEASON,
                    output=str(CFG.OUTPUT_DIR / "team_advancement_probs.csv"),
                )
            except Exception as e:
                print(f"  WARNING: Could not generate advancement CSV: {e}")

    # ── STAGE 11: Generate submission ───────────────────────────
    print("\nSTAGE 11: Generating submission...")
    with _stage_timer("submission"):
        try:
            generate_submission(
                loader, final_preds, output=str(CFG.OUTPUT_DIR / "submission.csv"),
            )
        except Exception as e:
            print(f"  WARNING: Could not generate submission: {e}")
            print("  You may need SampleSubmissionStage2.csv in your data directory.")

    # ── Done ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE. Files in output/:")
    print("  submission.csv          — Upload to Kaggle")
    print("  team_ratings.csv        — Team Elo + efficiency + features")
    print("  base_predictions.csv    — Layer 1 probabilities")
    if matchup_signals is not None:
        print("  market_signals.csv      — Layer 2 matchup market signals")
        print("  futures_signals.csv     — Layer 2 championship futures")
    print("  blended_predictions.csv — Final blended probabilities")
    print("  bracket_men.txt         — Men's mock bracket")
    print("  bracket_women.txt       — Women's mock bracket")
    print("  backtest_results.json   — Historical validation")
    print("=" * 60)


if __name__ == "__main__":
    main()

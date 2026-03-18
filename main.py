#!/usr/bin/env python3
"""March Madness 2026 Prediction Pipeline — Main Orchestrator."""

import sys
from pathlib import Path

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


def main() -> None:
    """Run the full prediction pipeline."""
    print("=" * 60)
    print("March Machine Learning Mania 2026 — Prediction Pipeline")
    print("=" * 60)

    # Ensure output directory exists
    CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── STAGE 1: Load data ──────────────────────────────────────
    print("\nSTAGE 1: Loading Kaggle data...")
    try:
        loader = DataLoader(CFG.DATA_DIR)
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        print(f"  Please download Kaggle CSVs to {CFG.DATA_DIR}/")
        print("  https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data")
        sys.exit(1)

    teams = loader.get_teams()
    print(f"  Loaded {len(teams)} teams, seasons {loader.min_season}-{loader.max_season}")

    # ── STAGE 2: Compute Elo ratings ────────────────────────────
    print("\nSTAGE 2: Computing Elo ratings...")
    elo_ratings = compute_elo(loader)

    # Print top teams for sanity check
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

    # ── STAGE 3: Compute efficiency ─────────────────────────────
    print("\nSTAGE 3: Computing adjusted efficiency...")
    team_ratings = compute_efficiency(loader, elo_ratings)

    # ── STAGE 4: Compute tournament features ────────────────────
    print("\nSTAGE 4: Computing tournament-specific features...")
    team_ratings = compute_tournament_features(loader, team_ratings)

    # Save team ratings
    team_ratings.to_csv(CFG.OUTPUT_DIR / "team_ratings.csv", index=False)
    print(f"  Team ratings saved to {CFG.OUTPUT_DIR / 'team_ratings.csv'}")

    # ── STAGE 5: Backtest ───────────────────────────────────────
    print("\nSTAGE 5: Backtesting base model (2010-2025)...")
    backtest_results = run_backtest(loader, team_ratings)

    # Determine best features from backtest
    final_features = ["elo_diff", "seed_diff"]
    if "final_model" in backtest_results:
        final_features = backtest_results["final_model"].get("features_kept", final_features)
        print(f"  Final model Brier: {backtest_results['final_model']['avg_brier']:.4f}")
        print(f"  Features kept: {final_features}")

    # ── STAGE 6: Train final base model ─────────────────────────
    print("\nSTAGE 6: Training base model on all data...")
    model = MarchMadnessModel(features=final_features)
    train_info = model.train(loader, team_ratings)
    print(f"  Trained on {train_info['n_games']} games, train Brier: {train_info['train_brier']:.4f}")

    base_preds = model.predict_all_matchups(loader, team_ratings, season=CFG.TARGET_SEASON)
    base_preds.to_csv(CFG.OUTPUT_DIR / "base_predictions.csv", index=False)
    print(f"  Base predictions: {len(base_preds)} matchups")

    # ── STAGE 7-9: Market data (optional) ───────────────────────
    if market_data_available():
        print("\nSTAGE 7: Fetching market data...")
        market_data = fetch_market_data()

        print("\nSTAGE 8: Analyzing order books...")
        market_signals = analyze_order_books(market_data)
        market_signals.to_csv(CFG.OUTPUT_DIR / "market_signals.csv", index=False)

        print("\nSTAGE 9: Blending predictions...")
        final_preds = blend_predictions(base_preds, market_signals)
    else:
        print("\nSTAGE 7-9: Skipping market layer (no API keys configured)")
        final_preds = blend_predictions(base_preds)  # Pass through with proper columns

    final_preds.to_csv(CFG.OUTPUT_DIR / "blended_predictions.csv", index=False)

    # ── STAGE 10: Generate brackets ─────────────────────────────
    print("\nSTAGE 10: Generating mock brackets...")
    for gender, label in [("M", "men"), ("W", "women")]:
        try:
            bracket_output = str(CFG.OUTPUT_DIR / f"bracket_{label}.txt")
            bracket_str = generate_bracket(
                loader, final_preds, gender=gender, output=bracket_output,
                season=CFG.TARGET_SEASON,
            )
            # Print top contenders
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

    # Generate advancement probability CSV
    try:
        generate_advancement_csv(
            loader, final_preds, season=CFG.TARGET_SEASON,
            output=str(CFG.OUTPUT_DIR / "team_advancement_probs.csv"),
        )
    except Exception as e:
        print(f"  WARNING: Could not generate advancement CSV: {e}")

    # ── STAGE 11: Generate submission ───────────────────────────
    print("\nSTAGE 11: Generating submission...")
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
    if market_data_available():
        print("  market_signals.csv      — Layer 2 market analysis")
    print("  blended_predictions.csv — Final blended probabilities")
    print("  bracket_men.txt         — Men's mock bracket")
    print("  bracket_women.txt       — Women's mock bracket")
    print("  backtest_results.json   — Historical validation")
    print("=" * 60)


if __name__ == "__main__":
    main()

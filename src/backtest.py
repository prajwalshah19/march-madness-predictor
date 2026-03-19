"""Historical tournament backtesting engine with feature ablation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.base_model import MarchMadnessModel, build_training_data
from src.config import CFG
from src.data_loader import DataLoader


def _brier_score(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Compute Brier score (lower is better)."""
    return float(np.mean((predictions - actuals) ** 2))


def run_backtest(
    loader: DataLoader,
    team_ratings: pd.DataFrame,
    feature_sets: Optional[dict[str, list[str]]] = None,
) -> dict:
    """Run leave-one-season-out cross-validation on historical tournaments.

    Args:
        loader: Data loader.
        team_ratings: DataFrame with all team ratings and features.
        feature_sets: Dict mapping name -> feature list for ablation.
            If None, uses default ablation study.

    Returns:
        Dict with backtest results per feature set and final recommendation.
    """
    if feature_sets is None:
        feature_sets = _default_ablation_sets(team_ratings)

    tourney = loader.get_tourney_compact()
    seeds = loader.get_seeds()
    available_seasons = sorted(tourney["Season"].unique())

    # Filter to backtest range
    test_seasons = [
        s for s in available_seasons
        if CFG.BACKTEST_START_YEAR <= s <= CFG.BACKTEST_END_YEAR
        and s not in CFG.BACKTEST_SKIP_YEARS
    ]

    if not test_seasons:
        print("  WARNING: No seasons available for backtesting")
        return {}

    print(f"  Backtesting across {len(test_seasons)} seasons: {test_seasons[0]}-{test_seasons[-1]}")

    results: dict = {}

    for set_name, features in feature_sets.items():
        print(f"  Testing feature set: {set_name} ({features})")
        by_year: dict[str, float] = {}
        all_preds = []
        all_actuals = []

        for held_out_season in test_seasons:
            # Train on all other seasons
            model = MarchMadnessModel(features=features)
            try:
                model.train(loader, team_ratings, exclude_seasons={held_out_season})
            except ValueError:
                continue

            # Predict held-out tournament games
            season_tourney = tourney[tourney["Season"] == held_out_season]
            season_preds = []
            season_actuals = []

            for _, game in season_tourney.iterrows():
                gender = game["Gender"]
                w_id = int(game["WTeamID"])
                l_id = int(game["LTeamID"])
                team_a = min(w_id, l_id)
                team_b = max(w_id, l_id)
                actual = 1 if team_a == w_id else 0

                pred = model.predict_matchup(
                    held_out_season, gender, team_a, team_b, team_ratings, seeds
                )
                season_preds.append(pred)
                season_actuals.append(actual)

            if season_preds:
                season_brier = _brier_score(
                    np.array(season_preds), np.array(season_actuals)
                )
                by_year[str(held_out_season)] = round(season_brier, 4)
                all_preds.extend(season_preds)
                all_actuals.extend(season_actuals)

        if all_preds:
            avg_brier = _brier_score(np.array(all_preds), np.array(all_actuals))
            results[set_name] = {
                "features": features,
                "avg_brier": round(avg_brier, 4),
                "by_year": by_year,
                "n_games": len(all_preds),
            }
            print(f"    Average Brier: {avg_brier:.4f} ({len(all_preds)} games)")

    # Determine best feature set
    if results:
        best_name = min(results, key=lambda k: results[k]["avg_brier"])
        results["final_model"] = {
            "best_set": best_name,
            "features_kept": results[best_name]["features"],
            "avg_brier": results[best_name]["avg_brier"],
        }
        print(f"\n  Best feature set: {best_name}")
        print(f"  Features kept: {results[best_name]['features']}")
        print(f"  Average Brier: {results[best_name]['avg_brier']:.4f}")

    # Save results
    output_path = CFG.OUTPUT_DIR / "backtest_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {output_path}")

    # Print summary table
    _print_summary_table(results, test_seasons)

    return results


def _default_ablation_sets(team_ratings: pd.DataFrame) -> dict[str, list[str]]:
    """Build default feature ablation sets based on available columns."""
    sets: dict[str, list[str]] = {
        "baseline_elo_seed": ["elo_diff", "seed_diff"],
    }

    # Add efficiency if available
    if "NetEff" in team_ratings.columns:
        sets["with_efficiency"] = ["elo_diff", "seed_diff", "eff_diff"]

    # Add tournament-specific features if available
    if "Opp3PtPct" in team_ratings.columns:
        prev_best = list(sets[max(sets.keys())]) if sets else ["elo_diff", "seed_diff"]
        sets["with_3pt_def"] = prev_best + ["opp3ptpct_diff"]

    if "CloseGameTOMargin" in team_ratings.columns:
        prev_best = list(sets[max(sets.keys())]) if sets else ["elo_diff", "seed_diff"]
        sets["with_close_to"] = prev_best + ["closegametomargin_diff"]

    if "OffRebRate" in team_ratings.columns:
        prev_best = list(sets[max(sets.keys())]) if sets else ["elo_diff", "seed_diff"]
        sets["with_off_reb"] = prev_best + ["offrebrate_diff"]

    if "LateFTPct" in team_ratings.columns:
        prev_best = list(sets[max(sets.keys())]) if sets else ["elo_diff", "seed_diff"]
        sets["with_late_ft"] = prev_best + ["lateftpct_diff"]

    # Market log-odds from Massey composite (historical proxy for market signal)
    if "MasseyLogOdds" in team_ratings.columns and (team_ratings["MasseyLogOdds"] != 0).any():
        prev_best = list(sets[max(sets.keys())]) if sets else ["elo_diff", "seed_diff"]
        sets["with_market_logodds"] = prev_best + ["market_logodds_diff"]

    return sets


def _print_summary_table(results: dict, test_seasons: list[int]) -> None:
    """Print a formatted summary table of backtest results."""
    if not results:
        return

    # Exclude 'final_model' meta-entry
    set_names = [k for k in results if k != "final_model"]
    if not set_names:
        return

    print("\n  " + "=" * 70)
    print("  BACKTEST SUMMARY")
    print("  " + "=" * 70)

    # Header
    header = f"  {'Feature Set':<25} {'Avg Brier':>10} {'Games':>8}"
    print(header)
    print("  " + "-" * 45)

    for name in set_names:
        r = results[name]
        marker = " <-- BEST" if results.get("final_model", {}).get("best_set") == name else ""
        print(f"  {name:<25} {r['avg_brier']:>10.4f} {r['n_games']:>8}{marker}")

    print("  " + "=" * 70)

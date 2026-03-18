"""Base prediction model using logistic regression on feature differentials."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import CFG
from src.data_loader import DataLoader


# Default feature columns (v1: just Elo + seed)
DEFAULT_FEATURES = ["elo_diff", "seed_diff"]


def build_training_data(
    loader: DataLoader,
    team_ratings: pd.DataFrame,
    features: list[str] = DEFAULT_FEATURES,
    exclude_seasons: Optional[set[int]] = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build training dataset from historical tournament games.

    Returns:
        Tuple of (feature_df, X array, y array) where each row is a matchup
        with TeamA = lower TeamID.
    """
    tourney = loader.get_tourney_compact()
    seeds = loader.get_seeds()

    if tourney.empty:
        raise ValueError("No tournament data available")

    if exclude_seasons:
        tourney = tourney[~tourney["Season"].isin(exclude_seasons)]

    rows = []
    for _, game in tourney.iterrows():
        season = game["Season"]
        gender = game["Gender"]
        w_id = int(game["WTeamID"])
        l_id = int(game["LTeamID"])

        # Ensure lower TeamID is always TeamA (to match submission format)
        team_a = min(w_id, l_id)
        team_b = max(w_id, l_id)
        target = 1 if team_a == w_id else 0

        row = _build_matchup_features(
            season, gender, team_a, team_b, team_ratings, seeds, features
        )
        if row is not None:
            row["target"] = target
            row["Season"] = season
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No valid training rows could be built")

    feature_cols = [f for f in features if f in df.columns]
    X = df[feature_cols].values
    y = df["target"].values
    return df, X, y


def _build_matchup_features(
    season: int,
    gender: str,
    team_a: int,
    team_b: int,
    team_ratings: pd.DataFrame,
    seeds: pd.DataFrame,
    features: list[str],
) -> Optional[dict]:
    """Build feature dictionary for a single matchup (TeamA = lower ID)."""
    # Look up ratings
    r_a = team_ratings[
        (team_ratings["Season"] == season)
        & (team_ratings["TeamID"] == team_a)
        & (team_ratings["Gender"] == gender)
    ]
    r_b = team_ratings[
        (team_ratings["Season"] == season)
        & (team_ratings["TeamID"] == team_b)
        & (team_ratings["Gender"] == gender)
    ]

    if r_a.empty or r_b.empty:
        return None

    r_a = r_a.iloc[0]
    r_b = r_b.iloc[0]

    row: dict = {"Gender": gender, "TeamA": team_a, "TeamB": team_b}

    # Elo differential
    if "elo_diff" in features:
        row["elo_diff"] = r_a.get("EloPreTourney", 1500) - r_b.get("EloPreTourney", 1500)

    # Seed differential
    if "seed_diff" in features:
        s_a = seeds[
            (seeds["Season"] == season) & (seeds["TeamID"] == team_a) & (seeds["Gender"] == gender)
        ]
        s_b = seeds[
            (seeds["Season"] == season) & (seeds["TeamID"] == team_b) & (seeds["Gender"] == gender)
        ]
        seed_a = int(s_a.iloc[0]["SeedNum"]) if not s_a.empty else 8
        seed_b = int(s_b.iloc[0]["SeedNum"]) if not s_b.empty else 8
        row["seed_diff"] = seed_a - seed_b

    # Efficiency differential
    if "eff_diff" in features:
        row["eff_diff"] = r_a.get("NetEff", 0) - r_b.get("NetEff", 0)

    # Tournament-specific feature diffs
    for feat in ["Opp3PtPct", "CloseGameTOMargin", "OffRebRate", "LateFTPct"]:
        col = f"{feat.lower()}_diff"
        if col in features:
            val_a = r_a.get(feat, 0)
            val_b = r_b.get(feat, 0)
            if feat == "Opp3PtPct":
                # Lower is better for 3pt defense, so negate: A wants lower
                row[col] = val_b - val_a
            else:
                row[col] = val_a - val_b

    return row


class MarchMadnessModel:
    """Logistic regression model for March Madness predictions."""

    def __init__(self, features: list[str] = DEFAULT_FEATURES) -> None:
        self.features = features
        self.scaler = StandardScaler()
        self.model: Optional[CalibratedClassifierCV] = None

    def train(self, loader: DataLoader, team_ratings: pd.DataFrame,
              exclude_seasons: Optional[set[int]] = None) -> dict:
        """Train the model on historical tournament data.

        Returns dict with training info.
        """
        df, X, y = build_training_data(
            loader, team_ratings, self.features, exclude_seasons
        )

        X_scaled = self.scaler.fit_transform(X)

        base_lr = LogisticRegression(max_iter=1000, random_state=42)
        self.model = CalibratedClassifierCV(base_lr, method="isotonic", cv=5)
        self.model.fit(X_scaled, y)

        train_preds = self.predict_proba(X)
        train_brier = np.mean((train_preds - y) ** 2)

        return {
            "n_games": len(y),
            "train_brier": round(float(train_brier), 4),
            "win_rate": round(float(y.mean()), 4),
            "features": self.features,
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict P(TeamA wins) for feature matrix X."""
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[:, 1]
        return np.clip(probs, CFG.CLIP_LOW, CFG.CLIP_HIGH)

    def predict_matchup(
        self,
        season: int,
        gender: str,
        team_a: int,
        team_b: int,
        team_ratings: pd.DataFrame,
        seeds: pd.DataFrame,
    ) -> float:
        """Predict P(team_a wins) for a single matchup."""
        row = _build_matchup_features(
            season, gender, team_a, team_b, team_ratings, seeds, self.features
        )
        if row is None:
            return 0.5

        feature_cols = [f for f in self.features if f in row]
        X = np.array([[row[f] for f in feature_cols]])
        return float(self.predict_proba(X)[0])

    def predict_all_matchups(
        self,
        loader: DataLoader,
        team_ratings: pd.DataFrame,
        season: int = CFG.TARGET_SEASON,
        stage: int = 2,
    ) -> pd.DataFrame:
        """Generate predictions for all matchups in the submission template."""
        try:
            template = loader.get_submission_template(stage)
        except KeyError:
            print(f"  WARNING: No submission template for stage {stage}, trying stage 1")
            template = loader.get_submission_template(1)

        seeds = loader.get_seeds()
        predictions = []

        for _, row in template.iterrows():
            match_id = row["ID"]
            parts = match_id.split("_")
            match_season = int(parts[0])
            team_a = int(parts[1])
            team_b = int(parts[2])

            # Determine gender from TeamID range
            if CFG.MENS_ID_MIN <= team_a <= CFG.MENS_ID_MAX:
                gender = "M"
            else:
                gender = "W"

            pred = self.predict_matchup(
                match_season, gender, team_a, team_b, team_ratings, seeds
            )
            predictions.append({"ID": match_id, "Pred": round(pred, 6)})

        return pd.DataFrame(predictions)

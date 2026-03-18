"""Generate final Kaggle submission CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import CFG
from src.data_loader import DataLoader


def generate_submission(
    loader: DataLoader,
    predictions: pd.DataFrame,
    output: Optional[str] = None,
    stage: int = 2,
) -> pd.DataFrame:
    """Generate submission CSV from predictions.

    Args:
        loader: Data loader (for submission template).
        predictions: DataFrame with columns ID, Pred (or BasePred/BlendedPred).
        output: Output file path. Defaults to output/submission.csv.
        stage: Submission stage (1 for validation, 2 for final).

    Returns:
        Submission DataFrame with columns ID, Pred.
    """
    # Get template
    try:
        template = loader.get_submission_template(stage)
    except KeyError:
        print(f"  WARNING: Stage {stage} template not found, trying stage 1")
        template = loader.get_submission_template(1)

    required_ids = set(template["ID"])

    # Determine prediction column
    pred_col = "Pred"
    for col in ["BlendedPred", "BasePred", "Pred"]:
        if col in predictions.columns:
            pred_col = col
            break

    # Build lookup from predictions
    pred_lookup = dict(zip(predictions["ID"], predictions[pred_col]))

    # Generate submission
    rows = []
    missing_count = 0
    for match_id in template["ID"]:
        if match_id in pred_lookup:
            pred = float(pred_lookup[match_id])
        else:
            pred = 0.5  # Default for missing matchups
            missing_count += 1

        # Clip to valid range
        pred = max(CFG.CLIP_LOW, min(CFG.CLIP_HIGH, pred))
        rows.append({"ID": match_id, "Pred": round(pred, 6)})

    submission = pd.DataFrame(rows)

    # Validation
    assert len(submission) == len(template), (
        f"Row count mismatch: {len(submission)} vs {len(template)} expected"
    )
    assert not submission["Pred"].isna().any(), "NaN predictions found"
    assert (submission["Pred"] >= CFG.CLIP_LOW).all(), "Predictions below clip minimum"
    assert (submission["Pred"] <= CFG.CLIP_HIGH).all(), "Predictions above clip maximum"

    if missing_count > 0:
        print(f"  WARNING: {missing_count} matchups had no prediction (defaulted to 0.5)")

    # Save
    output_path = Path(output) if output else CFG.OUTPUT_DIR / "submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    print(f"  Submission saved: {output_path} ({len(submission)} matchups)")
    print(f"  Pred range: [{submission['Pred'].min():.4f}, {submission['Pred'].max():.4f}]")
    print(f"  Mean pred: {submission['Pred'].mean():.4f}")

    return submission

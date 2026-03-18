#!/usr/bin/env python3
"""March Madness 2026 Prediction Pipeline — Main Orchestrator."""

from src.config import CFG


def main() -> None:
    """Run the full prediction pipeline."""
    print("=" * 60)
    print("March Machine Learning Mania 2026 — Prediction Pipeline")
    print("=" * 60)

    # TODO: Pipeline stages will be wired in as they are implemented.
    print("\nPipeline not yet implemented. Build modules first.")
    print(f"Target season: {CFG.TARGET_SEASON}")
    print(f"Data directory: {CFG.DATA_DIR}")
    print(f"Output directory: {CFG.OUTPUT_DIR}")


if __name__ == "__main__":
    main()

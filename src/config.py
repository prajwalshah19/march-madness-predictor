"""Configuration constants for March Madness predictor."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Global configuration."""

    # Directories
    DATA_DIR: Path = Path("data/kaggle")
    MARKET_DIR: Path = Path("data/market")
    OUTPUT_DIR: Path = Path("output")

    # Target season
    TARGET_SEASON: int = 2026

    # Prediction clipping bounds
    CLIP_LOW: float = 0.03
    CLIP_HIGH: float = 0.97

    # TeamID ranges
    MENS_ID_MIN: int = 1000
    MENS_ID_MAX: int = 1999
    WOMENS_ID_MIN: int = 3000
    WOMENS_ID_MAX: int = 3999

    # Elo parameters
    ELO_K: int = 28
    ELO_INITIAL: int = 1500
    ELO_SEASON_REGRESS: float = 0.75
    ELO_HOME_ADVANTAGE: int = 65

    # Efficiency parameters
    FTA_COEFFICIENT: float = 0.475  # Free throw attempt coefficient for possessions

    # Backtesting
    BACKTEST_START_YEAR: int = 2010
    BACKTEST_END_YEAR: int = 2025
    BACKTEST_SKIP_YEARS: tuple[int, ...] = (2020,)  # No tournament in 2020

    # Blending defaults
    BLEND_BASE_WEIGHT: float = 0.6
    BLEND_MARKET_WEIGHT: float = 0.4
    BLEND_LOW_CONFIDENCE_BASE: float = 0.85
    BLEND_LOW_CONFIDENCE_MARKET: float = 0.15

    # Monte Carlo
    MONTE_CARLO_SIMULATIONS: int = 10_000


CFG = Config()

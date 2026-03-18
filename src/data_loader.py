"""Load, parse, and validate all Kaggle CSV data files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import CFG


@dataclass
class ParsedSeed:
    """Parsed tournament seed."""

    region: str
    seed_num: int
    play_in: bool


def parse_seed(seed_str: str) -> ParsedSeed:
    """Parse a seed string like 'W01' or 'W16a' into components."""
    match = re.match(r"^([WXYZ])(\d{2})([a-z]?)$", seed_str)
    if not match:
        raise ValueError(f"Invalid seed string: {seed_str!r}")
    region = match.group(1)
    seed_num = int(match.group(2))
    play_in = bool(match.group(3))
    return ParsedSeed(region=region, seed_num=seed_num, play_in=play_in)


class DataLoader:
    """Load and provide access to all Kaggle competition CSV files."""

    # Files that exist for both M (men's) and W (women's)
    GENDERED_PREFIXES = {"M", "W"}

    # Known file base names (without M/W prefix)
    KNOWN_FILES = {
        "Teams",
        "Seasons",
        "RegularSeasonCompactResults",
        "RegularSeasonDetailedResults",
        "NCAATourneyCompactResults",
        "NCAATourneyDetailedResults",
        "NCAATourneySeeds",
        "NCAATourneySeedRoundSlots",
        "NCAATourneySlots",
        "GameCities",
        "TeamCoaches",
        "TeamConferences",
        "ConferenceTourneyGames",
        "SecondaryTourneyCompactResults",
        "SecondaryTourneyTeams",
    }

    # Files without gender prefix
    SHARED_FILES = {
        "MasseyOrdinals",
        "Cities",
        "SampleSubmissionStage1",
        "SampleSubmissionStage2",
    }

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else CFG.DATA_DIR
        self._tables: dict[str, pd.DataFrame] = {}
        self._load_all()
        self._validate()
        self._print_summary()

    def _load_all(self) -> None:
        """Scan data directory and load all CSV files."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        csv_files = sorted(self.data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")

        for csv_path in csv_files:
            name = csv_path.stem
            df = pd.read_csv(csv_path)
            self._tables[name] = df

        print(f"  Loaded {len(self._tables)} CSV files from {self.data_dir}")

    def _get_gender(self, team_id: int) -> str:
        """Determine gender from TeamID range."""
        if CFG.MENS_ID_MIN <= team_id <= CFG.MENS_ID_MAX:
            return "M"
        elif CFG.WOMENS_ID_MIN <= team_id <= CFG.WOMENS_ID_MAX:
            return "W"
        else:
            raise ValueError(f"TeamID {team_id} not in valid range")

    def _validate(self) -> None:
        """Run basic validation checks on loaded data."""
        errors: list[str] = []

        # Validate compact results (no negative scores)
        for key in ["MRegularSeasonCompactResults", "WRegularSeasonCompactResults",
                     "MNCAATourneyCompactResults", "WNCAATourneyCompactResults"]:
            if key not in self._tables:
                continue
            df = self._tables[key]
            if (df["WScore"] < 0).any() or (df["LScore"] < 0).any():
                errors.append(f"{key}: contains negative scores")
            if (df["WScore"] < df["LScore"]).any():
                errors.append(f"{key}: winning score less than losing score")

        # Validate TeamIDs are in valid ranges
        for key, df in self._tables.items():
            if "TeamID" in df.columns:
                invalid = df["TeamID"][
                    ~(
                        ((df["TeamID"] >= CFG.MENS_ID_MIN) & (df["TeamID"] <= CFG.MENS_ID_MAX))
                        | ((df["TeamID"] >= CFG.WOMENS_ID_MIN) & (df["TeamID"] <= CFG.WOMENS_ID_MAX))
                    )
                ]
                if len(invalid) > 0:
                    errors.append(f"{key}: {len(invalid)} TeamIDs outside valid ranges")

        if errors:
            for e in errors:
                print(f"  WARNING: {e}")
        else:
            print("  Validation passed: no issues found")

    def _print_summary(self) -> None:
        """Print summary statistics."""
        teams_m = self._tables.get("MTeams")
        teams_w = self._tables.get("WTeams")
        n_men = len(teams_m) if teams_m is not None else 0
        n_women = len(teams_w) if teams_w is not None else 0
        print(f"  Teams: {n_men} men's, {n_women} women's ({n_men + n_women} total)")

        seasons = set()
        for key in ["MRegularSeasonCompactResults", "WRegularSeasonCompactResults"]:
            if key in self._tables:
                seasons.update(self._tables[key]["Season"].unique())
        if seasons:
            print(f"  Seasons: {min(seasons)}-{max(seasons)}")

    @property
    def min_season(self) -> int:
        """Earliest season in the data."""
        seasons = set()
        for key in ["MRegularSeasonCompactResults", "WRegularSeasonCompactResults"]:
            if key in self._tables:
                seasons.update(self._tables[key]["Season"].unique())
        return min(seasons) if seasons else 0

    @property
    def max_season(self) -> int:
        """Latest season in the data."""
        seasons = set()
        for key in ["MRegularSeasonCompactResults", "WRegularSeasonCompactResults"]:
            if key in self._tables:
                seasons.update(self._tables[key]["Season"].unique())
        return max(seasons) if seasons else 0

    def _get_table(self, name: str) -> pd.DataFrame:
        """Get a table by name, raising if not found."""
        if name not in self._tables:
            raise KeyError(f"Table {name!r} not found. Available: {sorted(self._tables.keys())}")
        return self._tables[name].copy()

    def get_teams(self) -> pd.DataFrame:
        """Get combined men's and women's teams."""
        frames = []
        for prefix in ["M", "W"]:
            key = f"{prefix}Teams"
            if key in self._tables:
                df = self._tables[key].copy()
                df["Gender"] = prefix
                frames.append(df)
        if not frames:
            raise KeyError("No team files found")
        return pd.concat(frames, ignore_index=True)

    def _get_results(self, prefix: str, table_type: str, season: Optional[int] = None) -> pd.DataFrame:
        """Get results table with optional season filter."""
        key = f"{prefix}{table_type}"
        df = self._get_table(key)
        df["Gender"] = prefix
        if season is not None:
            df = df[df["Season"] == season]
        return df

    def get_regular_season_compact(self, season: Optional[int] = None, gender: Optional[str] = None) -> pd.DataFrame:
        """Get regular season compact results."""
        frames = []
        prefixes = [gender] if gender else ["M", "W"]
        for prefix in prefixes:
            key = f"{prefix}RegularSeasonCompactResults"
            if key in self._tables:
                df = self._tables[key].copy()
                df["Gender"] = prefix
                if season is not None:
                    df = df[df["Season"] == season]
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_regular_season_detailed(self, season: Optional[int] = None, gender: Optional[str] = None) -> pd.DataFrame:
        """Get regular season detailed results."""
        frames = []
        prefixes = [gender] if gender else ["M", "W"]
        for prefix in prefixes:
            key = f"{prefix}RegularSeasonDetailedResults"
            if key in self._tables:
                df = self._tables[key].copy()
                df["Gender"] = prefix
                if season is not None:
                    df = df[df["Season"] == season]
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_tourney_compact(self, season: Optional[int] = None, gender: Optional[str] = None) -> pd.DataFrame:
        """Get NCAA tournament compact results."""
        frames = []
        prefixes = [gender] if gender else ["M", "W"]
        for prefix in prefixes:
            key = f"{prefix}NCAATourneyCompactResults"
            if key in self._tables:
                df = self._tables[key].copy()
                df["Gender"] = prefix
                if season is not None:
                    df = df[df["Season"] == season]
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_tourney_detailed(self, season: Optional[int] = None, gender: Optional[str] = None) -> pd.DataFrame:
        """Get NCAA tournament detailed results."""
        frames = []
        prefixes = [gender] if gender else ["M", "W"]
        for prefix in prefixes:
            key = f"{prefix}NCAATourneyDetailedResults"
            if key in self._tables:
                df = self._tables[key].copy()
                df["Gender"] = prefix
                if season is not None:
                    df = df[df["Season"] == season]
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_seeds(self, season: Optional[int] = None, gender: Optional[str] = None) -> pd.DataFrame:
        """Get tournament seeds with parsed seed info."""
        frames = []
        prefixes = [gender] if gender else ["M", "W"]
        for prefix in prefixes:
            key = f"{prefix}NCAATourneySeeds"
            if key in self._tables:
                df = self._tables[key].copy()
                df["Gender"] = prefix
                if season is not None:
                    df = df[df["Season"] == season]
                # Parse seed strings
                parsed = df["Seed"].apply(parse_seed)
                df["Region"] = parsed.apply(lambda p: p.region)
                df["SeedNum"] = parsed.apply(lambda p: p.seed_num)
                df["PlayIn"] = parsed.apply(lambda p: p.play_in)
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_massey(self, season: Optional[int] = None) -> pd.DataFrame:
        """Get Massey ordinal rankings."""
        # Massey ordinals may be stored as MMasseyOrdinals or MasseyOrdinals
        for key in ["MMasseyOrdinals", "MasseyOrdinals"]:
            if key in self._tables:
                df = self._tables[key].copy()
                if season is not None:
                    df = df[df["Season"] == season]
                return df
        raise KeyError("Massey ordinals file not found")

    def get_submission_template(self, stage: int = 2) -> pd.DataFrame:
        """Get the sample submission template."""
        key = f"SampleSubmissionStage{stage}"
        # Try with and without M prefix
        for k in [key, f"M{key}"]:
            if k in self._tables:
                return self._tables[k].copy()
        raise KeyError(f"Sample submission stage {stage} not found")

    def get_tourney_slots(self, season: Optional[int] = None, gender: Optional[str] = None) -> pd.DataFrame:
        """Get tournament bracket slot structure."""
        frames = []
        prefixes = [gender] if gender else ["M", "W"]
        for prefix in prefixes:
            key = f"{prefix}NCAATourneySlots"
            if key in self._tables:
                df = self._tables[key].copy()
                df["Gender"] = prefix
                if season is not None:
                    df = df[df["Season"] == season]
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_available_tables(self) -> list[str]:
        """List all loaded table names."""
        return sorted(self._tables.keys())

"""Team name matching: map market ticker names to Kaggle TeamIDs."""

from __future__ import annotations

import re
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.config import CFG


class TeamMatcher:
    """Match team names from prediction markets to Kaggle TeamIDs.

    Uses MTeamSpellings.csv / WTeamSpellings.csv for exact matching,
    with fuzzy matching as fallback.
    """

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else CFG.DATA_DIR
        # Gender-keyed lookups: normalized_name -> TeamID
        self._mens_lookup: dict[str, int] = {}
        self._womens_lookup: dict[str, int] = {}
        # TeamID -> canonical name
        self._id_to_name: dict[int, str] = {}
        self._load()

    def _load(self) -> None:
        """Load team spellings and canonical names."""
        for prefix, lookup in [("M", self._mens_lookup), ("W", self._womens_lookup)]:
            spellings_path = self.data_dir / f"{prefix}TeamSpellings.csv"
            teams_path = self.data_dir / f"{prefix}Teams.csv"

            if spellings_path.exists():
                df = pd.read_csv(spellings_path, encoding="latin-1")
                for _, row in df.iterrows():
                    name = self._normalize(str(row.iloc[0]))
                    team_id = int(row.iloc[1])
                    if name:
                        lookup[name] = team_id

            if teams_path.exists():
                df = pd.read_csv(teams_path)
                for _, row in df.iterrows():
                    team_id = int(row["TeamID"])
                    team_name = str(row["TeamName"])
                    self._id_to_name[team_id] = team_name
                    norm = self._normalize(team_name)
                    if norm:
                        lookup[norm] = team_id

        total = len(self._mens_lookup) + len(self._womens_lookup)
        print(f"  TeamMatcher: {total} name variants "
              f"({len(self._mens_lookup)} M, {len(self._womens_lookup)} W), "
              f"{len(self._id_to_name)} teams")

    @staticmethod
    def _normalize(name: str) -> str:
        """Normalize a team name for matching."""
        name = name.lower().strip()
        # Replace separators with space
        name = re.sub(r"[-_./]", " ", name)
        # Collapse whitespace
        name = re.sub(r"\s+", " ", name).strip()
        return name

    def _get_lookup(self, gender: str) -> dict[str, int]:
        """Get the gender-specific name lookup."""
        return self._mens_lookup if gender == "M" else self._womens_lookup

    def match(self, name: str, gender: str = "M") -> Optional[int]:
        """Match a team name to a Kaggle TeamID.

        Tries in order:
            1. Exact match on normalized name
            2. Exact match with common suffix/prefix variations
            3. Fuzzy match (cutoff=0.8)

        Returns TeamID if matched, None otherwise.
        """
        if not name:
            return None

        lookup = self._get_lookup(gender)
        norm = self._normalize(name)

        # 1. Exact match
        team_id = lookup.get(norm)
        if team_id is not None:
            return team_id

        # 2. Try variations
        for variant in self._generate_variants(norm):
            team_id = lookup.get(variant)
            if team_id is not None:
                return team_id

        # 3. Fuzzy match
        candidates = list(lookup.keys())
        matches = get_close_matches(norm, candidates, n=1, cutoff=0.8)
        if matches:
            return lookup[matches[0]]

        return None

    def match_any_gender(self, name: str) -> Optional[int]:
        """Try matching against men's first, then women's."""
        result = self.match(name, "M")
        if result is None:
            result = self.match(name, "W")
        return result

    def get_name(self, team_id: int) -> str:
        """Get canonical team name for a TeamID."""
        return self._id_to_name.get(team_id, str(team_id))

    @staticmethod
    def _generate_variants(name: str) -> list[str]:
        """Generate common spelling variations of a team name."""
        variants = []
        # "st" <-> "state", "st." <-> "state"
        if " st " in f" {name} " or name.endswith(" st"):
            variants.append(re.sub(r"\bst\b", "state", name))
        if " state " in f" {name} " or name.endswith(" state"):
            variants.append(re.sub(r"\bstate\b", "st", name))
        # "n " <-> "north ", "s " <-> "south ", etc.
        for abbr, full in [("n ", "north "), ("s ", "south "), ("e ", "east "), ("w ", "west ")]:
            if name.startswith(abbr):
                variants.append(full + name[len(abbr):])
            if name.startswith(full):
                variants.append(abbr + name[len(full):])
        # Remove "university", "univ", etc.
        stripped = re.sub(r"\b(university|univ|college|coll)\b", "", name).strip()
        stripped = re.sub(r"\s+", " ", stripped)
        if stripped != name:
            variants.append(stripped)
        return variants

    def parse_kalshi_market(self, market: dict[str, Any]) -> dict[str, Any]:
        """Parse a Kalshi market to extract team info and market type.

        Returns:
            dict with market_type ("matchup"|"outright"), team_names,
            team_ids, and gender.
        """
        ticker = market.get("ticker", "")
        title = market.get("title", "")
        subtitle = market.get("subtitle", "")
        text = f"{title} {subtitle}".strip()

        result: dict[str, Any] = {
            "market_type": "outright",
            "team_names": [],
            "team_ids": [],
            "gender": "M",
        }

        # Detect gender
        if any(w in text.lower() for w in ["women", "wbb", "w basketball"]):
            result["gender"] = "W"

        # Try to detect matchup: "X vs Y", "X v Y", "X at Y Winner?"
        matchup_match = re.search(
            r"(.+?)\s+(?:vs?\.?|versus|against|over|at)\s+(.+?)(?:\s*[-–—:|]|\s+Winner\??|\s*$)",
            text, re.IGNORECASE,
        )
        if matchup_match:
            result["market_type"] = "matchup"
            team_a_raw = _clean_team_name(matchup_match.group(1))
            team_b_raw = _clean_team_name(matchup_match.group(2))
            result["team_names"] = [team_a_raw, team_b_raw]
            id_a = self.match(team_a_raw, result["gender"])
            id_b = self.match(team_b_raw, result["gender"])
            result["team_ids"] = [id_a or 0, id_b or 0]
        else:
            # Outright: "TeamX to win championship" / ticker-based
            result["market_type"] = "outright"
            team_name = self._extract_outright_team(text, ticker)
            if team_name:
                result["team_names"] = [team_name]
                team_id = self.match(team_name, result["gender"])
                result["team_ids"] = [team_id or 0]

        return result

    def _extract_outright_team(self, text: str, ticker: str) -> Optional[str]:
        """Extract team name from an outright/futures market title or ticker."""
        # "X to win ..."
        m = re.search(r"(.+?)\s+to\s+win", text, re.IGNORECASE)
        if m:
            name = _clean_team_name(m.group(1))
            if name:
                return name

        # "Will X win ..."
        m = re.search(r"will\s+(.+?)\s+win", text, re.IGNORECASE)
        if m:
            name = _clean_team_name(m.group(1))
            if name:
                return name

        # Fallback: parse ticker segments (e.g., MARCHMAD-26-DUKE-WIN)
        parts = ticker.upper().replace("-", " ").replace("_", " ").split()
        skip = {
            "NCAA", "NCAAM", "NCAAW", "MARCHMAD", "MM", "WIN", "WINNER",
            "CHAMP", "CHAMPIONSHIP", "R1", "R2", "R32", "R64", "S16", "E8",
            "F4", "FINAL", "FOUR", "ROUND", "OF", "YES", "NO",
        }
        candidates = [p for p in parts if p not in skip and not p.isdigit() and len(p) > 1]

        # Try matching each candidate
        for candidate in candidates:
            if self.match(candidate.lower()):
                return candidate.title()

        # Return first plausible candidate
        if candidates:
            return candidates[0].title()

        return None


def _clean_team_name(name: str) -> str:
    """Clean up an extracted team name string."""
    # Remove leading/trailing punctuation and whitespace
    name = re.sub(r"^[\s\-–—:]+|[\s\-–—:]+$", "", name)
    # Remove seed numbers like "(1)", "#1 ", etc.
    name = re.sub(r"^\s*[\(#]?\d{1,2}[\).]?\s*", "", name)
    # Remove round labels
    name = re.sub(
        r"\b(?:round of \d+|round \d+|r\d+|first round|second round)\b",
        "", name, flags=re.IGNORECASE,
    )
    return name.strip()

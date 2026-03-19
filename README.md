# March Madness 2026 Predictor

A prediction pipeline for the Kaggle March Machine Learning Mania 2026 competition. Combines Elo ratings, efficiency metrics, tournament-specific features, and live Kalshi prediction market data to generate win probabilities for every possible NCAA tournament matchup.

**Current Brier score: 0.1713** (leave-one-season-out CV, 2010-2025)

## Quick Start

```bash
# Install dependencies
uv sync  # or pip install -r requirements.txt

# Run with cached artifacts (fast iteration)
python main.py

# Recompute everything from scratch
python main.py --fresh

# Skip bracket generation for faster iteration
python main.py --skip-brackets
```

### Market Integration (Optional)

Set Kalshi API credentials in `.env`:

```
KALSHI_ACCESS_KEY=your-key-id
KALSHI_PRIVATE_KEY_PATH=./kalshi_private_key.pem
```

Without credentials, the pipeline runs on the base model alone.

---

## Pipeline Overview

```
Kaggle CSVs ──> Elo ──> Efficiency ──> Tournament Features ──> Backtest
                                                                  │
                                                           Feature Selection
                                                                  │
                                                         Train Base Model
                                                                  │
                                                        Base Predictions
                                                                  │
Kalshi API ──> Market Signals ──────────────────────────> Blend ──> Submission
                                                                  │
                                                              Brackets
```

Each stage saves artifacts to `output/`. On subsequent runs, cached artifacts are reused automatically. Delete an output file or pass `--fresh` to force recomputation.

| Stage | What | Time (fresh) | Cached? |
|-------|------|-------------|---------|
| 1. Load data | Parse 35 Kaggle CSVs | ~2s | No |
| 2. Elo ratings | Season-by-season Elo updates | ~10s | Yes |
| 3. Efficiency | Possession-based adjusted efficiency | ~8min | Yes |
| 4. Tournament features | 4 box-score-derived features | ~26s | Yes |
| 5. Backtest | Leave-one-season-out CV, 6 feature sets | ~25min | Yes |
| 6. Train base model | Logistic regression + isotonic calibration | ~18s | Yes |
| 7. Fetch market data | Kalshi API (2100+ markets) | ~2min | Yes |
| 8. Analyze order books | Team matching, signal extraction | <1s | Yes |
| 9. Blend predictions | Merge base + market signals | <1s | No |
| 10. Generate brackets | 10K Monte Carlo simulations | ~2min | No |
| 11. Generate submission | Format for Kaggle upload | <1s | No |

---

## Layer 1: Base Model

### Elo Ratings (`src/elo.py`)

Standard Elo with margin-of-victory adjustment, computed for every team across all seasons (1985-2026).

**Expected score:**
```
E(A) = 1 / (1 + 10^((R_B - R_A) / 400))
```

**MOV multiplier** (dampens blowouts against weak opponents):
```
MOV = log(|score_diff| + 1) * (2.2 / (elo_diff * 0.001 + 2.2))
```

**Rating update:**
```
delta = K * MOV * (1 - E(A))
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| K factor | 28 | Balances responsiveness vs stability |
| Initial rating | 1500 | Standard baseline |
| Season regression | 0.75 | `new = 1500 + 0.75 * (old - 1500)` — roster turnover |
| Home advantage | 65 | Applied to home team's Elo; 0 for tournament (neutral sites) |

Games are processed chronologically within each season. Pre-tournament Elo is snapshotted before tournament games are processed.

### Adjusted Efficiency (`src/efficiency.py`)

Tempo-free offensive and defensive efficiency per 100 possessions.

**Possession estimate:**
```
Poss = FGA - OR + TO + 0.475 * FTA
```

**Per-game efficiency** (average of both teams' possession estimates):
```
OffEff = (Score / AvgPoss) * 100
DefEff = (OppScore / AvgPoss) * 100
```

**Recency weighting:** Games in the last 30% of the season (by DayNum) receive 1.5x weight. Earlier games get 1.0x.

**Strength-of-schedule adjustment** using Massey ordinal rankings:
```
SOS_factor = 1.0 + (175 - avg_opponent_rank) / 175 * 0.1
AdjOff = AvgOff * SOS_factor
AdjDef = AvgDef / SOS_factor
NetEff = AdjOff - AdjDef
```

The median rank baseline of 175 assumes ~350 D1 teams per gender. A team facing tougher-than-average opponents gets its offense boosted and defense credit increased.

### Tournament Features (`src/features.py`)

Four features designed to capture tournament-specific dynamics not reflected in standard ratings:

| Feature | Formula | Why It Matters |
|---------|---------|----------------|
| **Opp3PtPct** | `Opp_3FGM / Opp_3FGA` | 3-point defense is amplified in tournament settings where opponents shoot more threes |
| **CloseGameTOMargin** | Avg `(Opp_TO - Team_TO)` in games decided by <=5 pts | Turnover discipline in close games predicts tournament clutch performance |
| **OffRebRate** | `Team_OR / (Team_OR + Opp_DR)` | Second-chance points extend possessions in low-scoring tournament games |
| **LateFTPct** | FT% in last 25% of season | Late-season free throw shooting correlates with tournament pressure performance |

If fewer than 5 close games exist for a team, CloseGameTOMargin falls back to overall turnover margin.

### Model Architecture (`src/base_model.py`)

**Logistic regression** on feature differentials (TeamA - TeamB, where TeamA has the lower TeamID per Kaggle convention):

```python
LogisticRegression(max_iter=1000) + CalibratedClassifierCV(method="isotonic", cv=5)
```

Isotonic calibration transforms the logistic output into well-calibrated probabilities using 5-fold cross-validation.

**Feature differentials used** (selected by backtest):
- `elo_diff` = EloPreTourney_A - EloPreTourney_B
- `seed_diff` = Seed_A - Seed_B
- `eff_diff` = NetEff_A - NetEff_B
- `closegametomargin_diff` = CloseGameTOMargin_A - CloseGameTOMargin_B

Note: `opp3ptpct_diff` is **negated** (lower opponent 3pt% = better defense for that team).

All features are standardized with `StandardScaler` before model fitting. Predictions are clipped to [0.03, 0.97].

### Backtesting (`src/backtest.py`)

**Leave-one-season-out cross-validation** across 2010-2025 (skip 2020 — no tournament).

For each of 6 feature sets, trains on 14 seasons and evaluates Brier score on the held-out tournament:

```
Brier = mean((prediction - actual)^2)
```

Results from the current model:

| Feature Set | Brier | Notes |
|-------------|-------|-------|
| elo + seed | 0.1719 | Baseline |
| + efficiency | 0.1716 | Small gain |
| + 3pt defense | 0.1716 | No gain |
| **+ close-game TO** | **0.1713** | **Best** |
| + off rebounding | 0.1715 | Overfitting |
| + late FT% | 0.1718 | Overfitting |

The pipeline automatically selects the feature set with the lowest average Brier score.

---

## Layer 2: Market Integration

### Why Markets

Prediction markets aggregate information our model can't capture: injuries, coaching adjustments, eye-test evaluations, matchup-specific dynamics. Market prices are calibrated by participants with money at stake.

Markets are most valuable for:
- Close matchups (5v12, 7v10) where the model is uncertain
- Games where late-breaking news (injuries, suspensions) shifts odds
- High-liquidity markets with deep order books

### Data Collection (`src/market_scraper.py`)

Fetches from Kalshi using RSA-PSS signed authentication. Series monitored:

| Series | Description | Market Type |
|--------|-------------|-------------|
| `KXMARMAD` | Men's championship winner | Outright/Futures |
| `KXWMARMAD` | Women's championship winner | Outright/Futures |
| `KXNCAAMBGAME` | Individual game winners | Matchup |
| `KXMARMADSEED` | Seed of champion | Supplementary |
| `KXMARMADCONFWIN` | Conference of champion | Supplementary |

Only **active markets** receive orderbook/trade data fetches (saves ~90% of API calls).

### Team Name Resolution (`src/team_matcher.py`)

Maps market ticker names to Kaggle TeamIDs using:

1. **Exact match** against `MTeamSpellings.csv` / `WTeamSpellings.csv` (1,559 name variants)
2. **Spelling variants** — st/state, north/n, etc.
3. **Fuzzy match** — `difflib.get_close_matches` with 0.8 cutoff

Gender-specific lookups prevent men's/women's ID collisions (same team name maps to different ID ranges: M=1000-1999, W=3000-3999).

Kalshi matchup games have **separate tickers per team side** (e.g., `KXNCAAMBGAME-...-DUKE` and `KXNCAAMBGAME-...-ARIZ`). These are paired by event ticker to produce a single matchup signal.

### Signal Extraction (`src/market_features.py`)

**Mid-price** (primary probability signal):
```
mid_price = (yes_bid_dollars + yes_ask_dollars) / 2
```
Fallback chain: best bid/ask -> last trade price -> legacy orderbook.

**Whale detection:**
```
whale_threshold = 5.0 * median(trade_sizes)
whale_signal = (whale_yes_vol - whale_no_vol) / total_whale_vol  # in [-1, +1]
```

**Market confidence** (sigmoid-scaled depth + volume):
```
depth_conf = 1 - 1 / (1 + depth / 1000)
volume_conf = 1 - 1 / (1 + volume / 10000)
confidence = 0.5 * depth_conf + 0.5 * volume_conf
```

**Futures-to-pairwise conversion** (for championship odds):
```
log_odds = log(p / (1-p))                    # per team
diff = log_odds_A - log_odds_B               # pairwise
P(A beats B) = 1 / (1 + exp(-0.3 * diff))    # logistic with 0.3 scale factor
```

The 0.3 scale factor compresses championship log-odds (which span a wide range) into game-level probabilities. A difference of 2.0 in log-odds maps to ~65% win probability. Futures-derived confidence is discounted by 0.6x versus direct matchup markets.

### Blending Algorithm (`src/blend.py`)

**Priority:** Direct matchup market > Futures-derived > Base model only.

**Weight determination:**

| Scenario | w_base | w_market | Rationale |
|----------|--------|----------|-----------|
| High-confidence matchup (conf >= 0.5) | 0.55-0.60 | 0.40-0.45 | Trust deep-liquidity markets |
| Low-confidence matchup (conf < 0.5) | 0.85 | 0.15 | Thin market, mostly trust model |
| Futures-derived (any confidence) | 0.85 | 0.15 | Indirect signal, conservative |
| Whale activity (abs(signal) > 0.5) | -0.05 boost to market | | Smart money nudge |

High-confidence matchup weight adjusts with confidence:
```
w_base = 0.6 - 0.1 * (confidence - 0.5)
```

All weights clipped to [0.1, 0.95]. Final blended prediction clipped to [0.03, 0.97].

```
blended = w_base * base_pred + w_market * market_prob
```

---

## Bracket Simulation (`src/bracket.py`)

Monte Carlo simulation with 10,000 iterations per gender:

1. Build region brackets from tournament seeds
2. For each simulation, advance teams probabilistically using blended predictions
3. Simulation 0 is the "chalk bracket" — always picks the higher-probability team
4. Track advancement counts per team per round

**Output:** Per-team probability of reaching each round (R64 through Champion), displayed as an ASCII bracket with top contenders.

---

## Output Files

| File | Description |
|------|-------------|
| `submission.csv` | Kaggle submission (132K matchups) |
| `base_predictions.csv` | Layer 1 probabilities |
| `market_signals.csv` | Direct matchup market signals |
| `futures_signals.csv` | Championship futures with log-odds |
| `blended_predictions.csv` | Final blended probabilities |
| `team_ratings.csv` | Elo + efficiency + features per team-season |
| `backtest_results.json` | CV results and feature selection |
| `bracket_men.txt` | Men's tournament bracket |
| `bracket_women.txt` | Women's tournament bracket |
| `team_advancement_probs.csv` | Per-team round-by-round advancement odds |

---

## Key Design Decisions

1. **Isotonic calibration over Platt scaling** — isotonic is more flexible for the nonlinear probability surface of tournament games
2. **Feature differentials, not raw values** — the model learns "how much better is A than B" directly, which is what matters for pairwise prediction
3. **Conservative market blending (60/40 default)** — base model Brier of 0.1713 is already competitive; aggressive market weighting risks blowing it up on thin markets
4. **Futures as log-odds, not direct probabilities** — championship odds of 20% vs 4% don't directly tell you P(A beats B), but the log-odds differential is an informative strength signal
5. **0.03-0.97 clipping** — Kaggle's log-loss scoring heavily penalizes extreme predictions that are wrong; clipping limits downside risk

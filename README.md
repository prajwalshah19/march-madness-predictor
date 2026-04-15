# March Madness 2026 Predictor

A prediction pipeline for the Kaggle March Machine Learning Mania 2026 competition. Combines Elo ratings, efficiency metrics, tournament-specific features, and live Kalshi prediction market data to generate win probabilities for every possible NCAA tournament matchup.

**Current Brier score: 0.1711** (leave-one-season-out CV, 2010-2025)

## Quick Start

```bash
# Install dependencies
uv sync  # or: pip install .

# Download Kaggle data (requires kaggle CLI)
pip install kaggle  # or: uv sync --extra cli
kaggle competitions download -c march-machine-learning-mania-2026
unzip march-machine-learning-mania-2026.zip -d data/kaggle/

# Run with cached artifacts (fast iteration)
python main.py

# Recompute everything from scratch
python main.py --fresh

# Skip bracket generation for faster iteration
python main.py --skip-brackets
```

### Market Integration (Optional)

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

```
KALSHI_ACCESS_KEY=your-key-id
KALSHI_PRIVATE_KEY_PATH=./kalshi_private_key.pem
```

Without credentials, the pipeline runs on the base model alone.

---

## Pipeline Overview

```
Kaggle CSVs ──> Elo ──> Efficiency ──> Tournament Features ──> Massey Log-Odds ──> Backtest
                                                                                       │
                                                                                Feature Selection
                                                                                       │
                                                                              Train Base Model
                                                                                       │
                                                                             Base Predictions
                                                                                       │
Kalshi API ──> Market Signals ──> Calibrate Scale Factor ──────────────────> Blend ──> Submission
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
| 4b. Massey log-odds | Composite ranking → log-odds proxy | ~5s | Yes |
| 5. Backtest | Leave-one-season-out CV, 7 feature sets | ~30min | Yes |
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

### Massey Composite Log-Odds (`src/features.py`)

A market-consensus strength proxy derived from Massey ordinal rankings, which aggregate 100+ independent ranking systems (computer polls, sagarin, KenPom-style ratings, etc.). This serves as a historical proxy for prediction market signal since actual market data (Kalshi, Vegas lines) doesn't exist for 2010-2024 backtesting.

**Computation:**
1. For each team-season, take rankings from the latest available day (near tournament time)
2. Average the ordinal rank across all ranking systems to get a composite rank
3. Convert to log-odds space for the logistic regression:

```
strength = (max_rank - composite_rank + 1) / max_rank    # [0, 1]
massey_logodds = log(strength / (1 - strength))           # (-inf, +inf)
```

The feature `market_logodds_diff = massey_logodds_A - massey_logodds_B` captures "crowd wisdom" signal beyond what our Elo or efficiency metrics provide. The backtest confirmed this feature improves Brier from 0.1713 to 0.1711.

For current-year (2026) predictions, the base model uses Massey ordinals in its features while the blending layer overlays actual Kalshi market data — both capture market consensus but through different mechanisms.

**Men's only:** Massey ordinals (`MMasseyOrdinals.csv`) are available only for men's teams. Women's teams get a neutral value of 0.

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
- `offrebrate_diff` = OffRebRate_A - OffRebRate_B
- `market_logodds_diff` = MasseyLogOdds_A - MasseyLogOdds_B

Note: `opp3ptpct_diff` is **negated** (lower opponent 3pt% = better defense for that team).

All features are standardized with `StandardScaler` before model fitting. Predictions are clipped to [0.03, 0.97].

### Backtesting (`src/backtest.py`)

**Leave-one-season-out cross-validation** across 2010-2025 (skip 2020 — no tournament).

For each of 7 feature sets, trains on 14 seasons and evaluates Brier score on the held-out tournament:

```
Brier = mean((prediction - actual)^2)
```

Results from the current model:

| Feature Set | Brier | Notes |
|-------------|-------|-------|
| elo + seed | 0.1719 | Baseline |
| + efficiency | 0.1716 | Small gain |
| + 3pt defense | 0.1716 | No gain |
| + close-game TO | 0.1713 | Previous best |
| + off rebounding | 0.1715 | Slight regression |
| + late FT% | 0.1718 | Overfitting |
| **+ market log-odds** | **0.1711** | **Best — Massey composite proxy** |

The pipeline automatically selects the feature set with the lowest average Brier score. The winning set adds `market_logodds_diff` (Massey composite) alongside `elo_diff`, `seed_diff`, `eff_diff`, and `offrebrate_diff`.

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
log_odds = log(p / (1-p))                      # per team
diff = log_odds_A - log_odds_B                 # pairwise
P(A beats B) = 1 / (1 + exp(-scale * diff))    # logistic with calibrated scale factor
```

**Scale factor calibration** (`calibrate_futures_scale` in `src/market_features.py`): The scale factor that converts championship log-odds differentials into game-level probabilities is now **calibrated empirically** rather than hardcoded. For games that have both a direct matchup market AND both teams in futures, the function grid-searches over scale factors [0.2, 0.3, 0.4, 0.5, 0.6, 0.8] to minimize RMSE between futures-derived probabilities and actual matchup market prices. Current calibration finds **0.8** (up from the original hardcoded 0.3), meaning the previous factor was over-compressing — a 1-seed vs 12-seed log-odds differential of 3.0 should map to ~92% (at 0.8) rather than 71% (at 0.3). Futures-derived confidence is discounted by 0.6x versus direct matchup markets.

### Blending Algorithm (`src/blend.py`)

**Priority:** Direct matchup market > Futures-derived > Base model only.

The blending layer applies five adaptive strategies to determine how much to trust market signals vs the base model:

**1. Base weight determination:**

| Scenario | w_base | w_market | Rationale |
|----------|--------|----------|-----------|
| High-confidence matchup (conf >= 0.5) | 0.55-0.60 | 0.40-0.45 | Trust deep-liquidity markets |
| **Very high-confidence matchup (conf > 0.9)** | **0.35** | **0.65** | **Liquid, well-priced — trust aggressively (Improvement 5)** |
| Low-confidence matchup (conf < 0.5) | 0.85 | 0.15 | Thin market, mostly trust model |
| Futures-derived (confidence-adaptive) | 0.65-0.85 | 0.15-0.35 | Scales with confidence (Improvement 3) |
| Whale activity (abs(signal) > 0.5) | -0.05 boost to market | | Smart money nudge |

**2. Confidence-adaptive futures weighting (Improvement 3):**

Instead of a flat 85/15 split for all futures-derived predictions, the market weight now scales with confidence:
```
w_market = 0.15 + 0.20 * confidence    # Range: [0.15, 0.35]
w_base = 1.0 - w_market
```
High-confidence futures (Duke/Arizona with millions in volume, confidence ~0.60) get up to 0.27 market weight. Low-confidence mid-majors (confidence ~0.05) stay near 0.16.

**3. Disagreement-based weight shifting (Improvement 4):**

When the model and market disagree by more than 10%, the market weight is boosted:
```
disagreement = abs(base_pred - market_prob)
if disagreement > 0.10:
    w_base -= 0.10 * (disagreement - 0.10)    # Gradual boost
```
Intuition: if the model says 60% and the market says 45%, someone knows something (injury, travel issue, matchup dynamic). The additional market weight is proportional to the magnitude of disagreement beyond the 10% threshold.

**4. Very high-confidence matchup override (Improvement 5):**

For the ~40 direct matchup markets with confidence > 0.9, the base weight drops to 0.35 (market gets 0.65). These are the exact games scored by Kaggle, with liquid order books and well-calibrated prices. This is the single highest-leverage change — trusting the market more on scored games.

**5. Final blending:**
```
blended = w_base * base_pred + w_market * market_prob
```
All weights clipped to [0.1, 0.95]. Final blended prediction clipped to [0.03, 0.97].

---

## Bracket Simulation (`src/bracket.py`)

Monte Carlo simulation with 10,000 iterations per gender:

1. Build region brackets from tournament seeds
2. For each simulation, advance teams probabilistically using blended predictions
3. Simulation 0 is the "chalk bracket" — always picks the higher-probability team
4. Track advancement counts per team per round

**Output:** Per-team probability of reaching each round (R64 through Champion), displayed as an ASCII bracket with top contenders.

---

## CLI: `mm-predict` (`mm_predict.py`)

Interactive bracket-building tool that looks up any matchup or team path from the cached predictions. Install with `uv pip install -e .` for the `mm-predict` command, or run directly with `python mm_predict.py`.

### Matchup Prediction

```bash
mm-predict "Duke" "Arizona"           # Head-to-head prediction
mm-predict "Duke" "Arizona" -v        # With detailed team profiles
mm-predict "UConn" "South Carolina" -g W   # Women's tournament
```

Example output:

```
  ========================================================
  (1) Duke  vs  (1) Arizona
  ========================================================

  Winner:      Duke
  Win Prob:    52.3%
  Confidence:  Toss-up
  Variance:    0.2495

  ── Probability Breakdown ──
  Duke                       52.3%
  Arizona                    47.7%

  ── Model vs Market ──
                               Model   Market  Blended
  Duke                        52.2%   52.7%   52.3%
  Arizona                     47.8%   47.3%   47.7%
  Market source: futures (confidence: 0.60)

  ── Championship Futures ──
  Duke                       19.5% to win title  (vol: $6,204,060)
  Arizona                    17.5% to win title  (vol: $5,656,110)
```

With `-v`, also shows Elo, net efficiency, adjusted offense/defense, tempo, 3pt defense, offensive rebounding rate, FT%, and Massey log-odds for both teams.

### Bracket Path

Shows a team's full path through the actual bracket with round-by-round advancement probabilities computed via forward propagation through the bracket tree.

```bash
mm-predict path "Duke"          # Full bracket path
mm-predict path "UConn" -g W    # Women's bracket path
```

Example output:

```
  ==================================================================
  (1) Duke — Region W Bracket Path
  ==================================================================

  Round of 64    vs (16) Siena
                 Win: 97.0%
                 Advance prob: 97.0%

  Round of 32    Likely opponents:
                  ( 8) Ohio St              56% chance  |  beat: 92%
                  ( 9) TCU                  44% chance  |  beat: 92%
                 Weighted win: 91.7%
                 Advance prob: 88.9%

  Sweet 16       Likely opponents:
                  ( 5) St John's            52% chance  |  beat: 78%
                  ( 4) Kansas               40% chance  |  beat: 84%
                  (12) Northern Iowa         6% chance  |  beat: 95%
                  (13) Cal Baptist           3% chance  |  beat: 96%
                 Weighted win: 82.1%
                 Advance prob: 73.0%

  Elite 8        Likely opponents:
                  ( 2) Connecticut          44% chance  |  beat: 76%
                  ( 3) Michigan St          29% chance  |  beat: 79%
                  ( 6) Louisville           11% chance  |  beat: 85%
                  ( 7) UCLA                  9% chance  |  beat: 87%
                  ... and 2 others
                 Weighted win: 80.2%
                 Advance prob: 58.6%

  Final Four     vs Region X champion:
                  ( 1) Florida              43% chance  |  beat: 64%
                  ( 2) Houston              23% chance  |  beat: 70%
                  ( 3) Illinois             17% chance  |  beat: 75%
                  ( 5) Vanderbilt            6% chance  |  beat: 82%
                 Weighted win: 70.2%
                 Reach F4: 58.6% | Win F4: 41.2%

  Championship   vs Region Y/Z champion
                  ( 1) Michigan             53% chance  |  beat: 56%
                  ( 1) Arizona              50% chance  |  beat: 52%
                  ( 2) Purdue               21% chance  |  beat: 71%
                  ( 2) Iowa St              18% chance  |  beat: 75%
                 Weighted win: 65.1%

  ── Summary ──
  Model championship prob:  26.8%
  Market championship prob: 19.5%  (vol: $6,204,060)
```

For each round, the path shows:
- **R64:** Actual opponent and win probability
- **R32-E8:** Likely opponents weighted by their probability of advancing, plus a weighted win probability across all possible opponents
- **F4/Championship:** Top contenders from opposing regions with head-to-head win probabilities
- **Summary:** Model vs market championship probability

### Upset Finder

Scans the actual bracket for upset candidates at any round, ranked by probability. For rounds beyond R64, assumes chalk (higher seed wins) in all prior rounds to determine matchups.

```bash
mm-predict upsets              # R64 upsets (default)
mm-predict upsets R32          # Round of 32 upsets
mm-predict upsets S16          # Sweet 16 upsets
mm-predict upsets E8           # Elite 8 upsets
mm-predict upsets F4           # Final Four matchups
mm-predict upsets C            # Championship
mm-predict upsets 12           # R64, 12-seeds only
mm-predict upsets R32 8        # R32, 8-seeds only
mm-predict upsets -g W         # Women's tournament
```

Example — R64 upsets:

```
  ==========================================================================
  Men's Round of 64 — Upset Candidates (Chalk Path)
  ==========================================================================

  Matchup                                       Upset% Region
  ------------------------------------------------------------
  ( 9) Utah St            > ( 8) Villanova         55.6%      Z ***
  ( 9) Iowa               > ( 8) Clemson           51.4%      X ***
  ( 9) TCU                > ( 8) Ohio St           44.3%      W  **
  (11) VCU                > ( 6) North Carolina    43.1%      X  **
  (10) Santa Clara        > ( 7) Kentucky          43.0%      Y  **
  (10) Missouri           > ( 7) Miami FL          40.9%      Z  **
  (11) Texas              > ( 6) BYU               39.8%      Z  **

  *** = model favors upset  ** = strong candidate  * = worth considering
  Matchups assume chalk (higher seed wins) in all prior rounds
```

Example — Sweet 16 upsets (chalk path):

```
  ==========================================================================
  Men's Sweet 16 — Upset Candidates (Chalk Path)
  ==========================================================================

  Matchup                                       Upset% Region
  ------------------------------------------------------------
  ( 3) Illinois           > ( 2) Houston           43.8%      X  **
  ( 3) Virginia           > ( 2) Iowa St           43.3%      Y  **
  ( 3) Michigan St        > ( 2) Connecticut       42.9%      W  **
  ( 3) Gonzaga            > ( 2) Purdue            42.5%      Z  **
  ( 5) Vanderbilt         > ( 1) Florida           25.9%      X   *
  ( 4) Arkansas           > ( 1) Arizona           24.6%      Z
  ( 4) Alabama            > ( 1) Michigan          22.8%      Y
  ( 5) St John's          > ( 1) Duke              21.9%      W

  *** = model favors upset  ** = strong candidate  * = worth considering
  Matchups assume chalk (higher seed wins) in all prior rounds
```

Valid rounds: `R64`, `R32`, `S16`, `E8`, `F4`, `C`.

### Full Bracket

Generates a visual ASCII bracket showing the model's chalk picks (always picks higher-probability team) at every round, with win probabilities and advancement odds.

```bash
mm-predict bracket             # Men's bracket
mm-predict bracket -g W        # Women's bracket
```

Example output (abbreviated):

```
  ==============================================================================
  Men's NCAA Tournament Bracket — 2026 (Model Picks)
  ==============================================================================

  ┌───────────────────────────────────────────────────────────────────────┐
  │ Region W                                                             │
  │ R64                   R32                   Sweet 16       Elite 8   │
  │──────────────────────────────────────────────────────────────────────│
  │ ( 1) Duke        97%  ( 1) Duke        92%  ( 1) Duke  78%  Duke 76%│
  │ ( 8) Ohio St     56%                                                 │
  │ ( 4) Kansas      87%  ( 5) St John's   59%                          │
  │ ( 5) St John's   83%                                                 │
  │ ( 6) Louisville   64%  ( 3) Michigan St 59%  ( 2) UConn 57%         │
  │ ( 3) Michigan St 92%                                                 │
  │ ( 7) UCLA        69%  ( 2) Connecticut 70%                          │
  │ ( 2) Connecticut 96%                                                 │
  └───────────────────────────────────────────────────────────────────────┘
  ... (Regions X, Y, Z)

  ┌───────────────────────────────────────────────────────────────────────┐
  │ Final Four                                                           │
  │  ( 1) Duke       [W]  vs  ( 1) Florida    [X]                       │
  │  Winner: ( 1) Duke       (64%)                                       │
  │                                                                       │
  │  ( 1) Michigan   [Y]  vs  ( 1) Arizona    [Z]                       │
  │  Winner: ( 1) Arizona    (53%)                                       │
  │──────────────────────────────────────────────────────────────────────│
  │  Championship: ( 1) Duke  vs  ( 1) Arizona                          │
  │  Champion: ( 1) Duke (52%)  (Kalshi: 20%)                           │
  └───────────────────────────────────────────────────────────────────────┘

  ┌───────────────────────────────────────────────────────────────────────┐
  │ Advancement Probabilities (Top 16)                                   │
  │  Team                  Rgn   R32   S16    E8    F4                   │
  │──────────────────────────────────────────────────────────────────────│
  │  ( 1) Duke               W   97%   89%   73%   59%                  │
  │  ( 1) Michigan           Y   97%   87%   70%   53%                  │
  │  ( 1) Arizona            Z   97%   86%   68%   50%                  │
  │  ( 1) Florida            X   97%   83%   65%   43%                  │
  │  ...                                                                 │
  └───────────────────────────────────────────────────────────────────────┘
```

### First Four

First Four play-in results are hardcoded in `mm_predict.py` (`FIRST_FOUR_WINNERS` / `FIRST_FOUR_LOSERS`). Losers are excluded from all bracket outputs. Current results:

| Game | Winner | Loser |
|------|--------|-------|
| Y11 | Miami OH | SMU |
| Z11 | Texas | NC State |
| X16 | Prairie View | Lehigh |
| Y16 | Howard | UMBC |

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
3. **Market signal in two layers** — Massey composite enters the logistic regression as a learnable feature (the model learns the optimal weight via backtest), while live Kalshi data enters in the blending layer. This avoids the chicken-and-egg problem of needing market data at training time.
4. **Calibrated scale factor over hardcoded** — the futures-to-matchup conversion scale factor (0.8) is calibrated against direct matchup markets rather than guessed. The original 0.3 was over-compressing, making 1-seed vs 12-seed look like 71% when it should be ~92%.
5. **Adaptive blending, not fixed weights** — market weight varies by source (matchup vs futures), confidence, and model-market disagreement. High-confidence matchup markets on scored games get 65% market weight; thin futures markets get 15%.
6. **Futures as log-odds, not direct probabilities** — championship odds of 20% vs 4% don't directly tell you P(A beats B), but the log-odds differential is an informative strength signal
7. **0.03-0.97 clipping** — Kaggle's log-loss scoring heavily penalizes extreme predictions that are wrong; clipping limits downside risk

# Hard to Trade? Predicting Intraday Execution Difficulty by (Ticker, Time-of-Week) Bucket

DS 402 · Group 3 (Andrew Eross, Khang Le). Adapts Google Research's
*Hard to Park? Estimating Parking Difficulty at Scale* (KDD 2019) to markets: classify each
`(ticker, weekday, hour-slot)` bucket as **easy / medium / hard** to trade, using effective spread as ground truth.

## Quick start
```bash
pip install -r requirements.txt
export ALPACA_API_KEY=...        # Windows: setx / $env:ALPACA_API_KEY
export ALPACA_SECRET_KEY=...
jupyter lab
```
Run the notebooks in order. Notebook 01 is resumable (every ticker-day is cached in `data/raw/`).

| Notebook | Purpose | Research question |
|---|---|---|
| `01_data_collection` | Alpaca trades+quotes -> per-bucket features and effective spread | data sufficiency |
| `02_labels_and_features` | per-stock tercile labels, lagged features, leakage checks | - |
| `03_models` | baselines, logistic regression, softmax and MLP trained on the reward loss | RQ1, RQ3 |
| `04_evaluation_ablation_generalization` | ablation, diagnostics, tech->financials(->crypto) transfer | RQ2, RQ4 |

Reusable code lives in `src/htt/` (`config`, `data`, `features`, `modeling`, `evaluation`); change the
universe, dates, feed and label in `config.py`.

## Method in one paragraph
Bucket = (ticker, weekday, slot), one observation per week. Label = per-ticker tercile of size-weighted
effective spread (thresholds from the training weeks only). Features for week *w* come only from weeks < *w*
(last week and trailing 4-week mean of liquidity, price-deviation-vs-sector-ETF and oscillation measures, plus
relative busyness and calendar terms), z-scored per ticker on training data. Chronological 60/20/20 split by
week. Models are scored with the proposal's asymmetric reward matrix and a Balanced Normalized Reward (BNR).

## Known caveats (put these in the report)
* **Feed:** the free plan is IEX-only, so spreads are relative to the IEX book, not the NBBO.
* **Effective spread is unsigned** (`2|P-M|/M`), so no trade-direction inference is needed.
* **Holidays/missing buckets:** "last week" means the previous *observed* week for that bucket.
* **Class drift:** terciles come from training weeks, so val/test class shares need not be 1/3.
* The proposal mentions both 1-hour buckets and 30-minute aggregation; the code uses 1-hour buckets
  (six 1-hour + one 30-minute slot).

## Layout
```
notebooks/   4 notebooks
src/htt/     reusable modules
data/        raw cache (git-ignored), processed tables
results/     metrics CSVs and predictions
```

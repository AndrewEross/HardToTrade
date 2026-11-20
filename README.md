# Hard to Trade? DS 402 Group 3

Research pipeline following the submitted project proposal. Python 3.10+ recommended.

## Install

```bash
python -m pip install -r requirements.txt
export APCA_API_KEY_ID='your_key'
export APCA_API_SECRET_KEY='your_secret'
```

## Run

```bash
# Smoke test with 2 symbols (first symbols in list)
python hard_to_trade.py fetch --start 2026-09-01 --end 2026-09-15 --max-symbols 2
# Full data (historical 6-month period; adjust dates and entitlements)
python hard_to_trade.py fetch --start 2026-04-01 --end 2026-10-01 --feed iex
python hard_to_trade.py prepare
python hard_to_trade.py train
```

**Important:** the smoke test is for API testing only; it does not fetch ETFs and will not yield enough weeks to train. Use the full fetch to train. IEX quotes/trades represent only the IEX exchange, not the consolidated US market. SIP access may require a paid subscription. No historical data or model metrics are bundled or fabricated.

## What the script does

- Collects minute bars and tick trades/quotes; saves day-by-day parquet caches to avoid repeated downloads.
- Aligns each trade to the most recent quote **at or before** the trade, within five seconds.
- Computes the proposal's effective spread: `2*abs(log(trade_price)-log(midpoint))*10000` basis points.
- Constructs seven regular-hours ET buckets: six 60-minute and one 30-minute.
- Aggregates intraday observations to weekly `(ticker, hour-slot)` buckets. Features for week `t-1` predict effective-spread difficulty in week `t`.
- Labels Easy/Medium/Hard by **ticker-specific training-set tertiles**, using a global-training fallback for degenerate distributions.
- Chronological 60/20/20 split by distinct weeks, not shuffled rows.
- Compares ticker/hour modal baseline, L2 multinomial logistic regression, and two-layer (16,8) ReLU MLP.
- Calculates accuracy, macro F1, and the proposal's asymmetric reward matrix; saves confusion plots, ablations and cross-sector test.

## Design choices and caveats

1. **Week vs day:** the proposal discusses predicting next week's difficulty, so one sample here is a `(ticker, week, time-of-day slot)` aggregate over the weekdays in that week. This makes week-ahead prediction explicit; it does not yet model individual `(ticker, date, hour)` outcomes. A future refinement could add day-of-week to the bucket.
2. **Labels:** thresholds are fit only on training rows. Do not compute quantiles on the whole dataset before splitting.
3. **Model differences:** sklearn `MLPClassifier` supports the specified (16,8) ReLU layers and early stopping but **not dropout**. To exactly reproduce the proposal's dropout requirement, replace the sklearn MLP with a PyTorch implementation.
4. **Execution cost limitations:** effective spread from trades matched to quotes is a proxy for transaction cost, not a forecast of the realized cost of a particular order. Tick-data quality, trade corrections, quote staleness, market-feed scope, and timestamp ordering can materially change results.
5. **SIP vs IEX:** the script defaults to IEX for accessibility; this is a limited-market sample. A complete-market study should use SIP entitlements or licensed TAQ data. Avoid interpreting IEX-only measurements as consolidated-market spread estimates.
6. **Missing data:** stocks without trades and quotes cannot receive effective-spread labels; they drop out. The alternative Amihud or high-low fallback described in the proposal is **not** silently substituted.
7. **Cryptocurrency:** the crypto generalization experiment is **not implemented** here because it requires separate data acquisition and a 24-hour bucketing definition.
8. **Sample selection:** static stock list can cause survivorship bias; document this in the report.
9. **Validation:** validation is reported separately, but model hyperparameters are fixed rather than tuned. The test set is not used for tuning.

## Outputs

- `data/weekly_modeling.parquet`: processed weekly features, effective-spread target, lagged predictors
- `results/metrics.csv`: baseline and model scores, ablations, sector transfer
- `results/train_thresholds.csv`: per-ticker training thresholds
- `results/*_confusion.png`: confusion matrices
- `results/intraday_spread.png`: median effective spread by ET bucket
- `results/*.joblib`: fitted model pipelines

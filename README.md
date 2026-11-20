# Hard to Trade? — DS 402 Group 3

This project uses machine learning to predict how difficult it will be to trade a stock during different hours of the trading day.

We use historical stock prices, trades, and quotes from Alpaca to calculate trading difficulty. We then compare a simple baseline, logistic regression, and a neural network.

The main goal is to predict whether trading difficulty during the following week will be **Easy, Medium, or Hard**.

## 1. Requirements

- Python 3.10 or newer
- Alpaca API account and API keys
- Internet connection for downloading data
- Enough disk space for historical stock data

Install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

Set your Alpaca API keys in Terminal (macOS/Linux):

```bash
export APCA_API_KEY_ID='your_api_key'
export APCA_API_SECRET_KEY='your_secret_key'
```

Replace the placeholders with your own keys. Never upload API keys to GitHub.

**Note:** You only need Alpaca API keys when downloading data. If the processed dataset already exists, you can run training without downloading again.

## 2. How to Run the Project

Open Terminal and navigate to the project folder:

```bash
cd ~/Downloads/hard_to_trade_project
```

If your project is stored somewhere else, change the path accordingly.

### Step 1 — Download stock data

First, run a small test to confirm that the Alpaca API works:

```bash
python hard_to_trade.py fetch \
  --start 2026-09-01 \
  --end 2026-09-15 \
  --max-symbols 2
```

This downloads only two stocks and is intended for testing the API connection.

For the full six-month dataset, run:

```bash
python hard_to_trade.py fetch \
  --start 2026-04-01 \
  --end 2026-10-03 \
  --feed iex \
  --workers 4
```

The full download may take several hours depending on the internet connection and Alpaca rate limits.

The downloader saves data as daily Parquet files. Previously downloaded data can be reused, so there is normally no need to start from scratch.

**Important:** The small API test does not download enough data to train the models. Use the full dataset for the actual experiment.

### Step 2 — Prepare the dataset

After downloading, run:

```bash
python hard_to_trade.py prepare
```

This step:

- Reads historical bars, trades, and quotes.
- Matches trades with the most recent valid quotes.
- Calculates effective spread as a measure of trading difficulty.
- Aggregates the data into hourly and weekly observations.
- Creates lagged features for next-week prediction.

The preparation code processes daily files in smaller batches to reduce RAM usage.

The processed dataset is saved as:

```text
data/weekly_modeling.parquet
```

**You only need to run `prepare` again if the raw data or feature-engineering code changes.**

### Step 3 — Train and evaluate the models

Run:

```bash
python hard_to_trade.py train
```

This step trains the models, evaluates their performance, and saves the results.

You do not need to download or prepare the data again when experimenting with the training code, provided `data/weekly_modeling.parquet` already exists.

All evaluation outputs are saved in the `results/` folder.

## 3. How the Project Works

### Data collection

We collect historical data for technology and financial stocks using the Alpaca API.

The main data sources are:

- Minute bars: price and volume information.
- Trades: executed transactions.
- Quotes: available bid and ask prices.

We use the IEX feed because it is more accessible. However, IEX represents only one exchange, not the entire US stock market.

### Trading difficulty

Trading difficulty is measured using the effective spread:

\[
\text{Effective Spread (bps)}
=2|\ln(P_{\text{trade}})-\ln(P_{\text{mid}})|\times 10{,}000
\]

where:

- \(P_{\text{trade}}\) is the executed trade price.
- \(P_{\text{mid}}\) is the midpoint between the bid and ask prices.
- bps means basis points.

Higher effective spread generally indicates greater trading costs and more difficult execution.

### Data preparation

The trading day is divided into seven time slots:

- 9:30–10:30 AM
- 10:30–11:30 AM
- 11:30 AM–12:30 PM
- 12:30–1:30 PM
- 1:30–2:30 PM
- 2:30–3:30 PM
- 3:30–4:00 PM

All times are US Eastern Time.

Observations are aggregated by stock, week, and time slot.

Features from week \(t-1\) are used to predict trading difficulty in week \(t\).

The target has three categories: **Easy, Medium, and Hard**. Classification thresholds are calculated using training data only.

## 4. Machine Learning Models

We compare three main approaches:

| Model | Description |
|---|---|
| Baseline | Uses historical ticker/time-slot patterns |
| Logistic Regression | Multiclass linear classification model |
| Neural Network | Two-hidden-layer MLP with 16 and 8 neurons |

The dataset is split chronologically into:

- 60% training
- 20% validation
- 20% testing

This prevents randomly mixing earlier and later weeks.

### Standard vs. reward-aware predictions

Each machine-learning model is evaluated using two decision methods.

**Standard predictions** choose the class with the highest predicted probability.

**Reward-aware predictions** choose the class with the highest expected reward under our custom reward matrix.

Reward-aware predictions may have lower classification accuracy but better reward scores because different mistakes have different costs.

These reward scores are not actual trading profits.

## 5. Additional Experiments

Besides the main model comparison, we perform several experiments.

### Feature ablation

We remove selected feature groups and retrain the model to understand which features contribute most to performance.

We also test removing only `lag_effective_spread_bps`, which represents historical effective spread.

### Cross-sector generalization

We compare models trained on different sectors:

- Technology → Financial: train on technology stocks and test on financial stocks.
- Financial → Financial: train and test on financial stocks using different time periods.
- Financial baseline: provides a simple benchmark for financial-stock predictions.

This helps us evaluate whether models generalize across sectors.

### Bootstrap confidence intervals

We use 2,000 ticker-cluster bootstrap resamples to estimate 95% confidence intervals.

We also compare models directly using paired bootstrap differences, including logistic regression versus the baseline and neural network versus logistic regression.

This accounts for dependence between observations from the same stock, although it does not fully account for market-wide dependence across stocks.

### Data-quality sensitivity analysis

TFC and USB are excluded from the current training and evaluation analysis because their effective-spread thresholds were nearly zero.

The original full-sample results can be retained separately for comparison.

## 6. Output Files

After running the pipeline, the main files are:

```text
hard_to_trade_project/
│
├── hard_to_trade.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   └── weekly_modeling.parquet
│
└── results/
    ├── metrics.csv
    ├── train_thresholds.csv
    ├── ticker_bootstrap_ci.csv
    ├── financial_transfer_gap_ci.csv
    ├── model_comparison_ci.csv
    ├── intraday_spread.png
    ├── *_confusion.png
    └── *.joblib
```

**Important output files:**

| File | Purpose |
|---|---|
| `metrics.csv` | Accuracy, macro-F1, and reward for each experiment |
| `train_thresholds.csv` | Training-based Easy/Medium/Hard thresholds |
| `ticker_bootstrap_ci.csv` | Confidence intervals for individual models |
| `financial_transfer_gap_ci.csv` | Financial versus technology transfer comparisons |
| `model_comparison_ci.csv` | Paired comparisons between the main models |
| `intraday_spread.png` | Trading difficulty throughout the trading day |
| `*_confusion.png` | Confusion matrices showing classification errors |
| `*.joblib` | Saved fitted models |

## 7. Important Limitations

- IEX data covers only one exchange, so results may differ from consolidated-market data.
- Effective spread is a proxy for execution cost, not the realized cost of a specific trading order.
- Missing or low-quality quotes and trades can affect the labels.
- The current experiment predicts weekly trading difficulty rather than individual trade outcomes.
- The neural network uses scikit-learn's MLPClassifier and does not implement dropout.
- Cryptocurrency generalization is not implemented.
- The fixed stock list may introduce survivorship bias.
- The bootstrap accounts for within-stock dependence but not all market-wide effects.
- Results are based on historical observations and do not guarantee future trading performance.

## 8. Quick Commands for Teammates

If you already have the complete raw dataset:

```bash
python hard_to_trade.py prepare
python hard_to_trade.py train
```

If you already have `data/weekly_modeling.parquet`:

```bash
python hard_to_trade.py train
```

If you have neither raw nor processed data, start with the full `fetch` command in Section 2.

**Tip:** For our current project, the easiest way to reproduce the results is to share the processed `weekly_modeling.parquet` file and run `train`. This avoids downloading six months of historical data again.

---

**DS 402 — Group 3**

Project: *Hard to Trade? Predicting Intraday Stock Trading Difficulty Using Machine Learning*

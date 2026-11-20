"""DS 402 Group 3: Hard to Trade?

Predict intraday execution difficulty by (ticker, hour) bucket.

Commands:
    python hard_to_trade.py fetch --start 2026-04-01 --end 2026-10-03
    python hard_to_trade.py prepare
    python hard_to_trade.py train

Requires environment variables:
    APCA_API_KEY_ID
    APCA_API_SECRET_KEY

Note: IEX data has limited exchange coverage.

CHANGES IN THIS VERSION
    1. Bad-quote filter (very wide / crossed IEX quotes) + outlier cap on
       effective spread + minimum matched trades per bucket.
    2. quoted_spread_bps bug fixed (was dollars*10000, now cents).
    3. Reward-aware predictions (pick the class with best expected reward).
    4. Logistic regression uses class_weight="balanced".
    5. Per-ticker thresholds fall back to global when a ticker has too few rows.
    6. Reports show predicted class counts, plus "always Easy/Medium/Hard"
       reference rows, so you can see when a model is just guessing.
    7. prepare() prints a data-quality summary per ticker.
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed

import argparse
import os
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results"

TECH = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
    "AVGO", "ORCL", "CRM", "ADBE", "AMD", "INTC",
    "CSCO", "QCOM", "TXN", "IBM", "NOW", "INTU",
]

FIN = [
    "JPM", "BAC", "WFC", "C", "GS", "MS",
    "BLK", "SCHW", "AXP", "V", "MA", "COF",
    "USB", "PNC", "BK", "TFC", "AIG", "MET",
]

ETF = {
    "tech": "XLK",
    "financial": "XLF",
}

CLASSES = ["Easy", "Medium", "Hard"]

# Rows = predicted class; columns = actual class.
# Reward matrix from the project proposal.
REWARD = np.array([
    [1, -1, -3],
    [0,  1, -1],
    [-1, 0,  1],
], dtype=float)

# ---- Data-quality settings (tune these) ----
# IEX shows only IEX's own top of book. For many stocks that book is
# stale or very wide, which makes the "spread" look like hundreds of bps.
MAX_QUOTE_SPREAD_BPS = 50.0     # drop quotes wider than this (bps of midpoint)
MAX_EFFECTIVE_SPREAD_BPS = 200.0  # drop trades this far from midpoint (bad ticks)
MIN_BUCKET_TRADES = 3           # min matched trades in a day-hour bucket
MIN_WEEKLY_TRADES = 20          # min matched trades in a week-hour bucket
MIN_TICKER_TRAIN_ROWS = 6       # fewer rows -> use global thresholds

FEATURE_GROUPS = {
    "liquidity": [
        "quoted_spread_cents",
        "relative_spread_bps",
        "bid_size",
        "ask_size",
        "depth_imbalance",
        "dollar_volume",
        "volume",
        "trade_count",
        "effective_spread_bps",
    ],
    "deviation": [
        "return",
        "sector_return",
        "excess_return",
        "abs_excess_return",
    ],
    "activity": [
        "realized_volatility",
        "price_range",
        "reversals",
        "volume_volatility",
        "bar_count",
    ],
}

FEATURES = [
    feature
    for group in FEATURE_GROUPS.values()
    for feature in group
]


# ============================================================
# 2. TIME BUCKETS
# ============================================================

def bucket_index(ts):
    """Split 9:30 AM-4:00 PM ET into seven time slots."""
    minutes = ts.dt.hour * 60 + ts.dt.minute - 570
    return (minutes // 60).astype("int64")


def session_filter(frame):
    """Keep regular US market hours, Monday-Friday."""
    if frame.empty:
        return frame

    frame = frame.copy()

    frame["timestamp"] = (
        pd.to_datetime(frame["timestamp"], utc=True)
        .dt.tz_convert("America/New_York")
    )

    local = frame["timestamp"]
    minutes = local.dt.hour * 60 + local.dt.minute

    mask = (
        (minutes >= 570)
        & (minutes < 960)
        & (local.dt.dayofweek < 5)
    )

    return frame.loc[mask].copy()


def flatten_alpaca(df):
    """Convert Alpaca response to a regular DataFrame."""
    if df is None or len(df) == 0:
        return pd.DataFrame()

    result = df.reset_index().copy()

    if "symbol" in result.columns:
        result = result.drop(columns=["symbol"])

    return session_filter(result)


# ============================================================
# 3. DOWNLOAD ALPACA DATA
# ============================================================

def fetch(args):
    """Fetch missing ticker-days concurrently, with bounded retries."""
    import threading
    import time
    import random
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import (
        StockTradesRequest, StockQuotesRequest, StockBarsRequest,
    )
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed

    key = os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY before fetching.")

    feed = DataFeed.SIP if args.feed == "sip" else DataFeed.IEX
    symbols = TECH + FIN + ["XLK", "XLF"]
    if args.max_symbols:
        symbols = symbols[:args.max_symbols]

    days = pd.bdate_range(
        args.start, pd.Timestamp(args.end) - pd.Timedelta(days=1)
    )
    jobs = []
    for sym in symbols:
        for day in days:
            dest = DATA / "raw" / sym / f"{day:%Y-%m-%d}.parquet"
            if args.force or not dest.exists():
                jobs.append((sym, day, dest))

    print(f"Missing ticker-days to check: {len(jobs):,}; workers: {args.workers}", flush=True)
    if not jobs:
        print("No missing bar files to fetch.")
        return

    # One client per thread avoids sharing a client/session across workers.
    local = threading.local()

    def get_client():
        if not hasattr(local, "client"):
            local.client = StockHistoricalDataClient(key, secret)
        return local.client

    def download_one(job):
        sym, day, dest = job
        if dest.exists() and not args.force:
            return "skipped", sym, day, ""

        start = day.tz_localize("America/New_York").tz_convert("UTC").to_pydatetime()
        end = (day + pd.Timedelta(days=1)).tz_localize("America/New_York").tz_convert("UTC").to_pydatetime()
        dest.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(1, args.retries + 1):
            try:
                client = get_client()
                bars = flatten_alpaca(client.get_stock_bars(StockBarsRequest(
                    symbol_or_symbols=sym, timeframe=TimeFrame.Minute,
                    start=start, end=end, feed=feed,
                )).df)
                if bars.empty:
                    return "empty", sym, day, "No bars (possibly a market holiday)"
                bars["symbol"] = sym

                if sym not in ("XLK", "XLF"):
                    quotes = flatten_alpaca(client.get_stock_quotes(StockQuotesRequest(
                        symbol_or_symbols=sym, start=start, end=end, feed=feed,
                    )).df)
                    trades = flatten_alpaca(client.get_stock_trades(StockTradesRequest(
                        symbol_or_symbols=sym, start=start, end=end, feed=feed,
                    )).df)
                    if not quotes.empty:
                        quotes.to_parquet(dest.with_name(dest.stem + "_quotes.parquet"))
                    if not trades.empty:
                        trades.to_parquet(dest.with_name(dest.stem + "_trades.parquet"))

                # Mark the day complete only after the other requests succeed.
                # Atomic replacement prevents a partially written bar file being
                # mistaken for a completed download after an interruption.
                tmp = dest.with_suffix(".parquet.tmp")
                try:
                    bars.to_parquet(tmp)
                    os.replace(tmp, dest)
                finally:
                    tmp.unlink(missing_ok=True)
                return "saved", sym, day, ""
            except Exception as exc:
                message = str(exc)
                # Authentication/authorization errors won't improve by retrying.
                if any(token in message.lower() for token in (
                    "401 unauthorized", "403 forbidden", "subscription required",
                )):
                    return "failed", sym, day, message
                if attempt == args.retries:
                    return "failed", sym, day, message
                time.sleep(min(2 ** (attempt - 1), 30) + random.uniform(0, 1))

    counts = {"saved": 0, "empty": 0, "skipped": 0, "failed": 0}
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_one, job) for job in jobs]
        for future in as_completed(futures):
            status, sym, day, detail = future.result()
            counts[status] += 1
            if status == "saved":
                print(f"Saved {sym} {day.date()}", flush=True)
            elif status == "failed":
                failed.append((sym, day.date(), detail))
                warnings.warn(f"{sym} {day.date()}: {detail}")

    print("Fetch summary:", counts, flush=True)
    if failed:
        log = DATA / "fetch_failures.csv"
        pd.DataFrame(failed, columns=["symbol", "date", "error"]).to_csv(log, index=False)
        print(f"Failed dates listed in {log}. Re-run fetch to retry them.", flush=True)
    print("Fetch finished. IEX data is not a full-market representation; SIP requires subscription.")


# ============================================================
# 4. LOAD HISTORICAL DATA
# ============================================================

def load_raw(symbol, suffix=""):
    folder = DATA / "raw" / symbol

    files = sorted(
        folder.glob(f"????-??-??{suffix}.parquet")
    )

    if not files:
        return pd.DataFrame()

    return pd.concat(
        [pd.read_parquet(file) for file in files],
        ignore_index=True,
    )


# ============================================================
# 5. QUOTE CLEANING + EFFECTIVE SPREAD
# ============================================================

def clean_quotes(quotes):
    """Keep sane quotes only.

    Drops: missing values, non-positive bid, crossed/locked-bad quotes,
    and quotes wider than MAX_QUOTE_SPREAD_BPS (stale or empty books).
    Adds: midpoint, rel_spread_bps (quoted spread in bps of midpoint).
    """
    if quotes.empty:
        return quotes

    q = quotes.dropna(
        subset=["bid_price", "ask_price", "bid_size", "ask_size"]
    ).copy()

    q = q[
        (q.bid_price > 0)
        & (q.ask_price >= q.bid_price)
    ].copy()

    q["midpoint"] = (q.bid_price + q.ask_price) / 2

    q["rel_spread_bps"] = (
        10000 * (q.ask_price - q.bid_price) / q["midpoint"]
    )

    q = q[q.rel_spread_bps <= MAX_QUOTE_SPREAD_BPS].copy()

    return q


def calc_effective_spread(trades, quotes):
    """Match each trade to the latest preceding (clean) quote.

    Effective spread in basis points:
        20000 * abs(log(trade_price / midpoint))

    This is a log-price approximation to the standard
    relative effective spread 2*|P - M| / M.
    """

    empty = pd.DataFrame(
        columns=["timestamp", "effective_spread_bps"]
    )

    if trades.empty or quotes.empty:
        return empty

    t = (
        trades[["timestamp", "price"]]
        .dropna()
        .sort_values("timestamp")
        .copy()
    )

    q = clean_quotes(quotes)

    if t.empty or q.empty:
        return empty

    q = (
        q[["timestamp", "midpoint"]]
        .sort_values("timestamp")
        .rename(columns={"timestamp": "quote_time"})
    )

    merged = pd.merge_asof(
        t,
        q,
        left_on="timestamp",
        right_on="quote_time",
        direction="backward",
        tolerance=pd.Timedelta(seconds=5),
    )

    merged = merged.dropna(subset=["midpoint"])

    merged = merged[
        (merged.price > 0)
        & (merged.midpoint > 0)
    ].copy()

    merged["effective_spread_bps"] = (
        20000
        * np.abs(
            np.log(merged.price)
            - np.log(merged.midpoint)
        )
    )

    # Drop obvious bad ticks (late prints, out-of-sequence trades)
    merged = merged[
        merged.effective_spread_bps <= MAX_EFFECTIVE_SPREAD_BPS
    ].copy()

    return merged


# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================

def minute_features(bars, quotes, trades):

    b = bars.copy().sort_values("timestamp")

    b["date"] = b.timestamp.dt.date
    b["slot"] = bucket_index(b.timestamp)

    b["dollar_volume"] = (
        b["close"] * b["volume"]
    )

    b["log_ret"] = np.log(b["close"]).diff()

    b.loc[
        b.date.ne(b.date.shift()),
        "log_ret",
    ] = np.nan

    b["ret_sign"] = np.sign(b["log_ret"])

    b["reversal"] = (
        (b.ret_sign * b.ret_sign.shift() < 0)
        & b.date.eq(b.date.shift())
    ).astype(int)

    b["range"] = (
        (b["high"] - b["low"])
        / b["close"].replace(0, np.nan)
    )

    keys = ["date", "slot"]

    out = b.groupby(keys).agg(
        first_price=("open", "first"),
        last_price=("close", "last"),
        volume=("volume", "sum"),
        dollar_volume=("dollar_volume", "sum"),
        trade_count=(
            ("trade_count", "sum")
            if "trade_count" in b
            else ("volume", "count")
        ),
        realized_volatility=(
            "log_ret",
            lambda x: np.sqrt(
                np.nansum(np.square(x))
            ),
        ),
        price_range=("range", "mean"),
        reversals=("reversal", "sum"),
        volume_volatility=("volume", "std"),
        bar_count=("close", "count"),
    ).reset_index()

    out["return"] = np.log(
        out.last_price / out.first_price
    )

    # Spread and liquidity characteristics (clean quotes only)
    q = clean_quotes(quotes) if not quotes.empty else quotes

    if not q.empty:

        q = q.copy()

        q["date"] = q.timestamp.dt.date
        q["slot"] = bucket_index(q.timestamp)

        # Dollar spread in cents (NOT bps; the old version mislabeled it)
        q["quoted_spread_cents"] = 100 * (
            q.ask_price - q.bid_price
        )

        q["relative_spread_bps"] = q["rel_spread_bps"]

        q["depth_imbalance"] = (
            (q.bid_size - q.ask_size)
            / (q.bid_size + q.ask_size).replace(
                0, np.nan
            )
        )

        qa = q.groupby(keys).agg(
            quoted_spread_cents=(
                "quoted_spread_cents", "median"
            ),
            relative_spread_bps=(
                "relative_spread_bps", "median"
            ),
            bid_size=("bid_size", "median"),
            ask_size=("ask_size", "median"),
            depth_imbalance=(
                "depth_imbalance", "median"
            ),
        ).reset_index()

        out = out.merge(
            qa, on=keys, how="left"
        )

    # Ground truth from actual trade prices
    eff = calc_effective_spread(trades, quotes)

    if not eff.empty:

        eff["date"] = eff.timestamp.dt.date
        eff["slot"] = bucket_index(eff.timestamp)

        ea = eff.groupby(keys).agg(
            effective_spread_bps=(
                "effective_spread_bps", "median"
            ),
            matched_trades=(
                "effective_spread_bps", "count"
            ),
        ).reset_index()

        out = out.merge(
            ea, on=keys, how="left"
        )

        # Too few matched trades -> median is noise
        thin = out["matched_trades"].fillna(0) < MIN_BUCKET_TRADES
        out.loc[thin, "effective_spread_bps"] = np.nan

    # Make sure these columns always exist
    for col in ["effective_spread_bps", "matched_trades"]:
        if col not in out.columns:
            out[col] = np.nan

    return out


# ============================================================
# 7. PREPARE WEEKLY DATASET
# ============================================================

def prepare(args):
    """Build daily hourly features without loading months of quotes into RAM."""
    import gc

    DATA.mkdir(exist_ok=True)

    def daily_features(sym, include_microstructure=True):
        folder = DATA / "raw" / sym
        bar_files = sorted(folder.glob("????-??-??.parquet"))
        pieces = []
        for idx, bar_file in enumerate(bar_files, 1):
            quotes_file = bar_file.with_name(bar_file.stem + "_quotes.parquet")
            trades_file = bar_file.with_name(bar_file.stem + "_trades.parquet")
            bars = pd.read_parquet(bar_file)
            if bars.empty:
                continue
            quotes = (pd.read_parquet(quotes_file)
                      if include_microstructure and quotes_file.exists()
                      else pd.DataFrame())
            trades = (pd.read_parquet(trades_file)
                      if include_microstructure and trades_file.exists()
                      else pd.DataFrame())
            result = minute_features(bars, quotes, trades)
            if not result.empty:
                pieces.append(result)
            del bars, quotes, trades, result
            if idx % 20 == 0:
                print(f"  {sym}: {idx}/{len(bar_files)} days processed", flush=True)
                gc.collect()
        if not pieces:
            return pd.DataFrame()
        return pd.concat(pieces, ignore_index=True)

    etfs = {}
    for sym in ["XLK", "XLF"]:
        print(f"Preparing ETF {sym}...", flush=True)
        bars_features = daily_features(sym, include_microstructure=False)
        if not bars_features.empty:
            etfs[sym] = bars_features[["date", "slot", "return"]].rename(
                columns={"return": "sector_return"}
            )

    frames = []
    for sym in TECH + FIN:
        print(f"Preparing {sym}...", flush=True)
        f = daily_features(sym)
        if f.empty:
            continue
        sector = "tech" if sym in TECH else "financial"
        f["ticker"] = sym
        f["sector"] = sector
        if ETF[sector] in etfs:
            f = f.merge(etfs[ETF[sector]], on=["date", "slot"], how="left")
        else:
            f["sector_return"] = np.nan
        f["excess_return"] = f["return"] - f["sector_return"]
        f["abs_excess_return"] = f["excess_return"].abs()
        frames.append(f)
        gc.collect()

    if not frames:
        raise RuntimeError(
            "No raw Alpaca files. Run fetch first."
        )

    all_buckets = pd.concat(
        frames, ignore_index=True
    )

    all_buckets["date"] = pd.to_datetime(
        all_buckets.date
    )

    # Monday-based weekly periods
    all_buckets["week"] = (
        all_buckets.date
        - pd.to_timedelta(
            all_buckets.date.dt.weekday,
            unit="D",
        )
    )

    agg = {
        f: "median"
        for f in FEATURES
        if f in all_buckets.columns
    }

    agg["effective_spread_bps"] = "median"
    agg["bar_count"] = "sum"
    agg["matched_trades"] = "sum"

    weekly = all_buckets.groupby(
        ["ticker", "sector", "week", "slot"],
        as_index=False,
    ).agg(agg)

    weekly = weekly.sort_values(
        ["ticker", "slot", "week"]
    )

    # Previous week's features predict current week
    weekly["prior_week"] = (
        weekly["week"] - pd.Timedelta(days=7)
    )

    past = weekly[
        ["ticker", "slot", "week"]
        + [f for f in FEATURES if f in weekly]
    ].copy()

    past = past.rename(
        columns={
            "week": "prior_week",
            **{
                f: "lag_" + f
                for f in FEATURES
                if f in past
            },
        }
    )

    weekly = weekly.merge(
        past,
        on=["ticker", "slot", "prior_week"],
        how="inner",
    )

    before = len(weekly)

    weekly = weekly.dropna(
        subset=["effective_spread_bps"]
    )

    weekly = weekly[
        weekly["matched_trades"] >= MIN_WEEKLY_TRADES
    ].copy()

    print(
        f"Data quality: kept {len(weekly):,} of {before:,} "
        f"ticker-week-slot rows "
        f"(needs >= {MIN_WEEKLY_TRADES} matched trades)."
    )

    summary = (
        weekly.groupby("ticker")
        .agg(
            rows=("effective_spread_bps", "size"),
            median_eff_bps=("effective_spread_bps", "median"),
            p90_eff_bps=(
                "effective_spread_bps",
                lambda s: s.quantile(0.9),
            ),
            median_trades=("matched_trades", "median"),
        )
        .round(2)
    )

    print("\nPer-ticker summary (check for weird tickers):")
    print(summary.to_string())

    missing = sorted(
        set(TECH + FIN) - set(summary.index)
    )

    if missing:
        print("\nNo usable rows for:", ", ".join(missing))

    output = DATA / "weekly_modeling.parquet"

    weekly.to_parquet(
        output, index=False
    )

    print(
        f"\nWrote {len(weekly):,} weekly ticker-slot "
        f"rows to {output}"
    )

    print(
        "NOTE: Class thresholds are fitted only "
        "on the training set during train."
    )


# ============================================================
# 8. CHRONOLOGICAL SPLIT (MINIMUM 3 WEEKS)
# ============================================================

def split_weeks(df):
    """Chronological train/validation/test split."""

    weeks = np.array(
        sorted(df.week.dropna().unique())
    )

    n = len(weeks)

    if n < 3:
        raise ValueError(
            f"Found {n} distinct target weeks. "
            "Need at least 3 weeks for training, "
            "validation, and testing. "
            "Download more data and rerun prepare."
        )

    if n < 8:
        warnings.warn(
            f"Only {n} target weeks available. "
            "Results are preliminary and should "
            "not be used as final research findings.",
            UserWarning,
        )

    # Approximately 60/20/20 by complete weeks.
    a = max(
        1,
        min(n - 2, int(n * 0.6))
    )

    b = max(
        a + 1,
        min(n - 1, int(n * 0.8))
    )

    train_df = df[
        df.week.isin(weeks[:a])
    ].copy()

    val_df = df[
        df.week.isin(weeks[a:b])
    ].copy()

    test_df = df[
        df.week.isin(weeks[b:])
    ].copy()

    print("Total target weeks:", n)
    print("Training weeks:", len(weeks[:a]))
    print("Validation weeks:", len(weeks[a:b]))
    print("Testing weeks:", len(weeks[b:]))

    return train_df, val_df, test_df


# ============================================================
# 9. GROUND TRUTH LABELS
# ============================================================

def label_with_train_thresholds(train, val, test):
    """Per-ticker Easy/Medium/Hard cutoffs from TRAINING data only."""

    grp = train.groupby("ticker")["effective_spread_bps"]

    per = grp.quantile([1 / 3, 2 / 3]).unstack()
    counts = grp.count()

    global_cut = np.quantile(
        train["effective_spread_bps"].dropna(),
        [1 / 3, 2 / 3]
    )

    thresholds = {}

    for ticker, values in per.iterrows():

        lo = values.iloc[0]
        hi = values.iloc[1]

        if (
            counts.get(ticker, 0) >= MIN_TICKER_TRAIN_ROWS
            and np.isfinite(lo)
            and np.isfinite(hi)
            and lo < hi
        ):
            thresholds[ticker] = (float(lo), float(hi))
        else:
            thresholds[ticker] = tuple(global_cut)

    def label(frame):

        d = frame.copy()

        def assign(row):

            lo, hi = thresholds.get(
                row.ticker,
                tuple(global_cut)
            )

            return (
                int(row.effective_spread_bps > lo)
                + int(row.effective_spread_bps > hi)
            )

        d["y"] = d.apply(assign, axis=1)

        return d

    return (
        label(train),
        label(val),
        label(test),
        thresholds
    )


# ============================================================
# 10. MACHINE LEARNING MODELS
# ============================================================

def build_pipeline(feature_cols, kind, seed=42):

    numeric = [
        f for f in feature_cols
        if f.startswith("lag_")
    ]

    categorical = [
        f for f in feature_cols
        if f in ("ticker", "slot")
    ]

    transformer = ColumnTransformer([
        (
            "num",
            Pipeline([
                (
                    "fill",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scale",
                    StandardScaler()
                ),
            ]),
            numeric,
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            categorical,
        ),
    ])

    if kind == "logistic":

        model = LogisticRegression(
            C=1.0,
            max_iter=1500,
            class_weight="balanced",
            random_state=seed,
        )

    else:

        # Two hidden layers: 16 and 8 neurons.
        # sklearn MLP does not support dropout.
        model = MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            alpha=0.01,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            max_iter=400,
            random_state=seed,
        )

    return Pipeline([
        ("pre", transformer),
        ("model", model),
    ])


def predict_max_reward(model, X):
    """Pick the class with the best EXPECTED reward.

    expected_reward[i] = sum_j REWARD[i, j] * P(actual = j)

    Example: P(Easy)=0.40, P(Medium)=0.30, P(Hard)=0.30
      predict Easy   : 1*0.4 - 1*0.3 - 3*0.3 = -0.8
      predict Medium : 0*0.4 + 1*0.3 - 1*0.3 =  0.0   <- best
      predict Hard   : -1*0.4 + 0 + 1*0.3    = -0.1
    So the model hedges with Medium when it is unsure.
    """

    probs = model.predict_proba(X)

    aligned = np.zeros((len(X), 3))

    for index, class_label in enumerate(model.classes_):
        aligned[:, int(class_label)] = probs[:, index]

    expected = aligned @ REWARD.T

    return np.argmax(expected, axis=1)


# ============================================================
# 11. EVALUATION
# ============================================================

def reward_score(y_true, y_pred):

    return float(
        REWARD[
            np.asarray(y_pred, dtype=int),
            np.asarray(y_true, dtype=int),
        ].mean()
    )


def report(name, y, pred, save_plot=True):

    pred = np.asarray(pred, dtype=int)

    scores = {
        "model": name,
        "n": len(y),
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(
            y,
            pred,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        ),
        "reward": reward_score(y, pred),
        "pred_easy": int((pred == 0).sum()),
        "pred_medium": int((pred == 1).sum()),
        "pred_hard": int((pred == 2).sum()),
    }

    print(
        f"{name:<34} acc={scores['accuracy']:.3f}  "
        f"f1={scores['macro_f1']:.3f}  "
        f"reward={scores['reward']:+.3f}  "
        f"pred E/M/H = {scores['pred_easy']}/"
        f"{scores['pred_medium']}/{scores['pred_hard']}"
    )

    if save_plot:

        cm = confusion_matrix(
            y, pred, labels=[0, 1, 2]
        )

        ConfusionMatrixDisplay(
            cm,
            display_labels=CLASSES,
        ).plot(
            cmap="Blues",
            values_format="d",
        )

        plt.title(name)
        plt.tight_layout()

        plt.savefig(
            OUT / f"{name}_confusion.png",
            dpi=160,
        )

        plt.close()

    return scores


def evaluate_both(name, model, X, y, save_plot=True):
    """Report standard (argmax) AND reward-aware predictions."""

    return [
        report(
            name + "_standard",
            y,
            model.predict(X),
            save_plot,
        ),
        report(
            name + "_reward",
            y,
            predict_max_reward(model, X),
            save_plot,
        ),
    ]



def ticker_cluster_bootstrap(evaluations, n_boot=2000, seed=42):
    """Paired cluster bootstrap: resample entire ticker histories, not rows.

    evaluations: mapping name -> (test_frame, predictions). For comparisons,
    all entries must share the same ticker/row ordering.
    """
    rng = np.random.default_rng(seed)
    out = []
    for name, (frame, predictions) in evaluations.items():
        y = frame.y.to_numpy(dtype=int)
        pred = np.asarray(predictions, dtype=int)
        ticker = frame.ticker.to_numpy()
        tickers = np.unique(ticker)
        groups = [np.flatnonzero(ticker == t) for t in tickers]
        if len(tickers) < 2:
            continue
        estimates = {"accuracy": [], "macro_f1": [], "reward": []}
        for _ in range(n_boot):
            sampled = rng.integers(0, len(groups), size=len(groups))
            idx = np.concatenate([groups[i] for i in sampled])
            yt, yp = y[idx], pred[idx]
            estimates["accuracy"].append(accuracy_score(yt, yp))
            estimates["macro_f1"].append(f1_score(
                yt, yp, labels=[0, 1, 2], average="macro", zero_division=0
            ))
            estimates["reward"].append(reward_score(yt, yp))
        for metric, values in estimates.items():
            lo, hi = np.percentile(values, [2.5, 97.5])
            out.append({"model": name, "metric": metric,
                        "estimate": {"accuracy": accuracy_score,
                                     "macro_f1": lambda a,b: f1_score(a,b,labels=[0,1,2],average="macro",zero_division=0),
                                     "reward": reward_score}[metric](y,pred),
                        "ci_lower": lo, "ci_upper": hi,
                        "n_tickers": len(tickers), "n_boot": n_boot})
    return pd.DataFrame(out)


def paired_ticker_bootstrap_difference(frame, preds_a, preds_b,
                                       name_a, name_b, n_boot=2000, seed=42):
    """CI for A minus B on identical financial test observations."""
    rng = np.random.default_rng(seed)
    tickers = frame.ticker.to_numpy()
    unique = np.unique(tickers)
    groups = [np.flatnonzero(tickers == t) for t in unique]
    y = frame.y.to_numpy(dtype=int)
    a, b = np.asarray(preds_a, dtype=int), np.asarray(preds_b, dtype=int)
    output = []
    if len(unique) < 2:
        return pd.DataFrame()
    for metric, fn in [
        ("accuracy", accuracy_score),
        ("macro_f1", lambda yt, yp: f1_score(yt, yp, labels=[0,1,2], average="macro", zero_division=0)),
        ("reward", reward_score),
    ]:
        differences = []
        for _ in range(n_boot):
            chosen = rng.integers(0, len(groups), size=len(groups))
            idx = np.concatenate([groups[i] for i in chosen])
            differences.append(fn(y[idx], a[idx]) - fn(y[idx], b[idx]))
        lo, hi = np.percentile(differences, [2.5,97.5])
        output.append({"model_a": name_a, "model_b": name_b,
                       "metric": metric, "difference_a_minus_b": fn(y,a)-fn(y,b),
                       "ci_lower": lo, "ci_upper": hi,
                       "n_tickers": len(unique), "n_boot": n_boot})
    return pd.DataFrame(output)

# ============================================================
# 12. BASELINE MODEL
# ============================================================

def baseline(train, other):

    mode = (
        train.groupby(["ticker", "slot"])["y"]
        .agg(
            lambda s: int(
                s.value_counts().index[0]
            )
        )
        .to_dict()
    )

    ticker_mode = (
        train.groupby("ticker")["y"]
        .agg(
            lambda s: int(
                s.value_counts().index[0]
            )
        )
        .to_dict()
    )

    overall = int(
        train.y.value_counts().index[0]
    )

    return np.array([
        mode.get(
            (row.ticker, row.slot),
            ticker_mode.get(
                row.ticker, overall
            ),
        )
        for row in other.itertuples()
    ])


# ============================================================
# 13. TRAIN AND COMPARE MODELS
# ============================================================

def train(args):

    OUT.mkdir(exist_ok=True)

    df = pd.read_parquet(
        DATA / "weekly_modeling.parquet"
    )

    # Sensitivity analysis: exclude tickers with degenerate near-zero spread thresholds.
    excluded = {"TFC", "USB"}
    df = df.loc[~df.ticker.isin(excluded)].copy()
    print("Sensitivity analysis: excluded tickers:", ", ".join(sorted(excluded)))

    tr, va, te = split_weeks(df)

    tr, va, te, thresholds = (
        label_with_train_thresholds(
            tr, va, te
        )
    )

    print(
        "Train/validation/test rows:",
        len(tr), len(va), len(te)
    )

    print(
        "Class mix in test (E/M/H):",
        [int((te.y == k).sum()) for k in range(3)],
    )

    cols = (
        ["ticker", "slot"]
        + [
            "lag_" + f
            for f in FEATURES
            if "lag_" + f in df.columns
        ]
    )

    results = []
    bootstrap_evaluations = {}
    preds = {}

    print("\n--- Reference rows (what dumb guessing scores) ---")

    for k, cname in enumerate(CLASSES):
        results.append(
            report(
                f"always_{cname.lower()}_test",
                te.y,
                np.full(len(te), k),
                False,
            )
        )

    print("\n--- Baseline ---")

    base_pred = baseline(tr, te)
    results.append(report("baseline_test", te.y, base_pred))
    bootstrap_evaluations["baseline_test"] = (te, base_pred)

    models = {}

    print("\n--- Logistic regression and neural network ---")

    for kind in ["logistic", "neural_net"]:

        pipe = build_pipeline(
            cols, kind
        )

        pipe.fit(
            tr[cols], tr.y
        )

        results += evaluate_both(
            kind + "_validation",
            pipe, va[cols], va.y,
        )

        results += evaluate_both(
            kind + "_test",
            pipe, te[cols], te.y,
        )

        preds[kind + "_standard"] = pipe.predict(te[cols])
        preds[kind + "_reward"] = predict_max_reward(pipe, te[cols])
        bootstrap_evaluations[kind + "_test_standard"] = (te, preds[kind + "_standard"])
        bootstrap_evaluations[kind + "_test_reward"] = (te, preds[kind + "_reward"])

        models[kind] = pipe

        joblib.dump(
            pipe,
            OUT / f"{kind}.joblib",
        )

    print("\n--- Paired model comparisons (95% ticker CI) ---")
    comparisons = [
        ("logistic_standard", "baseline", preds["logistic_standard"], base_pred),
        ("logistic_reward", "baseline", preds["logistic_reward"], base_pred),
        ("neural_net_standard", "logistic_standard", preds["neural_net_standard"], preds["logistic_standard"]),
        ("neural_net_reward", "logistic_reward", preds["neural_net_reward"], preds["logistic_reward"]),
    ]
    comparison_ci = pd.concat([
        paired_ticker_bootstrap_difference(te, a, b, name_a, name_b)
        for name_a, name_b, a, b in comparisons
    ], ignore_index=True)
    comparison_ci.to_csv(OUT / "model_comparison_ci.csv", index=False)
    print(comparison_ci.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    print("\n--- Feature ablation (logistic, test set) ---")

    for group, removed in FEATURE_GROUPS.items():

        ab_cols = [
            c for c in cols
            if c not in [
                "lag_" + f
                for f in removed
            ]
        ]

        model = build_pipeline(
            ab_cols, "logistic"
        )

        model.fit(
            tr[ab_cols], tr.y
        )

        results += evaluate_both(
            "ablation_without_" + group,
            model, te[ab_cols], te.y,
            save_plot=False,
        )

    print("\n--- Ablation: remove ONLY lag_effective_spread_bps ---")
    single_cols = [c for c in cols if c != "lag_effective_spread_bps"]
    if len(single_cols) == len(cols):
        print("WARNING: lag_effective_spread_bps not found; skipping single-feature ablation")
    else:
        single_model = build_pipeline(single_cols, "logistic")
        single_model.fit(tr[single_cols], tr.y)
        results += evaluate_both("ablation_without_lag_effective_spread_bps",
                                 single_model, te[single_cols], te.y, save_plot=False)
        bootstrap_evaluations["ablation_without_lag_effective_spread_bps_standard"] = (
            te, single_model.predict(te[single_cols]))
        bootstrap_evaluations["ablation_without_lag_effective_spread_bps_reward"] = (
            te, predict_max_reward(single_model, te[single_cols]))

    print("\n--- Technology -> Financial generalization ---")

    tech_train = tr[
        tr.sector == "tech"
    ]

    fin_test = te[
        te.sector == "financial"
    ]

    if (
        len(tech_train) > 20
        and len(fin_test) > 0
        and tech_train.y.nunique() > 1
    ):

        cross_cols = [
            c for c in cols
            if c != "ticker"
        ]

        model = build_pipeline(
            cross_cols, "logistic"
        )

        model.fit(
            tech_train[cross_cols],
            tech_train.y,
        )

        results += evaluate_both(
            "tech_to_financial",
            model, fin_test[cross_cols], fin_test.y,
        )
        tech_std = model.predict(fin_test[cross_cols])
        tech_reward = predict_max_reward(model, fin_test[cross_cols])
        bootstrap_evaluations["tech_to_financial_standard"] = (fin_test, tech_std)
        bootstrap_evaluations["tech_to_financial_reward"] = (fin_test, tech_reward)

        print("\n--- Financial-only baseline and in-sector model ---")
        fin_train = tr[tr.sector == "financial"]
        if len(fin_train) > 20 and fin_train.y.nunique() > 1:
            fin_base = baseline(fin_train, fin_test)
            results.append(report("financial_baseline_test", fin_test.y, fin_base))
            bootstrap_evaluations["financial_baseline_test"] = (fin_test, fin_base)
            fin_model = build_pipeline(cross_cols, "logistic")
            fin_model.fit(fin_train[cross_cols], fin_train.y)
            results += evaluate_both("financial_to_financial", fin_model,
                                     fin_test[cross_cols], fin_test.y)
            fin_std = fin_model.predict(fin_test[cross_cols])
            fin_reward = predict_max_reward(fin_model, fin_test[cross_cols])
            bootstrap_evaluations["financial_to_financial_standard"] = (fin_test, fin_std)
            bootstrap_evaluations["financial_to_financial_reward"] = (fin_test, fin_reward)
            diffs = pd.concat([
                paired_ticker_bootstrap_difference(
                    fin_test, fin_std, tech_std,
                    "financial_to_financial_standard", "tech_to_financial_standard"),
                paired_ticker_bootstrap_difference(
                    fin_test, fin_reward, tech_reward,
                    "financial_to_financial_reward", "tech_to_financial_reward"),
            ], ignore_index=True)
            diffs.to_csv(OUT / "financial_transfer_gap_ci.csv", index=False)
            print("\n--- Financial in-sector minus cross-sector transfer (95% ticker CI) ---")
            print(diffs.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
        else:
            print("Insufficient financial training data for in-sector comparison")

    print("\n--- Ticker-cluster bootstrap 95% confidence intervals ---")
    ci = ticker_cluster_bootstrap(bootstrap_evaluations)
    ci.to_csv(OUT / "ticker_bootstrap_ci.csv", index=False)
    print(ci.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Save metrics
    pd.DataFrame(results).to_csv(
        OUT / "metrics.csv",
        index=False,
    )

    # Save per-stock thresholds
    pd.DataFrame([
        {
            "ticker": ticker,
            "easy_medium": values[0],
            "medium_hard": values[1],
        }
        for ticker, values in thresholds.items()
    ]).to_csv(
        OUT / "train_thresholds.csv",
        index=False,
    )

    # Intraday spread visualization
    g = df.groupby("slot")[
        "effective_spread_bps"
    ].median()

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.plot(
        g.index,
        g.values,
        marker="o",
    )

    ax.set_xticks(
        range(7),
        [
            "9:30", "10:30", "11:30",
            "12:30", "13:30", "14:30",
            "15:30",
        ],
    )

    ax.set_xlabel(
        "ET bucket start"
    )

    ax.set_ylabel(
        "Median effective spread (bps)"
    )

    ax.set_title(
        "Intraday Trading Difficulty"
    )

    fig.tight_layout()

    fig.savefig(
        OUT / "intraday_spread.png",
        dpi=160,
    )

    plt.close(fig)

    print("\nSaved results to", OUT)


# ============================================================
# 14. COMMAND LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="DS 402 Hard to Trade project"
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    fetch_parser = sub.add_parser("fetch")

    fetch_parser.add_argument(
        "--start",
        required=True,
    )

    fetch_parser.add_argument(
        "--end",
        required=True,
    )

    fetch_parser.add_argument(
        "--feed",
        choices=["iex", "sip"],
        default="iex",
    )

    fetch_parser.add_argument(
        "--max-symbols",
        type=int,
    )

    fetch_parser.add_argument(
        "--force",
        action="store_true",
    )

    fetch_parser.add_argument("--workers", type=int, default=4,
                              choices=range(1, 9), metavar="1-8")
    fetch_parser.add_argument("--retries", type=int, default=3)

    sub.add_parser("prepare")
    sub.add_parser("train")

    args = parser.parse_args()

    commands = {
        "fetch": fetch,
        "prepare": prepare,
        "train": train,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
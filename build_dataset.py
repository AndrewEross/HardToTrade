"""Step 4: combine everything into one modeling table.

    python build_dataset.py

Reads the interim spread aggregates and 1-minute bars, computes bar-based features,
creates easy/medium/hard labels (per-ticker terciles from the training period only),
adds one-week-lagged features, and saves:
    data/processed/bucket_dataset.parquet  (and .csv)

Modeling rule: use ONLY the *_lag1w columns as inputs (features from the same
ticker/weekday/bucket one week earlier) and `label` as the target.
"""
import pandas as pd

import config as C
from features import add_labels, add_lagged_features, bars_bucket_features


def load_spreads() -> pd.DataFrame:
    files = sorted(C.INTERIM_SPREADS.glob("*/*.parquet"))
    if not files:
        raise SystemExit("No spread files found. Run download_spreads.py first.")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def load_bar_features() -> pd.DataFrame:
    frames = []
    for ticker, sector in C.TICKER_SECTOR.items():
        sp, ep = C.INTERIM_BARS / f"{ticker}.parquet", C.INTERIM_BARS / f"{C.SECTOR_ETF[sector]}.parquet"
        if not (sp.exists() and ep.exists()):
            print(f"  skipping {ticker}: bars missing (run download_bars.py)")
            continue
        f = bars_bucket_features(pd.read_parquet(sp), pd.read_parquet(ep)).reset_index()
        f.insert(0, "ticker", ticker)
        frames.append(f)
    if not frames:
        raise SystemExit("No bar files found. Run download_bars.py first.")
    return pd.concat(frames, ignore_index=True)


def main():
    spreads = load_spreads()
    bars = load_bar_features()
    df = spreads.merge(bars, on=["ticker", "date", "bucket"], how="left")
    df = df.dropna(subset=[C.LABEL_COLUMN])                 # a bucket needs trades to have a label
    df["sector"] = df["ticker"].map(C.TICKER_SECTOR)
    df["bucket_label"] = df["bucket"].map(dict(enumerate(C.BUCKET_LABELS)))

    df = add_labels(df)

    id_cols = {"ticker", "date", "bucket", "sector", "bucket_label", "split",
               "q_lo", "q_hi", "label", "label_name"}
    feature_cols = [c for c in df.columns if c not in id_cols]
    df = add_lagged_features(df, feature_cols)
    df["has_lag"] = df[f"{C.LABEL_COLUMN}_lag1w"].notna()   # False for each series' first week

    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(C.PROCESSED / "bucket_dataset.parquet")
    df.to_csv(C.PROCESSED / "bucket_dataset.csv", index=False)

    # ------------------------------------------------ sanity summary
    print(f"\nRows: {len(df):,}   tickers: {df['ticker'].nunique()}   "
          f"dates: {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")
    print(f"Expected rows if complete: ~{df['ticker'].nunique() * df['date'].nunique() * C.N_BUCKETS:,}")
    print("\nLabel counts by split (should be roughly 1/3 each in train):")
    print(pd.crosstab(df["split"], df["label_name"]))
    print("\nMissing values, features (top 10):")
    print(df[feature_cols].isna().mean().sort_values(ascending=False).head(10).round(3))
    print("\nMean effective spread (bps) by bucket -- look for the U-shape:")
    print(df.groupby("bucket_label")[C.LABEL_COLUMN].mean().round(2))


if __name__ == "__main__":
    main()

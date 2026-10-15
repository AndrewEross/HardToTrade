"""Step 2: download 1-minute bars for every stock and sector ETF (cheap, one pull each).

    python download_bars.py            # skips symbols already downloaded
    python download_bars.py --force    # re-download everything

Saved to data/interim/bars/<SYMBOL>.parquet (regular hours only, UTC timestamps).
These feed the price-deviation, oscillation and volume features.
"""
import argparse
import time

import pandas as pd

import config as C
from alpaca_io import fetch_bars, get_client
from features import bucket_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    C.INTERIM_BARS.mkdir(parents=True, exist_ok=True)
    client = get_client()
    start = pd.Timestamp(f"{C.START_DATE} 04:00", tz="America/New_York").to_pydatetime()
    end = pd.Timestamp(f"{C.END_DATE} 20:00", tz="America/New_York").to_pydatetime()

    symbols = list(C.TICKER_SECTOR) + list(C.SECTOR_ETF.values())
    for i, sym in enumerate(symbols, 1):
        path = C.INTERIM_BARS / f"{sym}.parquet"
        if path.exists() and not args.force:
            print(f"[{i}/{len(symbols)}] {sym}: already downloaded, skipping")
            continue
        t0 = time.time()
        bars = fetch_bars(client, sym, start, end)
        if bars is None:
            print(f"[{i}/{len(symbols)}] {sym}: NO DATA (check the symbol or your feed access)")
            continue
        bars = bars[bucket_of(bars.index) >= 0]        # regular trading hours only
        bars.to_parquet(path)
        print(f"[{i}/{len(symbols)}] {sym}: {len(bars):,} bars saved ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()

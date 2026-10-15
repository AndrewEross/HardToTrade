"""Step 3: download quotes + trades one ticker-day at a time and keep only bucket aggregates.

    python download_spreads.py                       # all tickers, full window
    python download_spreads.py --tickers AAPL KEY    # a test run on two tickers
    python download_spreads.py --start 2026-09-01 --end 2026-09-05 --feed iex

Resumable: finished ticker-days are skipped, so you can stop (Ctrl+C) and rerun any time.
Raw ticks are never saved -- only the small aggregate table, as described in the proposal.
Output: data/interim/spreads/<TICKER>/<YYYY-MM-DD>.parquet
        (a <date>.empty marker means no data that day, e.g. a market holiday)
"""
import argparse
import time

import pandas as pd

import config as C
from alpaca_io import fetch_quotes, fetch_trades, get_client
from features import spread_bucket_features

NY = "America/New_York"


def process_day(client, ticker, day, feed):
    start = pd.Timestamp(f"{day:%Y-%m-%d} 09:25", tz=NY).to_pydatetime()   # a few minutes early so the
    end = pd.Timestamp(f"{day:%Y-%m-%d} 16:00", tz=NY).to_pydatetime()     # first trades have a quote
    quotes = fetch_quotes(client, ticker, start, end, feed=feed)
    if quotes is None:
        return None
    trades = fetch_trades(client, ticker, start, end, feed=feed)
    if trades is None:
        return None
    out = spread_bucket_features(quotes, trades)
    if out.empty:
        return None
    out = out.reset_index()
    out.insert(0, "ticker", ticker)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=list(C.TICKER_SECTOR))
    ap.add_argument("--start", default=C.START_DATE)
    ap.add_argument("--end", default=C.END_DATE)
    ap.add_argument("--feed", default=C.FEED, choices=["sip", "iex"])
    args = ap.parse_args()

    client = get_client()
    days = pd.bdate_range(args.start, args.end)
    t_start = time.time()
    done = 0
    for ticker in args.tickers:
        folder = C.INTERIM_SPREADS / ticker
        folder.mkdir(parents=True, exist_ok=True)
        for day in days:
            path = folder / f"{day:%Y-%m-%d}.parquet"
            if path.exists() or path.with_suffix(".empty").exists():
                continue
            t0 = time.time()
            try:
                out = process_day(client, ticker, day, args.feed)
            except KeyboardInterrupt:
                print("\nStopped. Rerun the same command to resume.")
                return
            except Exception as e:                       # noqa: BLE001
                print(f"{ticker} {day:%Y-%m-%d}: ERROR {e!s:.120} (will retry on next run)")
                continue
            if out is None:
                path.with_suffix(".empty").touch()
                print(f"{ticker} {day:%Y-%m-%d}: no data")
            else:
                out.to_parquet(path)
                print(f"{ticker} {day:%Y-%m-%d}: {len(out)} buckets ({time.time() - t0:.1f}s)")
            done += 1
        print(f"-- {ticker} complete. Elapsed {(time.time() - t_start) / 60:.1f} min, {done} days fetched this run")


if __name__ == "__main__":
    main()

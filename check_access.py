"""Step 1: verify what your Alpaca account can actually return, and estimate download cost.

Run this BEFORE the big download:
    python check_access.py

For a liquid and a less liquid stock it pulls a 15-minute midday sample of quotes and
trades under each feed (SIP and IEX), then extrapolates the number of API calls and
minutes needed for the whole project at the free plan's 200 calls/minute limit.
"""
import math
import time

import pandas as pd

import config as C
from alpaca_io import fetch_bars, fetch_quotes, fetch_trades, get_client

SAMPLE_DAY = "2026-09-15"                 # an ordinary Tuesday inside the study window
SAMPLE_SYMBOLS = ["AAPL", "KEY"]          # very liquid vs. less liquid
WINDOW = ("11:00", "11:15")               # 15 minutes of a typical midday stretch
PAGE_ROWS = 10_000                        # rows per API call (alpaca-py default max)
CALLS_PER_MIN = 200                       # free Basic plan


def ts(day, hhmm):
    return pd.Timestamp(f"{day} {hhmm}", tz="America/New_York").to_pydatetime()


def n_trading_days():
    return len(pd.bdate_range(C.START_DATE, C.END_DATE))


def main():
    client = get_client()
    n_days = n_trading_days()
    n_tickers = len(C.TICKER_SECTOR)
    print(f"Study plan: {n_tickers} tickers x ~{n_days} weekdays\n")

    # 1) Bars (cheap): do the stocks and the sector ETFs return data at all?
    print("== 1-minute bars, one sample day ==")
    missing = []
    for sym in list(C.TICKER_SECTOR) + list(C.SECTOR_ETF.values()):
        try:
            b = fetch_bars(client, sym, ts(SAMPLE_DAY, "09:30"), ts(SAMPLE_DAY, "16:00"))
            n = 0 if b is None else len(b)
        except Exception as e:                      # noqa: BLE001
            n = 0
            print(f"  {sym}: ERROR {e!s:.100}")
        if n == 0:
            missing.append(sym)
    print("  symbols with no data:", missing or "none", "\n")

    # 2) Quotes and trades (expensive): sample both feeds
    print(f"== Quotes/trades sample: {SAMPLE_DAY} {WINDOW[0]}-{WINDOW[1]} ET ==")
    scale = 390 / 15                                 # 15 minutes -> full session
    for feed in ("sip", "iex"):
        day_rows = []
        for sym in SAMPLE_SYMBOLS:
            try:
                t0 = time.time()
                q = fetch_quotes(client, sym, ts(SAMPLE_DAY, WINDOW[0]), ts(SAMPLE_DAY, WINDOW[1]), feed=feed)
                t = fetch_trades(client, sym, ts(SAMPLE_DAY, WINDOW[0]), ts(SAMPLE_DAY, WINDOW[1]), feed=feed)
                nq, nt = (0 if q is None else len(q)), (0 if t is None else len(t))
                secs = time.time() - t0
                print(f"  [{feed}] {sym}: {nq:,} quotes, {nt:,} trades in window ({secs:.1f}s)")
                day_rows.append((nq + nt) * scale)
            except Exception as e:                  # noqa: BLE001
                print(f"  [{feed}] {sym}: ERROR {e!s:.140}")
        if day_rows:
            rows_per_ticker_day = sum(day_rows) / len(day_rows)
            calls = rows_per_ticker_day * n_tickers * n_days / PAGE_ROWS
            print(f"  [{feed}] rough estimate: ~{rows_per_ticker_day:,.0f} rows per ticker-day, "
                  f"~{calls:,.0f} calls total, ~{calls / CALLS_PER_MIN / 60:.1f} hours at "
                  f"{CALLS_PER_MIN} calls/min\n")

    print("Notes: the window is midday, so the open (busier) will cost more. If the estimate is far too "
          "large, use the IEX feed, fewer tickers, or a shorter window (edit config.py).")


if __name__ == "__main__":
    main()

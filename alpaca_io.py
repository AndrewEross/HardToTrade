"""Thin wrapper around alpaca-py: client setup, retries, DataFrame conversion.

Credentials come from environment variables (never hard-code or commit them):
    export APCA_API_KEY_ID="..."
    export APCA_API_SECRET_KEY="..."
"""
import os
import time

import pandas as pd

import config as C


def get_client():
    from alpaca.data.historical import StockHistoricalDataClient
    key = os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables first.")
    return StockHistoricalDataClient(key, secret)


def _feed(name=None):
    from alpaca.data.enums import DataFeed
    return DataFeed.SIP if (name or C.FEED).lower() == "sip" else DataFeed.IEX


def _retry(fn, tries=6):
    """Retry transient errors (rate limits, 5xx, network) with backoff.
    Auth/permission errors (401/403) are raised immediately."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:                      # noqa: BLE001
            status = getattr(e, "status_code", None)
            if status in (401, 403) or i == tries - 1:
                raise
            wait = min(2 ** i, 60)
            print(f"    request failed ({e!s:.80}); retrying in {wait}s")
            time.sleep(wait)


def _to_frame(result, symbol):
    df = result.df
    if df is None or df.empty:
        return None
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")
    idx = pd.DatetimeIndex(df.index)
    df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return df.sort_index()


def fetch_bars(client, symbol, start, end, feed=None):
    """1-minute bars (includes extended hours; filter later)."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute,
                           start=start, end=end, feed=_feed(feed))
    return _to_frame(_retry(lambda: client.get_stock_bars(req)), symbol)


def fetch_quotes(client, symbol, start, end, feed=None):
    from alpaca.data.requests import StockQuotesRequest
    req = StockQuotesRequest(symbol_or_symbols=symbol, start=start, end=end, feed=_feed(feed))
    return _to_frame(_retry(lambda: client.get_stock_quotes(req)), symbol)


def fetch_trades(client, symbol, start, end, feed=None):
    from alpaca.data.requests import StockTradesRequest
    req = StockTradesRequest(symbol_or_symbols=symbol, start=start, end=end, feed=_feed(feed))
    return _to_frame(_retry(lambda: client.get_stock_trades(req)), symbol)

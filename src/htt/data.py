# Alpaca download + per-(ticker, day) bucket summarization.
# Raw ticks are never stored: each (ticker, day) is reduced to <=7 bucket rows and cached.
import logging
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from . import config as C

log = logging.getLogger(__name__)
ET = "America/New_York"

SUMMARY_COLS = [
    "symbol", "slot", "n_trades", "n_matched", "volume", "dollar_volume",
    "eff_vw", "eff_mean", "eff_p10", "eff_med", "eff_p90",
    "n_quotes", "qspread_mean", "qspread_med", "qspread_p90", "depth_mean", "depth_med",
    "ret_bps", "range_bps", "realized_vol_bps", "reversals_rate", "vol_of_volume",
    "monotonicity_err",
]


# ---------------------------------------------------------------- slotting
def stock_slot(ts: pd.DatetimeIndex) -> np.ndarray:
    # ts must be tz-aware in America/New_York. -1 = outside regular trading hours.
    mins = np.asarray(ts.hour * 60 + ts.minute - (9 * 60 + 30))
    slot = np.minimum(mins // 60, 6)
    return np.where((mins < 0) | (mins >= 390), -1, slot)


def crypto_slot(ts: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(ts.hour)  # ts in UTC


KINDS = {"stock": (stock_slot, ET), "crypto": (crypto_slot, "UTC")}


# ---------------------------------------------------------------- summarize
def _price_path_features(g: pd.DataFrame) -> dict:
    # g: trades of one bucket, sorted by time, with columns timestamp, price, size.
    s = g.set_index("timestamp")
    px = s["price"].resample("1min").last().ffill()
    vol = s["size"].resample("1min").sum()
    open_, close_ = g["price"].iloc[0], g["price"].iloc[-1]
    r = np.log(px).diff().dropna()
    signs = np.sign(r[r != 0].to_numpy())
    reversals = int((signs[1:] * signs[:-1] < 0).sum()) if len(signs) > 1 else 0

    mono = np.nan
    if len(px) >= 3:
        y = px.to_numpy(float)
        x = np.arange(len(y))
        errs = []
        for inc in (True, False):
            fit = IsotonicRegression(increasing=inc).fit_transform(x, y)
            errs.append(np.sqrt(np.mean((y - fit) ** 2)) / np.mean(y) * 1e4)
        mono = min(errs)  # 0 = perfectly monotone price path (cf. Hard to Park, Sec. 5.2.1)

    return dict(
        ret_bps=(close_ / open_ - 1) * 1e4,
        range_bps=(g["price"].max() - g["price"].min()) / open_ * 1e4,
        realized_vol_bps=float(r.std() * 1e4) if len(r) > 1 else np.nan,
        reversals_rate=reversals / max(len(px), 1),
        vol_of_volume=float(vol.std() / vol.mean()) if len(vol) > 1 and vol.mean() > 0 else np.nan,
        monotonicity_err=mono,
    )


def summarize_day(trades: pd.DataFrame, quotes: pd.DataFrame, symbol: str, kind: str = "stock") -> pd.DataFrame:
    # Reduce one (ticker, day) of ticks to one row per bucket.
    slot_fn, tz = KINDS[kind]
    if trades is None or quotes is None or len(trades) == 0 or len(quotes) == 0:
        return pd.DataFrame(columns=SUMMARY_COLS)

    t = trades.loc[(trades["price"] > 0) & (trades["size"] > 0), ["timestamp", "price", "size"]].copy()
    q = quotes.loc[(quotes["bid_price"] > 0) & (quotes["ask_price"] > quotes["bid_price"])].copy()
    q["mid"] = (q["bid_price"] + q["ask_price"]) / 2
    q["qspread_bps"] = (q["ask_price"] - q["bid_price"]) / q["mid"] * 1e4
    q["depth"] = q["bid_size"] + q["ask_size"]
    q = q[q["qspread_bps"] <= C.MAX_QSPREAD_BPS]
    if t.empty or q.empty:
        return pd.DataFrame(columns=SUMMARY_COLS)

    for d in (t, q):
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(tz)
    t = t.sort_values("timestamp")
    q = q.sort_values("timestamp")

    # prevailing quote for every trade
    m = pd.merge_asof(
        t, q[["timestamp", "mid", "qspread_bps", "depth"]], on="timestamp",
        direction="backward", tolerance=pd.Timedelta(seconds=C.MAX_QUOTE_AGE_S),
    )
    m["slot"] = slot_fn(pd.DatetimeIndex(m["timestamp"]))
    q["slot"] = slot_fn(pd.DatetimeIndex(q["timestamp"]))
    m = m[m["slot"] >= 0]
    # Effective spread (bps): 2*|P - M| / M. Unsigned, so no trade-direction (Lee-Ready) needed.
    m["eff_bps"] = 2 * (m["price"] - m["mid"]).abs() / m["mid"] * 1e4

    rows = []
    for slot, g in m.groupby("slot"):
        ge = g.dropna(subset=["eff_bps"])
        if len(ge) < C.MIN_TRADES:
            continue
        gq = q[q["slot"] == slot]
        w = ge["size"].to_numpy(float)
        row = dict(
            symbol=symbol, slot=int(slot), n_trades=len(g), n_matched=len(ge),
            volume=float(g["size"].sum()), dollar_volume=float((g["price"] * g["size"]).sum()),
            eff_vw=float(np.average(ge["eff_bps"], weights=w)), eff_mean=float(ge["eff_bps"].mean()),
            eff_p10=float(ge["eff_bps"].quantile(0.1)), eff_med=float(ge["eff_bps"].median()),
            eff_p90=float(ge["eff_bps"].quantile(0.9)),
            n_quotes=len(gq), qspread_mean=gq["qspread_bps"].mean(), qspread_med=gq["qspread_bps"].median(),
            qspread_p90=gq["qspread_bps"].quantile(0.9), depth_mean=gq["depth"].mean(), depth_med=gq["depth"].median(),
        )
        row.update(_price_path_features(g))
        rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLS)


# ---------------------------------------------------------------- Alpaca I/O
def get_client(kind: str = "stock"):
    import os
    if kind == "crypto":
        from alpaca.data.historical import CryptoHistoricalDataClient
        return CryptoHistoricalDataClient()  # crypto data needs no key
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])


def _to_df(resp) -> pd.DataFrame:
    df = resp.df
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)
    return df.reset_index()  # gives a 'timestamp' column


def fetch_day(client, symbol: str, day: str, kind: str = "stock", feed: str = C.FEED):
    # Returns (trades_df, quotes_df) for one symbol-day (regular hours for stocks, UTC day for crypto).
    if kind == "stock":
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockQuotesRequest, StockTradesRequest
        t0 = pd.Timestamp(day, tz=ET) + pd.Timedelta(hours=9, minutes=30)
        t1 = pd.Timestamp(day, tz=ET) + pd.Timedelta(hours=16)
        kw = dict(symbol_or_symbols=symbol, start=t0.to_pydatetime(), end=t1.to_pydatetime(), feed=DataFeed(feed))
        return (_to_df(client.get_stock_trades(StockTradesRequest(**kw))),
                _to_df(client.get_stock_quotes(StockQuotesRequest(**kw))))
    from alpaca.data.requests import CryptoQuotesRequest, CryptoTradesRequest
    t0 = pd.Timestamp(day, tz="UTC")
    kw = dict(symbol_or_symbols=symbol, start=t0.to_pydatetime(), end=(t0 + pd.Timedelta(days=1)).to_pydatetime())
    return (_to_df(client.get_crypto_trades(CryptoTradesRequest(**kw))),
            _to_df(client.get_crypto_quotes(CryptoQuotesRequest(**kw))))


def cache_path(symbol: str, day: str, kind: str):
    return C.DATA_RAW / kind / symbol.replace("/", "-") / f"{day}.parquet"


def collect(symbols, days, kind: str = "stock", feed: str = C.FEED, retries: int = 3):
    # Resumable: every (symbol, day) is cached; re-running skips what exists and retries failures.
    from tqdm.auto import tqdm
    client = get_client(kind)
    failed = []
    for sym in symbols:
        for day in tqdm(days, desc=sym, leave=False):
            p = cache_path(sym, day, kind)
            if p.exists():
                continue
            for attempt in range(retries):
                try:
                    tr, qu = fetch_day(client, sym, day, kind, feed)
                    out = summarize_day(tr, qu, sym, kind)
                    break
                except Exception as e:  # noqa: BLE001
                    log.warning("%s %s attempt %d failed: %s", sym, day, attempt + 1, e)
                    time.sleep(2 ** attempt)
            else:
                failed.append((sym, day))
                continue
            out["date"] = day  # holidays / empty days are cached as empty frames
            p.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(p)
    return failed


def load_cached(symbols, kind: str = "stock") -> pd.DataFrame:
    frames = []
    for sym in symbols:
        for f in sorted((C.DATA_RAW / kind / sym.replace("/", "-")).glob("*.parquet")):
            d = pd.read_parquet(f)
            if len(d):
                frames.append(d)
    if not frames:
        return pd.DataFrame(columns=SUMMARY_COLS + ["date"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out

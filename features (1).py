"""Feature, label and lag logic. Pure pandas, no network calls."""
import numpy as np
import pandas as pd

import config as C

NY = "America/New_York"
_OPEN_MIN = 9 * 60 + 30     # 9:30 a.m. in minutes since midnight
_SESSION_MIN = 390          # 9:30-16:00


# ------------------------------------------------------------ time helpers
def _ny(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    return idx.tz_localize("UTC").tz_convert(NY) if idx.tz is None else idx.tz_convert(NY)


def bucket_of(idx) -> np.ndarray:
    """Bucket number 0..6 for each timestamp, or -1 outside 9:30-16:00 ET."""
    ny = _ny(idx)
    m = np.asarray(ny.hour * 60 + ny.minute - _OPEN_MIN)
    return np.where((m >= 0) & (m < _SESSION_MIN), m // C.BUCKET_MINUTES, -1)


def local_date(idx) -> pd.DatetimeIndex:
    """Trading date (New York time) as a naive midnight timestamp."""
    return _ny(idx).tz_localize(None).normalize()


# ------------------------------------------- price-based features (1-min bars)
def minute_returns(bars: pd.DataFrame) -> pd.DataFrame:
    """Keep regular-hours bars; add bucket, date, and within-day log return of close."""
    b = bars.sort_index().copy()
    b["bucket"] = bucket_of(b.index)
    b = b[b["bucket"] >= 0].copy()
    b["date"] = local_date(b.index)
    b["r"] = np.log(b["close"]).groupby(b["date"]).diff()   # first bar of each day is NaN
    return b


def rolling_beta(stock: pd.DataFrame, etf: pd.DataFrame, window: int) -> pd.Series:
    """Stock-vs-ETF beta for each day, estimated from the PREVIOUS `window` days only."""
    j = stock[["date", "r"]].join(etf[["r"]], rsuffix="_etf", how="inner").dropna()
    j["se"] = j["r"] * j["r_etf"]
    j["ee"] = j["r_etf"] ** 2
    daily = j.groupby("date")[["se", "ee"]].sum()
    num = daily["se"].rolling(window, min_periods=5).sum().shift(1)
    den = daily["ee"].rolling(window, min_periods=5).sum().shift(1)
    return (num / den).rename("beta")


def bars_bucket_features(stock_bars: pd.DataFrame, etf_bars: pd.DataFrame) -> pd.DataFrame:
    """Price deviation and oscillation features per (date, bucket) from 1-minute bars."""
    s = minute_returns(stock_bars)
    e = minute_returns(etf_bars)

    beta = rolling_beta(s, e, C.BETA_WINDOW_DAYS)
    s = s.join(e[["r"]].rename(columns={"r": "r_etf"}), how="left")
    s["beta"] = s["date"].map(beta).fillna(1.0)          # fall back to beta = 1 early on
    s["abs_dev"] = (s["r"] - s["beta"] * s["r_etf"]).abs()
    s["abs_r"] = s["r"].abs()
    s["r2"] = s["r"] ** 2

    # Price reversals: sign flips between consecutive non-zero 1-minute returns.
    keys = [s["date"], s["bucket"]]
    sign = np.sign(s["r"]).replace(0, np.nan).groupby(keys).ffill()
    prev = sign.groupby(keys).shift()
    s["rev"] = ((sign != prev) & sign.notna() & prev.notna()).astype(int)

    g = s.groupby(["date", "bucket"])
    out = g.agg(
        n_bars=("close", "size"),
        n_ret=("r", "count"),
        abs_dev_mean=("abs_dev", "mean"),
        rv=("r2", "sum"),
        n_reversals=("rev", "sum"),
        net_move=("r", "sum"),
        abs_path=("abs_r", "sum"),
        hi=("high", "max"),
        lo=("low", "min"),
        first_open=("open", "first"),
        vol_total=("volume", "sum"),
        vol_mean=("volume", "mean"),
        vol_std=("volume", "std"),
    )
    out["reversal_rate"] = out["n_reversals"] / (out["n_ret"] - 1).clip(lower=1)
    out["efficiency_ratio"] = out["net_move"].abs() / out["abs_path"].replace(0, np.nan)
    out["range_bps"] = (out["hi"] - out["lo"]) / out["first_open"] * 1e4
    out["vol_cv"] = out["vol_std"] / out["vol_mean"]
    out["realized_vol_bps"] = np.sqrt(out["rv"]) * 1e4
    return out.drop(columns=["net_move", "abs_path", "hi", "lo", "first_open",
                             "vol_mean", "vol_std", "rv"])


# ---------------------------------- spread features and label source (quotes+trades)
def _has_excluded(conds) -> bool:
    if conds is None or (isinstance(conds, float) and np.isnan(conds)):
        return False
    if isinstance(conds, str):
        conds = [conds]
    return any(c in C.EXCLUDED_TRADE_CONDITIONS for c in conds)


def spread_bucket_features(quotes: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Quoted-spread, depth, volume and effective-spread aggregates per (date, bucket).

    quotes: index = timestamp; columns bid_price, ask_price, bid_size, ask_size
    trades: index = timestamp; columns price, size, (optional) conditions
    Quotes before 9:30 are used only to find the prevailing quote for early trades.
    """
    q = quotes.sort_index().copy()
    q = q[(q["bid_price"] > 0) & (q["ask_price"] > 0) & (q["ask_price"] >= q["bid_price"])]
    q["mid"] = (q["bid_price"] + q["ask_price"]) / 2
    q["spread_usd"] = q["ask_price"] - q["bid_price"]
    q["rel_bps"] = q["spread_usd"] / q["mid"] * 1e4
    q = q[q["rel_bps"] <= C.MAX_REL_SPREAD * 1e4]
    if q.empty:
        return pd.DataFrame()

    # --- trades: match each to the prevailing quote, compute effective spread
    t = trades.sort_index().copy()
    t = t[(t["price"] > 0) & (t["size"] > 0)]
    if "conditions" in t.columns:
        t = t[~t["conditions"].apply(_has_excluded)]
    t["bucket"] = bucket_of(t.index)
    t = t[t["bucket"] >= 0].copy()
    t["date"] = local_date(t.index)
    t = pd.merge_asof(t, q[["mid"]], left_index=True, right_index=True, direction="backward")
    t = t.dropna(subset=["mid"])
    t["eff_bps"] = 2 * np.abs(np.log(t["price"]) - np.log(t["mid"])) * 1e4   # proposal formula, in bps
    t = t[t["eff_bps"] <= C.MAX_EFF_SPREAD_BPS]
    t["dollar_vol"] = t["price"] * t["size"]
    t["eff_x_dv"] = t["eff_bps"] * t["dollar_vol"]

    ta = t.groupby(["date", "bucket"]).agg(
        n_trades=("price", "size"),
        share_volume=("size", "sum"),
        dollar_volume=("dollar_vol", "sum"),
        eff_spread_bps_mean=("eff_bps", "mean"),
        eff_spread_bps_median=("eff_bps", "median"),
        _num=("eff_x_dv", "sum"),
    )
    ta["eff_spread_bps_vw"] = ta["_num"] / ta["dollar_volume"]   # dollar-volume weighted
    ta = ta.drop(columns="_num")

    # --- quotes: spread and depth during the bucket
    q["bucket"] = bucket_of(q.index)
    q = q[q["bucket"] >= 0].copy()
    q["date"] = local_date(q.index)
    qa = q.groupby(["date", "bucket"]).agg(
        n_quotes=("mid", "size"),
        qspread_usd_mean=("spread_usd", "mean"),
        rel_spread_bps_mean=("rel_bps", "mean"),
        rel_spread_bps_p90=("rel_bps", lambda s: s.quantile(0.9)),
        bid_size_mean=("bid_size", "mean"),
        ask_size_mean=("ask_size", "mean"),
    )
    return qa.join(ta, how="outer")


# ------------------------------------------------------------ labels & lags
def add_labels(df: pd.DataFrame, label_col: str = None, split=None) -> pd.DataFrame:
    """Chronological train/val/test split and easy/medium/hard labels.

    Thresholds are per-ticker terciles of the label column computed on the TRAINING
    period only, so 'hard' means hard relative to that stock's own normal.
    """
    label_col = label_col or C.LABEL_COLUMN
    split = split or C.SPLIT
    df = df.copy()
    dates = np.sort(df["date"].unique())
    n = len(dates)
    cut1 = dates[min(int(n * split[0]), n - 1)]
    cut2 = dates[min(int(n * (split[0] + split[1])), n - 1)]
    df["split"] = np.where(df["date"] < cut1, "train",
                           np.where(df["date"] < cut2, "val", "test"))

    th = (df[df["split"] == "train"].groupby("ticker")[label_col]
          .quantile([1 / 3, 2 / 3]).unstack())
    th.columns = ["q_lo", "q_hi"]
    df = df.merge(th, left_on="ticker", right_index=True, how="left")

    v = df[label_col]
    lab = np.where(v <= df["q_lo"], 0, np.where(v <= df["q_hi"], 1, 2)).astype(float)
    lab[(v.isna() | df["q_lo"].isna()).to_numpy()] = np.nan
    df["label"] = lab
    df["label_name"] = df["label"].map({0.0: "easy", 1.0: "medium", 2.0: "hard"})
    return df


def add_lagged_features(df: pd.DataFrame, feature_cols) -> pd.DataFrame:
    """Add <col>_lag1w: the same ticker/weekday/bucket one week earlier.

    Use ONLY these lagged columns as model inputs, so the model never sees the
    period it is predicting. (If a holiday removes a day, the lag skips to the
    previous occurrence of that weekday.)
    """
    df = df.sort_values(["ticker", "date", "bucket"]).copy()
    df["weekday"] = df["date"].dt.dayofweek
    g = df.groupby(["ticker", "weekday", "bucket"])
    lagged = {f"{c}_lag1w": g[c].shift(1) for c in feature_cols}
    return pd.concat([df, pd.DataFrame(lagged, index=df.index)], axis=1)

# Bucket table -> labelled, leakage-safe modelling dataset.
#
# Bucket = (ticker, weekday, slot): one observation per calendar week.
# Prediction target: difficulty class of the bucket in week w.
# Features: ONLY information from weeks < w for the same (ticker, weekday, slot)
#           (previous-week value and trailing-N-week mean), plus calendar one-hots.
import numpy as np
import pandas as pd

from . import config as C

# Raw per-bucket measurements, grouped into the feature families used for ablation (RQ2).
FAMILY_RAW = {
    "spread_liquidity": ["eff_vw", "eff_med", "eff_p10", "eff_p90", "qspread_mean", "qspread_med",
                         "qspread_p90", "depth_mean", "depth_med", "volume", "dollar_volume",
                         "n_trades", "n_quotes"],
    "price_deviation": ["ret_bps", "dev_bps", "abs_dev_bps"],
    "oscillation": ["range_bps", "realized_vol_bps", "reversals_rate", "vol_of_volume", "monotonicity_err"],
}
LOG_COLS = ["volume", "dollar_volume", "n_trades", "n_quotes", "depth_mean", "depth_med"]
META = ["symbol", "sector", "date", "week", "weekday", "slot", "split", "y"]


def add_calendar_index(p: pd.DataFrame, train_frac=C.TRAIN_FRAC, val_frac=C.VAL_FRAC) -> pd.DataFrame:
    p = p.copy()
    d0 = p["date"].min()
    start = d0 - pd.Timedelta(days=d0.weekday())
    p["week"] = ((p["date"] - start).dt.days // 7).astype(int)
    p["weekday"] = p["date"].dt.weekday
    weeks = np.sort(p["week"].unique())
    n = len(weeks)
    tr_w = weeks[: int(n * train_frac)]
    va_w = weeks[int(n * train_frac): int(n * (train_frac + val_frac))]
    p["split"] = np.where(p["week"].isin(tr_w), "train", np.where(p["week"].isin(va_w), "val", "test"))
    return p


def add_labels(p: pd.DataFrame, label_col=C.LABEL_COL) -> pd.DataFrame:
    # Per-ticker terciles of the label, thresholds from TRAIN rows only -> 0 easy, 1 medium, 2 hard.
    thr = (p[p["split"] == "train"].groupby("symbol")[label_col]
           .quantile([1 / 3, 2 / 3]).unstack())
    thr.columns = ["lo", "hi"]
    p = p.join(thr, on="symbol")
    v = p[label_col]
    p["y"] = np.where(v <= p["lo"], 0, np.where(v <= p["hi"], 1, 2)).astype(float)
    p.loc[v.isna(), "y"] = np.nan
    return p.drop(columns=["lo", "hi"])


def _lagged(p: pd.DataFrame, col: str, k: int = 1) -> pd.Series:
    return p.groupby(["symbol", "weekday", "slot"], sort=False)[col].shift(k)


def add_busyness(p: pd.DataFrame) -> pd.DataFrame:
    # Relative busyness (Hard to Park, Sec. 5.3.2) using last week's trade count,
    # normalised by the ticker's TRAIN-period max / p90 / p75 of trade counts.
    p = p.sort_values(["symbol", "weekday", "slot", "week"]).copy()
    nt_lag = _lagged(p, "n_trades")
    tr = p[p["split"] == "train"].groupby("symbol")["n_trades"]
    for name, q in (("max", 1.0), ("90th", 0.9), ("75th", 0.75)):
        ref = tr.quantile(q)
        p[f"rel_busy_{name}"] = nt_lag / p["symbol"].map(ref)
    return p


def add_lag_features(p: pd.DataFrame, n_trail=C.N_TRAIL) -> pd.DataFrame:
    p = p.sort_values(["symbol", "weekday", "slot", "week"]).copy()
    for c in LOG_COLS:
        p[c] = np.log1p(p[c])
    base = [c for cols in FAMILY_RAW.values() for c in cols]
    g = p.groupby(["symbol", "weekday", "slot"], sort=False)
    new = {}
    for c in base:
        new[f"{c}__lag1"] = g[c].shift(1)
        new[f"{c}__trail{n_trail}"] = g[c].transform(lambda s: s.shift(1).rolling(n_trail, min_periods=1).mean())
    return pd.concat([p, pd.DataFrame(new, index=p.index)], axis=1)


def add_calendar_onehots(p: pd.DataFrame) -> pd.DataFrame:
    wd = pd.get_dummies(p["weekday"], prefix="cal_wd").astype(float)
    sl = pd.get_dummies(p["slot"], prefix="cal_slot").astype(float)
    return pd.concat([p, wd, sl], axis=1)


def scale_per_ticker(df: pd.DataFrame, cols, train_mask) -> pd.DataFrame:
    # Robust z-score using each ticker's own TRAIN-period median/IQR, so features mean
    # "unusual for THIS stock" (matches the per-stock labels and enables transfer, RQ4).
    g = df.loc[train_mask].groupby("symbol")[cols]
    med, q75, q25, std = g.median(), g.quantile(0.75), g.quantile(0.25), g.std()
    scale = (q75 - q25).replace(0, np.nan).fillna(std).replace(0, np.nan).fillna(1.0)
    m = med.reindex(df["symbol"]).to_numpy()
    s = scale.reindex(df["symbol"]).to_numpy()
    z = np.nan_to_num(np.clip((df[cols].to_numpy(float) - m) / s, -5, 5), nan=0.0)
    out = df.copy()
    out[cols] = z
    return out


def build_dataset(raw: pd.DataFrame, sector_of: dict, bench_of: dict, label_col=C.LABEL_COL):
    # raw: concatenated cached bucket summaries (stocks, benchmark ETFs, optionally crypto).
    bench_syms = set(bench_of.values())
    bench = (raw[raw["symbol"].isin(bench_syms)][["symbol", "date", "slot", "ret_bps"]]
             .rename(columns={"symbol": "bench", "ret_bps": "bench_ret_bps"}))
    p = raw[~raw["symbol"].isin(bench_syms)].copy()
    p["sector"] = p["symbol"].map(sector_of)
    p["bench"] = p["symbol"].map(bench_of)
    p = p.merge(bench, on=["bench", "date", "slot"], how="left")
    p["dev_bps"] = p["ret_bps"] - p["bench_ret_bps"]
    p["abs_dev_bps"] = p["dev_bps"].abs()

    p = add_calendar_index(p)
    p = add_labels(p, label_col)
    p = add_busyness(p)
    p = add_lag_features(p)
    p = add_calendar_onehots(p)

    fam = {name: [f"{c}__{k}" for c in cols for k in ("lag1", f"trail{C.N_TRAIL}")]
           for name, cols in FAMILY_RAW.items()}
    fam["busyness"] = ["rel_busy_max", "rel_busy_90th", "rel_busy_75th"]
    fam["calendar"] = [c for c in p.columns if c.startswith("cal_")]

    numeric = [c for k, v in fam.items() if k != "calendar" for c in v]
    p = scale_per_ticker(p, numeric, p["split"] == "train")

    # keep rows with a label and at least one prior week of history
    p = p[p["y"].notna() & p[f"{label_col}__lag1"].notna()].copy()
    p["y"] = p["y"].astype(int)
    fam["all"] = [c for k, v in fam.items() for c in v]
    keep = META + sorted(set(fam["all"]))
    return p[keep].reset_index(drop=True), fam


def assert_no_leakage(ds: pd.DataFrame):
    w = ds.groupby("split")["week"].agg(["min", "max"])
    assert w.loc["train", "max"] < w.loc["val", "min"] <= w.loc["val", "max"] < w.loc["test", "min"], w
    return w

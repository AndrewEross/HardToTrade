"""Offline smoke test with synthetic data (no Alpaca account needed):  python test_features.py"""
import numpy as np
import pandas as pd

import config as C
from features import (add_labels, add_lagged_features, bars_bucket_features,
                      bucket_of, spread_bucket_features)

rng = np.random.default_rng(0)
NY = "America/New_York"


def fake_bars(days, seed):
    r = np.random.default_rng(seed)
    frames = []
    for d in days:
        idx = pd.date_range(f"{d} 09:30", f"{d} 15:59", freq="1min", tz=NY).tz_convert("UTC")
        close = 100 * np.exp(np.cumsum(r.normal(0, 5e-4, len(idx))))
        frames.append(pd.DataFrame({"open": close, "high": close * 1.0005, "low": close * 0.9995,
                                    "close": close, "volume": r.integers(100, 5000, len(idx))}, index=idx))
    return pd.concat(frames)


days = pd.bdate_range("2026-04-01", "2026-06-30")
stock, etf = fake_bars(days, 1), fake_bars(days, 2)

# bucket edges
edges = pd.DatetimeIndex(["2026-04-01 09:29", "2026-04-01 09:30", "2026-04-01 10:29", "2026-04-01 10:30",
                          "2026-04-01 15:30", "2026-04-01 15:59", "2026-04-01 16:00"], tz=NY)
assert list(bucket_of(edges)) == [-1, 0, 0, 1, 6, 6, -1], bucket_of(edges)

bf = bars_bucket_features(stock, etf)
assert len(bf) == len(days) * C.N_BUCKETS, len(bf)
assert bf["reversal_rate"].between(0, 1).all()
assert bf["efficiency_ratio"].between(0, 1).all()
print("bars features OK:", bf.shape)

# synthetic quotes/trades for one day
day = "2026-04-01"
qi = pd.date_range(f"{day} 09:25", f"{day} 16:00", freq="2s", tz=NY).tz_convert("UTC")
mid = 50 + np.cumsum(rng.normal(0, 0.002, len(qi)))
half = 0.01
quotes = pd.DataFrame({"bid_price": mid - half, "ask_price": mid + half,
                       "bid_size": rng.integers(1, 10, len(qi)), "ask_size": rng.integers(1, 10, len(qi))}, index=qi)
ti = qi[::7] + pd.Timedelta(milliseconds=500)
trades = pd.DataFrame({"price": mid[::7] + half * rng.choice([-1, 1], len(ti)),   # trades at bid or ask
                       "size": rng.integers(100, 500, len(ti)),
                       "conditions": [["@"]] * len(ti)}, index=ti)
sb = spread_bucket_features(quotes, trades)
assert len(sb) == C.N_BUCKETS, len(sb)
expected_bps = 2 * half / 50 * 1e4                    # trading at bid/ask -> eff spread = quoted spread
got = sb["eff_spread_bps_mean"].mean()
assert abs(got - expected_bps) < 0.5, (got, expected_bps)
print(f"spread features OK: effective spread {got:.2f} bps (expected about {expected_bps:.2f})")

# labels + lags on a fake multi-ticker table
frames = []
for t in ["AAA", "BBB"]:
    f = bf.reset_index()
    f.insert(0, "ticker", t)
    f["eff_spread_bps_mean"] = rng.gamma(2, 2, len(f)) * (1 + (t == "BBB"))
    frames.append(f)
df = add_labels(pd.concat(frames, ignore_index=True))
train = df[df.split == "train"]
share = train.groupby("ticker")["label_name"].value_counts(normalize=True)
assert share.between(0.28, 0.40).all(), share
assert set(df.split.unique()) == {"train", "val", "test"}
df = add_lagged_features(df, ["abs_dev_mean", "eff_spread_bps_mean"])
first = df.sort_values("date").groupby(["ticker", "bucket"]).head(1)
assert first["eff_spread_bps_mean_lag1w"].isna().all()
print("labels and lags OK")
print("ALL TESTS PASSED")

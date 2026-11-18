# Central configuration. Edit here, not inside the notebooks.
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
for _p in (DATA_RAW, DATA_PROC, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)

# ---- Collection window (~125 trading days) ----
START = "2026-04-01"
END = "2026-09-30"

# "iex" works on the free Alpaca plan; use "sip" (consolidated tape) if your plan allows it.
# NOTE: with IEX, quotes/trades are IEX-only, so effective spread is measured against the
# IEX midpoint, not the NBBO. State this as a limitation in the report.
FEED = "iex"

# ---- Universe (proposal: 15-20 tech + 15-20 financial) ----
TECH = ["AAPL", "MSFT", "NVDA", "AMD", "INTC", "CSCO", "ORCL", "ADBE", "CRM", "QCOM",
        "TXN", "AVGO", "IBM", "MU", "AMAT", "INTU"]
FIN = ["JPM", "BAC", "WFC", "C", "GS", "MS", "SCHW", "BLK", "AXP", "USB",
       "PNC", "TFC", "COF", "BK", "MET", "PRU"]
# Optional stretch goal for RQ4 (24/7 market). BTC/USD is only the benchmark.
CRYPTO = ["ETH/USD", "SOL/USD", "LTC/USD", "AVAX/USD", "LINK/USD", "DOGE/USD"]

SECTORS = {
    "tech":   {"symbols": TECH,   "bench": "XLK",     "kind": "stock"},
    "fin":    {"symbols": FIN,    "bench": "XLF",     "kind": "stock"},
    "crypto": {"symbols": CRYPTO, "bench": "BTC/USD", "kind": "crypto"},
}

# ---- Bucketing ----
# Stocks: 9:30-16:00 ET -> six 1-hour slots (0-5) + one 30-minute slot (6).
# Crypto: 24 one-hour slots per UTC day.
# A (ticker, weekday, slot) triple is the "hour-of-week" bucket; one observation per week.

# ---- Label / data quality ----
LABEL_COL = "eff_vw"          # size-weighted effective spread (bps) per bucket
MIN_TRADES = 5                # drop buckets with fewer quote-matched trades
MAX_QUOTE_AGE_S = 5           # a trade is matched to the latest quote at most this old
MAX_QSPREAD_BPS = 500         # drop absurd quotes

# ---- Splits (chronological, by week) ----
TRAIN_FRAC, VAL_FRAC = 0.6, 0.2
N_TRAIL = 4                   # weeks in the trailing-mean features
SEED = 42

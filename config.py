"""Project settings. Edit values here, not in the scripts."""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
INTERIM_BARS = DATA / "interim" / "bars"        # raw 1-minute bars (small)
INTERIM_SPREADS = DATA / "interim" / "spreads"  # per ticker-day bucket aggregates
PROCESSED = DATA / "processed"                  # final modeling table

# --------------------------------------------------------- study window
# About six months (~125 trading days), per the proposal.
START_DATE = "2026-04-01"
END_DATE = "2026-09-30"

# ------------------------------------------------------------- universe
# Starting list. check_access.py reports any symbol that returns no data,
# so swap those out. Mixing mega-caps with smaller names gives more
# variation in difficulty across stocks.
TECH = ["AAPL", "MSFT", "NVDA", "AMD", "INTC", "CSCO", "ORCL", "ADBE",
        "QCOM", "TXN", "MU", "IBM", "AMAT", "PANW", "ANET", "FTNT"]
FINANCIALS = ["JPM", "BAC", "WFC", "C", "GS", "MS", "SCHW", "BLK",
              "AXP", "USB", "PNC", "TFC", "COF", "KEY", "RF", "CFG"]
SECTOR_ETF = {"tech": "XLK", "fin": "XLF"}
TICKER_SECTOR = {**{t: "tech" for t in TECH}, **{t: "fin" for t in FINANCIALS}}

# ------------------------------------------------------------ data feed
# "sip" = consolidated feed from all exchanges (what you want for spreads).
# "iex" = IEX only, a small share of volume. On the free Basic plan, real-time
# data is IEX only and historical queries exclude the latest 15 minutes.
# Run check_access.py to see what your account returns for each feed.
FEED = "sip"

# -------------------------------------------------------------- buckets
# Six 1-hour buckets (9:30-15:30) plus a final 30-minute bucket (15:30-16:00).
BUCKET_MINUTES = 60
N_BUCKETS = 7
BUCKET_LABELS = ["09:30-10:30", "10:30-11:30", "11:30-12:30", "12:30-13:30",
                 "13:30-14:30", "14:30-15:30", "15:30-16:00"]

# ------------------------------------------------------ cleaning rules
MAX_REL_SPREAD = 0.05        # drop quotes whose spread exceeds 5% of the midpoint
MAX_EFF_SPREAD_BPS = 500     # drop trades whose effective spread exceeds 500 bps
# Trade condition codes to exclude (auctions, out-of-sequence, averaged or
# derivatively priced prints, etc.). REVIEW against Alpaca's condition code
# table before trusting the numbers.
EXCLUDED_TRADE_CONDITIONS = {"B", "C", "G", "H", "M", "P", "Q", "T", "U",
                             "V", "W", "Z", "4", "7"}

# ------------------------------------------------- features and labels
BETA_WINDOW_DAYS = 20            # rolling window (prior days only) for stock-vs-ETF beta
LABEL_COLUMN = "eff_spread_bps_mean"   # alternatives: eff_spread_bps_vw, eff_spread_bps_median
SPLIT = (0.6, 0.2, 0.2)          # chronological train / validation / test

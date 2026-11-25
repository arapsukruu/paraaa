"""Central configuration for the Crypto Futures Quant Lab (Gate.io USDT-M Perpetuals)."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

# Atomic bar granularity in minutes (5m enables 32 days of full Gate.io perp+spot+index+OI+liq stats)
BAR_MINS = 5
BARS_PER_DAY = 1440 // BAR_MINS  # 288 bars/day

# 16 liquid USDT-M Perpetual pairs spanning distinct crypto sectors (Gate.io contract format: BASE_USDT -> internal BASEUSDT)
UNIVERSE = [
    # Majors
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    # L1 / Move / Infra
    "AVAXUSDT",
    "SUIUSDT",
    "APTUSDT",
    "LINKUSDT",
    "NEARUSDT",
    "ADAUSDT",
    # ETH DeFi & L2 (tight natural pairs)
    "AAVEUSDT",
    "UNIUSDT",
    "ARBUSDT",
    "OPUSDT",
    # High-Beta / Meme
    "DOGEUSDT",
]

SECTOR_MAP = {
    "BTCUSDT": "MAJOR",
    "ETHUSDT": "MAJOR",
    "SOLUSDT": "L1",
    "BNBUSDT": "MAJOR",
    "XRPUSDT": "LEGACY_L1",
    "AVAXUSDT": "L1",
    "SUIUSDT": "MOVE_L1",
    "APTUSDT": "MOVE_L1",
    "LINKUSDT": "INFRA",
    "NEARUSDT": "L1",
    "ADAUSDT": "LEGACY_L1",
    "AAVEUSDT": "DEFI",
    "UNIUSDT": "DEFI",
    "ARBUSDT": "ETH_L2",
    "OPUSDT": "ETH_L2",
    "DOGEUSDT": "MEME",
}

# Realistic Perpetual Futures Cost Scenarios (one-way decimal rates)
COST_SCENARIOS = {
    "zero_cost": {"fee_one_way": 0.0, "slippage_one_way": 0.0},
    "maker": {"fee_one_way": 0.0002, "slippage_one_way": 0.0001},   # 2 bps fee + 1 bp adverse selection = 6 bps RT
    "taker": {"fee_one_way": 0.0005, "slippage_one_way": 0.0002},   # 5 bps fee + 2 bps slippage = 14 bps RT
}

# Chronological Walk-Forward Split Fractions (strict no-leakage)
TRAIN_FRAC = 0.50   # Discovery / pair selection / ML train
VAL_FRAC = 0.25     # Strategy selection & probability threshold calibration
TEST_FRAC = 0.25    # Untouched Out-of-Sample Holdout

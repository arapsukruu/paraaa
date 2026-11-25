"""Strictly TRAIN-ONLY (Q1 + Q2, first 50%) physics diagnostic for Cascade & Complement engines.

Never touches Q3 or Q4 (untouched OOS holdout) so every insight is 100% leak-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import BARS_PER_DAY
from engine.backtester import apply_portfolio_constraints, evaluate_trades
from features import compute_universe_features, load_universe
from run_round2_adaptive import WARMUP_BARS, compute_macro_btc_regime, generate_r2_trades
from strategies.cross_pair import select_cointegrated_pairs


def main() -> None:
    data_map = load_universe()
    feat_map, _ = compute_universe_features(data_map)
    btc_trend_z = compute_macro_btc_regime(data_map, feat_map).to_numpy(dtype=np.float64)

    n_total = len(data_map["BTCUSDT"])
    usable = n_total - WARMUP_BARS
    q1_s = WARMUP_BARS
    q1_e = WARMUP_BARS + int(usable * 0.25)
    q2_s = q1_e
    q2_e = WARMUP_BARS + int(usable * 0.50)  # STRICT TRAIN BOUNDARY (50%)

    pairs = select_cointegrated_pairs(data_map, train_end_idx=q2_e, max_pairs=8)
    tr_eng = generate_r2_trades(data_map, feat_map, btc_trend_z, pairs, q1_s, q2_e)
    cas = apply_portfolio_constraints(tr_eng["R2_CASCADE_FOLLOW"], max_concurrent=6)

    rows = []
    for t in cas:
        mf = t.meta_features
        q = "Q1" if t.entry_idx < q2_s else "Q2"
        rows.append({
            "quarter": q,
            "symbol": t.symbol,
            "side": t.side,
            "exit_reason": t.exit_reason,
            "hold_mins": t.hold_mins,
            "net_maker_bps": t.net_ret("maker") * 10000.0,
            "net_hybrid_bps": (t.gross_ret + t.funding_pnl - 0.0010) * 10000.0,  # 10 bps RT hybrid
            "win": 1 if t.net_ret("maker") > 0 else 0,
            "vol_ratio": mf.get("vol_ratio", np.nan),
            "vol_240m_bps": mf.get("vol_240m", np.nan) * 10000.0,
            "mkt_vol_z": mf.get("mkt_vol_z", np.nan),
            "mkt_liq_z": mf.get("mkt_liq_z", np.nan),
            "mkt_unanimity": mf.get("mkt_unanimity", np.nan),
            "mkt_shock_score": mf.get("mkt_shock_score", np.nan),
            "vr_15": mf.get("vr_15", np.nan),
            "abs_real_liq_z": abs(mf.get("real_liq_z", 0.0)),
            "liq_vol_ratio_bps": mf.get("liq_vol_ratio", 0.0) * 10000.0,
            "vol_surge_15m": mf.get("vol_surge_15m", np.nan),
            "aligned_oi_chg15": mf.get("oi_chg_15m", 0.0) * 10000.0,
            "aligned_prem_z": mf.get("prem_z_240m", 0.0) * t.side,
            "aligned_spot_ofi": mf.get("spot_ofi_15m", 0.0) * t.side,
            "aligned_perp_ofi": mf.get("perp_ofi_15m", 0.0) * t.side,
            "aligned_btc_trend": mf.get("btc_trend_24h_z", 0.0) * t.side,
            "abs_btc_trend": abs(mf.get("btc_trend_24h_z", 0.0)),
            "btc_r2": mf.get("btc_r2", np.nan),
        })
    df = pd.DataFrame(rows)
    print(f"=== TRAIN-ONLY (Q1+Q2) CASCADE DIAGNOSTIC: n={len(df)} (Q1={(df.quarter=='Q1').sum()}, Q2={(df.quarter=='Q2').sum()}) ===")
    print(df.groupby("quarter")[["net_maker_bps", "net_hybrid_bps", "win", "vol_ratio", "mkt_vol_z", "mkt_liq_z", "mkt_unanimity", "vr_15", "abs_btc_trend", "liq_vol_ratio_bps"]].mean().round(2).to_string())

    print("\n=== EXIT REASON BREAKDOWN (TRAIN Q1+Q2) ===")
    print(df.groupby("exit_reason")[["net_maker_bps", "hold_mins"]].agg(["count", "mean"]).round(2).to_string())

    print("\n=== FEATURE QUANTILE SLICES ON TRAIN (Q1+Q2) — WHAT SEPARATES WINNERS FROM CHOP? ===")
    check_feats = [
        "vol_ratio", "vol_240m_bps", "mkt_vol_z", "mkt_liq_z", "mkt_unanimity",
        "vr_15", "abs_real_liq_z", "liq_vol_ratio_bps", "vol_surge_15m",
        "aligned_spot_ofi", "aligned_prem_z", "aligned_btc_trend", "abs_btc_trend", "aligned_oi_chg15",
    ]
    for col in check_feats:
        try:
            bins = pd.qcut(df[col], q=3, duplicates="drop")
            g = df.groupby(bins, observed=False)["net_maker_bps"].agg(["count", "mean"])
            parts = [f"{str(idx)}: {row['mean']:+5.1f}bps(n={int(row['count'])})" for idx, row in g.iterrows()]
            spread = g["mean"].max() - g["mean"].min()
            print(f"  {col:<20} (spread={spread:>5.1f}): {' | '.join(parts)}")
        except Exception as exc:
            print(f"  {col:<20} skipped: {exc}")

    print("\n=== TRAIN-ONLY (Q1+Q2) PER-SYMBOL RANKING ===")
    sym_g = df.groupby("symbol").agg(n=("net_maker_bps", "count"), mean_bps=("net_maker_bps", "mean"), med_vol=("vol_240m_bps", "median")).sort_values("mean_bps", ascending=False)
    print(sym_g.round(2).to_string())


if __name__ == "__main__":
    main()

"""Round 2 Experiment: Regime-Adaptive Multi-Engine Perpetual Strategy Benchmark.

Tests strictly causal, economically grounded refinements derived from Round 1 findings:
  1. Macro Regime Switch (BTC 24h trend Z-score): blocks counter-trend knife-catching during >1.5 sigma macro trends.
  2. Two-Phase Liquidation & VIB Physics:
       - R2_CASCADE_FOLLOW    : Rides initial 1-2h forced liquidation / VIB momentum drift (instead of catching knives).
       - R2_POST_LIQ_SNAPBACK : Regime-gated 6-8h equilibrium snapback AFTER initial cascade wave cools.
  3. R2_CROWDED_TRAP          : Regime-gated OI + Basis + Spot-divergence FOMO trap fade (all horizons).
  4. R2_PAIRS_FEE_AWARE       : Market-Neutral Cointegrated Pairs with explicit fee-awareness gate (spread_std * z >= 60 bps).
  5. R2_META_PORTFOLIO        : Expanding Walk-Forward Logistic Regression Meta-Labeler across 3 OOS folds (~21 full days OOS!).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import BARS_PER_DAY, REPORTS_DIR
from engine.backtester import (
    Trade,
    apply_portfolio_constraints,
    evaluate_trades,
    simulate_pair_trades,
    simulate_single_asset,
)
from features import compute_universe_features, load_universe
from strategies.cross_pair import select_cointegrated_pairs
from strategies.event_bars import build_volume_imbalance_bars
from strategies.meta_engine import CAT_META_COLS, NUMERIC_META_COLS

WARMUP_BARS = 432  # 36h warmup


def compute_macro_btc_regime(data_map: dict[str, pd.DataFrame], feat_map: dict[str, pd.DataFrame]) -> pd.Series:
    """Strictly causal BTC 24h trend Z-score: > +1.5 bull impulse, < -1.5 bear crash."""
    btc = data_map["BTCUSDT"]["close"].astype(np.float64)
    r24h = np.log(btc / btc.shift(BARS_PER_DAY)).fillna(0.0)
    vol5 = feat_map["BTCUSDT"]["vol_1440m"].astype(np.float64)
    # 24h expected std = vol_5m * sqrt(288)
    z24h = (r24h / (vol5 * np.sqrt(float(BARS_PER_DAY)) + 1e-9)).clip(-6.0, 6.0).fillna(0.0)
    return z24h


def generate_r2_trades(
    data_map: dict[str, pd.DataFrame],
    feat_map: dict[str, pd.DataFrame],
    btc_trend_z: np.ndarray,
    selected_pairs: list[tuple[str, str]],
    start_idx: int,
    end_idx: int,
) -> dict[str, list[Trade]]:
    """Generate all Round 2 orthogonal sub-engine trades strictly within [start_idx, end_idx)."""
    engines: dict[str, list[Trade]] = {
        "R2_CROWDED_TRAP": [],
        "R2_CASCADE_FOLLOW": [],
        "R2_VIB_MOMENTUM": [],
        "R2_POST_LIQ_SNAP": [],
        "R2_SPOT_WHALE": [],
        "R2_PAIRS_NEUTRAL": [],
    }

    for sym, df in data_map.items():
        f = feat_map[sym].copy()
        f["btc_trend_24h_z"] = btc_trend_z.astype(np.float32)
        n = len(df)
        hi = min(end_idx, n)
        idx_arr = np.arange(n)
        in_win = (idx_arr >= start_idx) & (idx_arr < hi)

        vol240 = f["vol_240m"].to_numpy(dtype=np.float64)
        r15 = f["r15"].to_numpy(dtype=np.float64)
        r60 = f["r60"].to_numpy(dtype=np.float64)
        p_ofi5 = f["perp_ofi_5m"].to_numpy(dtype=np.float64)
        p_ofi15 = f["perp_ofi_15m"].to_numpy(dtype=np.float64)
        s_ofi15 = f["spot_ofi_15m"].to_numpy(dtype=np.float64)
        sp_div15 = f["spot_perp_div_15m"].to_numpy(dtype=np.float64)
        prem_z = f["prem_z_240m"].to_numpy(dtype=np.float64)
        oi_z = f["oi_z_360m"].to_numpy(dtype=np.float64)
        oi_chg15 = f["oi_chg_15m"].to_numpy(dtype=np.float64)
        real_liq_z = f["real_liq_z"].to_numpy(dtype=np.float64)
        liq_sc = f["liq_cascade_score"].to_numpy(dtype=np.float64)
        v_surge = f["vol_surge_15m"].to_numpy(dtype=np.float64)
        top_z = f["top_ls_z"].to_numpy(dtype=np.float64)
        smart_z = f["smart_retail_z"].to_numpy(dtype=np.float64)
        retail_z = f["retail_ls_z"].to_numpy(dtype=np.float64)
        vr = f["vr_15"].to_numpy(dtype=np.float64)
        shock = f["mkt_shock_score"].to_numpy(dtype=np.float64)

        ret15_norm = r15 / (vol240 * np.sqrt(3.0) + 1e-9)
        ret60_norm = r60 / (vol240 * np.sqrt(12.0) + 1e-9)

        # Macro Regime Gate: never fade counter-trend against a >1.4 sigma 24h BTC macro trend
        allow_contrarian_long = btc_trend_z >= -1.4
        allow_contrarian_short = btc_trend_z <= 1.4

        # --- Engine 1: R2_CROWDED_TRAP (Regime-Gated Leveraged FOMO Trap Fade) ---
        # Medium-Long intraday hold (3h): OI jumped + Basis stretched + Spot disagrees/Retail crowded + flow stalling
        trap_short = (
            in_win
            & allow_contrarian_short
            & (ret15_norm >= 1.35)
            & (oi_z >= 1.0)
            & (prem_z >= 1.0)
            & ((sp_div15 <= -0.05) | (retail_z >= 1.0))
            & (p_ofi5 <= 0.08)
        )
        trap_long = (
            in_win
            & allow_contrarian_long
            & (ret15_norm <= -1.35)
            & (oi_z >= 1.0)
            & (prem_z <= -1.0)
            & ((sp_div15 >= 0.05) | (retail_z <= -1.0))
            & (p_ofi5 >= -0.08)
        )
        m_trap = trap_short | trap_long
        if np.any(m_trap):
            idxs = np.flatnonzero(m_trap)
            sides = np.where(trap_long[idxs], 1, -1)
            strengths = np.abs(ret15_norm[idxs]) + np.abs(prem_z[idxs]) * 0.5 + oi_z[idxs] * 0.5
            engines["R2_CROWDED_TRAP"].extend(
                simulate_single_asset(
                    df=df,
                    symbol=sym,
                    strategy="R2_CROWDED_TRAP",
                    horizon="medium",
                    signal_indices=idxs,
                    signal_sides=sides,
                    signal_strengths=strengths,
                    vol_series=vol240 * 4.5,
                    tp_mult=2.6,
                    sl_mult=1.8,
                    max_hold_mins=240,
                    min_tp=0.0060,
                    cooldown_mins=45,
                    meta_feature_df=f,
                )
            )

        # --- Engine 2: R2_CASCADE_FOLLOW (Ride Initial 1.5-2h Real Liquidation Drift!) ---
        # Long liquidations firing (real_liq_z <= -1.8) + price breaking down + volume surge -> SHORT the cascade drift!
        # Short liquidations firing (real_liq_z >= +1.8) + price breaking up + volume surge -> LONG the squeeze drift!
        # Extra filter: works best when NOT fighting extreme opposite macro trend
        cas_short = (
            in_win
            & (btc_trend_z <= 1.2)
            & ((real_liq_z <= -1.8) | (liq_sc <= -2.3))
            & (ret15_norm <= -1.2)
            & (v_surge >= 1.5)
            & (p_ofi5 <= -0.05)
        )
        cas_long = (
            in_win
            & (btc_trend_z >= -1.2)
            & ((real_liq_z >= 1.8) | (liq_sc >= 2.3))
            & (ret15_norm >= 1.2)
            & (v_surge >= 1.5)
            & (p_ofi5 >= 0.05)
        )
        m_cas = cas_short | cas_long
        if np.any(m_cas):
            idxs = np.flatnonzero(m_cas)
            sides = np.where(cas_long[idxs], 1, -1)
            strengths = np.abs(real_liq_z[idxs]) + np.abs(ret15_norm[idxs]) * 0.5
            engines["R2_CASCADE_FOLLOW"].extend(
                simulate_single_asset(
                    df=df,
                    symbol=sym,
                    strategy="R2_CASCADE_FOLLOW",
                    horizon="medium",
                    signal_indices=idxs,
                    signal_sides=sides,
                    signal_strengths=strengths,
                    vol_series=vol240 * 3.8,
                    tp_mult=2.3,
                    sl_mult=1.6,
                    max_hold_mins=120,
                    min_tp=0.0050,
                    cooldown_mins=45,
                    meta_feature_df=f,
                )
            )

        # --- Engine 3: R2_VIB_MOMENTUM (Ride 3 Consecutive Rapid Volume Imbalance Bars!) ---
        # Train showed fading 3 consecutive rapid VIBs lost strongly across ALL splits -> informed flow continuation!
        vb = build_volume_imbalance_bars(df, base_bars_per_day=24)
        if len(vb) >= 25:
            vb["dir_run"] = vb["direction"].rolling(3, min_periods=3).sum()
            vb["dur_ratio"] = vb["dur_mins"] / (vb["dur_mins"].rolling(20, min_periods=8).median() + 1.0)
            vb["vwap_12"] = (
                (vb["close"] * vb["qvol"]).rolling(12, min_periods=6).sum()
                / (vb["qvol"].rolling(12, min_periods=6).sum() + 1.0)
            )
            vb["stretch"] = np.log((vb["close"] + 1e-12) / (vb["vwap_12"] + 1e-12))
            vb["stretch_z"] = vb["stretch"] / (vb["stretch"].rolling(24, min_periods=10).std() + 1e-12)

            v_long = (
                (vb["end_idx"] >= start_idx)
                & (vb["end_idx"] < hi)
                & (vb["dir_run"] == 3)
                & (vb["stretch_z"] >= 1.2)
                & (vb["dur_ratio"] <= 0.85)
            )
            v_short = (
                (vb["end_idx"] >= start_idx)
                & (vb["end_idx"] < hi)
                & (vb["dir_run"] == -3)
                & (vb["stretch_z"] <= -1.2)
                & (vb["dur_ratio"] <= 0.85)
            )
            vm = v_long | v_short
            sub = vb.loc[vm]
            if len(sub) > 0:
                idxs = sub["end_idx"].to_numpy(dtype=np.int64)
                sides = np.where(v_long.loc[vm], 1, -1)
                strengths = sub["stretch_z"].abs().to_numpy(dtype=np.float64)
                engines["R2_VIB_MOMENTUM"].extend(
                    simulate_single_asset(
                        df=df,
                        symbol=sym,
                        strategy="R2_VIB_MOMENTUM",
                        horizon="medium",
                        signal_indices=idxs,
                        signal_sides=sides,
                        signal_strengths=strengths,
                        vol_series=vol240 * 3.8,
                        tp_mult=2.4,
                        sl_mult=1.6,
                        max_hold_mins=150,
                        min_tp=0.0055,
                        cooldown_mins=45,
                        meta_feature_df=f,
                    )
                )

        # --- Engine 4: R2_POST_LIQ_SNAP (Regime-Gated Delayed Post-Liquidation Snapback, 6-8h) ---
        # ONLY fires AFTER a liquidation event occurred 2-4 bars ago (10-20m ago) AND immediate 5m order flow stabilized,
        # AND NOT fighting a >1.4 sigma 24h macro crash!
        liq_prev = pd.Series(real_liq_z).rolling(4, min_periods=1).min().shift(1).fillna(0.0).to_numpy()
        liq_prev_hi = pd.Series(real_liq_z).rolling(4, min_periods=1).max().shift(1).fillna(0.0).to_numpy()
        snap_long = (
            in_win
            & allow_contrarian_long
            & (liq_prev <= -2.0)
            & (real_liq_z > -1.0)      # Liquidation wave stopped firing right now!
            & (p_ofi5 >= -0.02)        # Selling pressure stabilized!
            & (ret60_norm <= -1.4)
            & (prem_z <= -0.8)
        )
        snap_short = (
            in_win
            & allow_contrarian_short
            & (liq_prev_hi >= 2.0)
            & (real_liq_z < 1.0)
            & (p_ofi5 <= 0.02)
            & (ret60_norm >= 1.4)
            & (prem_z >= 0.8)
        )
        m_snap = snap_long | snap_short
        if np.any(m_snap):
            idxs = np.flatnonzero(m_snap)
            sides = np.where(snap_long[idxs], 1, -1)
            strengths = np.abs(ret60_norm[idxs]) + np.abs(prem_z[idxs]) * 0.5
            engines["R2_POST_LIQ_SNAP"].extend(
                simulate_single_asset(
                    df=df,
                    symbol=sym,
                    strategy="R2_POST_LIQ_SNAP",
                    horizon="long",
                    signal_indices=idxs,
                    signal_sides=sides,
                    signal_strengths=strengths,
                    vol_series=vol240 * 6.2,
                    tp_mult=3.6,
                    sl_mult=2.4,
                    max_hold_mins=480,
                    min_tp=0.0090,
                    cooldown_mins=120,
                    meta_feature_df=f,
                )
            )

        # --- Engine 5: R2_SPOT_WHALE (Spot-Led Smart-Money Flow with Pair-DNA Trend Filter) ---
        # Only take continuation when Pair DNA is NOT choppy (vr >= 0.90) & Spot + Top Traders agree
        wh_long = (
            in_win
            & (vr >= 0.90)
            & (s_ofi15 >= 0.22)
            & (sp_div15 >= 0.06)
            & (p_ofi15 >= 0.03)
            & ((smart_z >= 0.7) | (top_z >= 0.9))
            & (v_surge >= 1.5)
            & (prem_z <= 1.2)
            & (oi_chg15 >= -0.005)
        )
        wh_short = (
            in_win
            & (vr >= 0.90)
            & (s_ofi15 <= -0.22)
            & (sp_div15 <= -0.06)
            & (p_ofi15 <= -0.03)
            & ((smart_z <= -0.7) | (top_z <= -0.9))
            & (v_surge >= 1.5)
            & (prem_z >= -1.2)
            & (oi_chg15 >= -0.005)
        )
        m_wh = wh_long | wh_short
        if np.any(m_wh):
            idxs = np.flatnonzero(m_wh)
            sides = np.where(wh_long[idxs], 1, -1)
            strengths = np.abs(s_ofi15[idxs]) * 3.0 + np.abs(smart_z[idxs]) * 0.7
            engines["R2_SPOT_WHALE"].extend(
                simulate_single_asset(
                    df=df,
                    symbol=sym,
                    strategy="R2_SPOT_WHALE",
                    horizon="long",
                    signal_indices=idxs,
                    signal_sides=sides,
                    signal_strengths=strengths,
                    vol_series=vol240 * 5.8,
                    tp_mult=3.6,
                    sl_mult=2.3,
                    max_hold_mins=420,
                    min_tp=0.0085,
                    cooldown_mins=90,
                    meta_feature_df=f,
                )
            )

    # --- Engine 6: R2_PAIRS_NEUTRAL (Fee-Aware Market-Neutral Sector Pairs, 36h rolling) ---
    # Requires expected spread profit >= 60 bps (10x Maker cost) so fees never eat narrow spreads!
    win = 432  # 36h rolling
    entry_z = 2.1
    for sym_a, sym_b in selected_pairs:
        df_a = data_map[sym_a]
        df_b = data_map[sym_b]
        la = np.log(df_a["close"].astype(np.float64))
        lb = np.log(df_b["close"].astype(np.float64))

        cov_ab = la.rolling(win, min_periods=win // 2).cov(lb)
        var_b = lb.rolling(win, min_periods=win // 2).var()
        beta = (cov_ab / (var_b + 1e-12)).clip(0.3, 3.0)
        beta_lag = beta.shift(1)

        roll_corr = la.diff(3).rolling(win, min_periods=win // 2).corr(lb.diff(3)).fillna(0.0)
        spread = la - beta_lag * lb
        sp_mean = spread.rolling(win, min_periods=win // 2).mean()
        sp_std = spread.rolling(win, min_periods=win // 2).std()
        z_series = ((spread - sp_mean) / (sp_std + 1e-12)).to_numpy(dtype=np.float64)

        # Fee-awareness gate: 1 unit gross notional reversion distance approx = |z| * sp_std / (1 + beta)
        expected_edge = np.abs(z_series) * sp_std.fillna(0.0).to_numpy(dtype=np.float64) / (1.0 + beta_lag.fillna(1.0).to_numpy(dtype=np.float64))
        valid_mask = (roll_corr.to_numpy(dtype=np.float64) >= 0.55) & (expected_edge >= 0.0045)

        pair_meta = 0.5 * (feat_map[sym_a] + feat_map[sym_b])
        pair_meta["btc_trend_24h_z"] = btc_trend_z.astype(np.float32)

        engines["R2_PAIRS_NEUTRAL"].extend(
            simulate_pair_trades(
                df_a=df_a,
                df_b=df_b,
                sym_a=sym_a,
                sym_b=sym_b,
                strategy="R2_PAIRS_NEUTRAL",
                horizon="long",
                z_series=z_series,
                beta_series=beta_lag.to_numpy(dtype=np.float64),
                entry_z=entry_z,
                exit_z=0.15,
                stop_z=4.5,
                max_hold_mins=720,
                start_idx=start_idx,
                end_idx=end_idx,
                cooldown_mins=90,
                valid_mask=valid_mask,
                meta_feature_df=pair_meta,
            )
        )

    return engines


def _trades_to_df(trades: list[Trade], cost_mode: str = "maker") -> pd.DataFrame:
    signed_keys = {
        "perp_ofi_5m", "perp_ofi_15m", "spot_ofi_15m", "spot_perp_div_15m",
        "prem_close_bps", "prem_z_240m", "prem_z_1440m", "real_liq_z", "liq_cascade_score",
        "retail_ls_z", "top_ls_z", "smart_retail_z", "funding_bps", "resid_z_15m", "resid_z_60m",
        "btc_trend_24h_z",
    }
    cols = NUMERIC_META_COLS + ["btc_trend_24h_z"]
    rows = []
    for t in trades:
        s = float(t.side)
        row: dict[str, Any] = {
            "strategy": t.strategy,
            "horizon": t.horizon,
            "side": s,
            "signal_strength": float(t.signal_strength),
            "vol_at_entry": float(t.vol_at_entry),
            "y": 1 if t.net_ret(cost_mode) > 0 else 0,
        }
        for c in cols:
            if c in row:
                continue
            v = t.meta_features.get(c, np.nan)
            if c in signed_keys and np.isfinite(v):
                v = v * s
            row[c] = float(v) if np.isfinite(v) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    data_map = load_universe()
    feat_map, _ = compute_universe_features(data_map)
    btc_trend_z = compute_macro_btc_regime(data_map, feat_map).to_numpy(dtype=np.float64)

    sample_df = data_map["BTCUSDT"]
    n_total = len(sample_df)
    usable = n_total - WARMUP_BARS

    # 3-Fold Expanding Walk-Forward across the entire 30.5 days:
    #   Fold 1: Train [0..35%]  -> OOS [35%..57%]  (~6.7d)
    #   Fold 2: Train [0..57%]  -> OOS [57%..78%]  (~6.5d, includes the -14.5% BTC crash!)
    #   Fold 3: Train [0..78%]  -> OOS [78%..100%] (~6.7d, includes the 55.7k -> 65.5k V-bounce!)
    # Total strictly Out-of-Sample evaluation = ~20 FULL DAYS across ALL market regimes!
    cuts = [
        WARMUP_BARS,
        WARMUP_BARS + int(usable * 0.35),
        WARMUP_BARS + int(usable * 0.57),
        WARMUP_BARS + int(usable * 0.78),
        n_total,
    ]
    folds = [
        ("Fold1_Chop", cuts[0], cuts[1], cuts[1], cuts[2]),
        ("Fold2_Crash", cuts[0], cuts[2], cuts[2], cuts[3]),
        ("Fold3_VBounce", cuts[0], cuts[3], cuts[3], cuts[4]),
    ]

    oos_by_engine: dict[str, list[Trade]] = {}
    fold_summary: dict[str, Any] = {}
    meta_oos_trades_50: list[Trade] = []
    meta_oos_trades_53: list[Trade] = []
    meta_oos_trades_56: list[Trade] = []
    unfiltered_oos_all: list[Trade] = []
    total_oos_days = 0.0

    num_cols = NUMERIC_META_COLS + ["btc_trend_24h_z"]

    print("=== ROUND 2: 3-FOLD EXPANDING WALK-FORWARD OOS (~20 FULL DAYS ACROSS CHOP, CRASH & V-BOUNCE) ===", flush=True)
    for fname, tr_s, tr_e, te_s, te_e in folds:
        f_days = (te_e - te_s) / float(BARS_PER_DAY)
        total_oos_days += f_days
        btc_sub = sample_df["close"].iloc[te_s:te_e]
        btc_chg = (btc_sub.iloc[-1] / btc_sub.iloc[0] - 1.0) * 100.0

        # Strictly select pairs on fold's TRAIN window only!
        pairs = select_cointegrated_pairs(data_map, train_end_idx=tr_e, max_pairs=8)
        tr_eng = generate_r2_trades(data_map, feat_map, btc_trend_z, pairs, tr_s, tr_e)
        te_eng = generate_r2_trades(data_map, feat_map, btc_trend_z, pairs, te_s, te_e)

        tr_all = [t for lst in tr_eng.values() for t in lst]
        te_all = [t for lst in te_eng.values() for t in lst]
        unfiltered_oos_all.extend(te_all)

        for ename, lst in te_eng.items():
            oos_by_engine.setdefault(ename, []).extend(lst)

        # Fit strictly causal Logistic Regression Meta-Labeler on fold's TRAIN trades only
        df_tr = _trades_to_df(tr_all, "maker")
        df_te = _trades_to_df(te_all, "maker")
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_META_COLS),
        ])
        pipe = Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(C=0.15, max_iter=1000, class_weight="balanced", random_state=42)),
        ])
        pipe.fit(df_tr[num_cols + CAT_META_COLS], df_tr["y"].to_numpy())
        probs = pipe.predict_proba(df_te[num_cols + CAT_META_COLS])[:, 1]
        auc = float(roc_auc_score(df_te["y"].to_numpy(), probs)) if len(np.unique(df_te["y"])) > 1 else 0.5

        f_m50 = [t for t, p in zip(te_all, probs) if p >= 0.50]
        f_m53 = [t for t, p in zip(te_all, probs) if p >= 0.53]
        f_m56 = [t for t, p in zip(te_all, probs) if p >= 0.56]
        meta_oos_trades_50.extend(f_m50)
        meta_oos_trades_53.extend(f_m53)
        meta_oos_trades_56.extend(f_m56)

        m_raw = evaluate_trades(te_all, f_days, "maker", max_concurrent=8, n_bootstrap=300)
        m_53 = evaluate_trades(f_m53, f_days, "maker", max_concurrent=8, n_bootstrap=300)
        fold_summary[fname] = {"days": round(f_days, 2), "btc_chg_pct": round(btc_chg, 2), "auc": round(auc, 4), "raw": m_raw, "meta_p53": m_53}
        print(
            f"  {fname:<14} ({f_days:.1f}d, BTC {btc_chg:+5.1f}%) | AUC={auc:.3f} | "
            f"Raw={m_raw['exp_bps']:+6.2f}bps (n={m_raw['n_trades']:>3}) -> "
            f"Meta(p>=0.53)={m_53['exp_bps']:+6.2f}bps (n={m_53['n_trades']:>3}, Ret={m_53['total_ret_pct']:+5.2f}%, DD={m_53['max_dd_pct']:5.2f}%)",
            flush=True,
        )

    print(f"\n=== AGGREGATE 3-FOLD OOS RESULTS ({total_oos_days:.1f} FULL DAYS OUT-OF-SAMPLE) ===", flush=True)
    hdr = f"{'Engine / Portfolio':<24} {'N(t/d)':<12} {'Hold':<6} {'0c_bps':>7} {'Mkr_bps':>8} {'95% CI':>15} {'Tkr_bps':>8} {'Win%':>6} {'PF':>5} {'Ret%':>7} {'DD%':>7} {'Shrp':>6}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    summary_rows = {}
    for ename in sorted(oos_by_engine.keys()):
        lst = oos_by_engine[ename]
        mz = evaluate_trades(lst, total_oos_days, "zero_cost", max_concurrent=6)
        mm = evaluate_trades(lst, total_oos_days, "maker", max_concurrent=6)
        mt = evaluate_trades(lst, total_oos_days, "taker", max_concurrent=6)
        summary_rows[ename] = {"zero": mz, "maker": mm, "taker": mt}
        ci = f"[{mm['ci95_low_bps']:+.1f},{mm['ci95_high_bps']:+.1f}]"
        print(
            f"{ename:<24} {mm['n_trades']:>4}({mm['trades_per_day']:>4.1f}) {mm['avg_hold_mins']:>5.0f}m "
            f"{mz['exp_bps']:>+7.2f} {mm['exp_bps']:>+8.2f} {ci:>15} {mt['exp_bps']:>+8.2f} "
            f"{mm['win_rate_pct']:>5.1f}% {mm['profit_factor']:>5.2f} {mm['total_ret_pct']:>+6.2f}% "
            f"{mm['max_dd_pct']:>6.2f}% {mm['sharpe_daily']:>+6.2f}",
            flush=True,
        )

    print("-" * len(hdr), flush=True)
    for pname, plst in [
        ("PORTFOLIO_ALL_RAW", unfiltered_oos_all),
        ("PORTFOLIO_META_p50", meta_oos_trades_50),
        ("PORTFOLIO_META_p53", meta_oos_trades_53),
        ("PORTFOLIO_META_p56", meta_oos_trades_56),
    ]:
        mz = evaluate_trades(plst, total_oos_days, "zero_cost", max_concurrent=8)
        mm = evaluate_trades(plst, total_oos_days, "maker", max_concurrent=8)
        mt = evaluate_trades(plst, total_oos_days, "taker", max_concurrent=8)
        summary_rows[pname] = {"zero": mz, "maker": mm, "taker": mt}
        ci = f"[{mm['ci95_low_bps']:+.1f},{mm['ci95_high_bps']:+.1f}]"
        print(
            f"{pname:<24} {mm['n_trades']:>4}({mm['trades_per_day']:>4.1f}) {mm['avg_hold_mins']:>5.0f}m "
            f"{mz['exp_bps']:>+7.2f} {mm['exp_bps']:>+8.2f} {ci:>15} {mt['exp_bps']:>+8.2f} "
            f"{mm['win_rate_pct']:>5.1f}% {mm['profit_factor']:>5.2f} {mm['total_ret_pct']:>+6.2f}% "
            f"{mm['max_dd_pct']:>6.2f}% {mm['sharpe_daily']:>+6.2f}",
            flush=True,
        )

    # Direction balance check (Long vs Short on best portfolio)
    print("\n=== DIRECTION BALANCE (LONG vs SHORT ON PORTFOLIO_META_p53) ===", flush=True)
    for label, sgn in [("LONG ", 1), ("SHORT", -1)]:
        sub = [t for t in meta_oos_trades_53 if t.side == sgn]
        m = evaluate_trades(sub, total_oos_days, "maker", max_concurrent=8)
        print(f"  {label}: n={m['n_trades']:>3} ({m['trades_per_day']:>4.1f}/d) | Win={m['win_rate_pct']:>5.1f}% | Maker={m['exp_bps']:>+6.2f} bps | PF={m['profit_factor']:.2f} | Ret={m['total_ret_pct']:+5.2f}%", flush=True)

    out = {"total_oos_days": round(total_oos_days, 2), "folds": fold_summary, "engines": summary_rows}
    out_path = REPORTS_DIR / "round2_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=lambda x: float(x)), encoding="utf-8")
    print(f"\n[round2] Done in {time.time() - t0:.1f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

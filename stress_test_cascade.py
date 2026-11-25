"""Paranoid Audit & Stress Test for R2_CASCADE_FOLLOW (and top Round 2 engines).

Checks:
  1. Exact lag cross-correlation between candle return r5[t] and contract_stats (lsr_taker, net_liq) at t-2..t+2
     to prove 100% causal timestamp alignment (zero forward shift in Gate.io API!).
  2. Execution Latency Stress Test: entry at bar +1 (5m close), +2 (+5m extra delay), +3 (+10m delay), +4 (+15m delay).
  3. High-Slippage Cost Stress Test: 6 bps (maker), 14 bps (taker), 20 bps (volatile taker), 30 bps (extreme stress).
  4. Per-Fold stability (Train + Fold 1 + Fold 2 + Fold 3) & Per-Symbol breadth across all 16 coins.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import BARS_PER_DAY
from engine.backtester import Trade, apply_portfolio_constraints, evaluate_trades, simulate_single_asset
from features import compute_universe_features, load_universe
from run_round2_adaptive import WARMUP_BARS, compute_macro_btc_regime


def check_timestamp_alignment(data_map: dict[str, pd.DataFrame]) -> None:
    print("=== CHECK 1: GATE.IO TIMESTAMP ALIGNMENT (r5[t] vs stats[t + lag]) ===")
    print("  (If stats[t] covers bar [t, t+5m), peak correlation MUST be at lag=0, NOT lag=-1 or +1!)")
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        df = data_map[sym]
        r5 = np.log(df["close"] / df["open"]).fillna(0.0)
        ofi = df["perp_ofi_bar"].fillna(0.0)
        net_liq = (df["short_liq_usd"].fillna(0.0) - df["long_liq_usd"].fillna(0.0))
        net_liq = np.sign(net_liq) * np.log1p(np.abs(net_liq))
        ofi_corrs = [f"lag{k:+d}:{r5.corr(ofi.shift(k)):+.3f}" for k in [-2, -1, 0, 1, 2]]
        liq_corrs = [f"lag{k:+d}:{r5.corr(net_liq.shift(k)):+.3f}" for k in [-2, -1, 0, 1, 2]]
        print(f"  {sym:<8} OFI -> {' | '.join(ofi_corrs)}")
        print(f"  {sym:<8} LIQ -> {' | '.join(liq_corrs)}")
    print()


def run_cascade_with_delay(
    data_map: dict[str, pd.DataFrame],
    feat_map: dict[str, pd.DataFrame],
    btc_trend_z: np.ndarray,
    start_idx: int,
    end_idx: int,
    extra_delay_bars: int = 0,
) -> list[Trade]:
    trades: list[Trade] = []
    for sym, df in data_map.items():
        f = feat_map[sym]
        n = len(df)
        hi = min(end_idx, n)
        idx_arr = np.arange(n)
        in_win = (idx_arr >= start_idx) & (idx_arr < hi - extra_delay_bars - 2)

        vol240 = f["vol_240m"].to_numpy(dtype=np.float64)
        r15 = f["r15"].to_numpy(dtype=np.float64)
        p_ofi5 = f["perp_ofi_5m"].to_numpy(dtype=np.float64)
        real_liq_z = f["real_liq_z"].to_numpy(dtype=np.float64)
        liq_sc = f["liq_cascade_score"].to_numpy(dtype=np.float64)
        v_surge = f["vol_surge_15m"].to_numpy(dtype=np.float64)
        ret15_norm = r15 / (vol240 * np.sqrt(3.0) + 1e-9)

        cas_short = (
            in_win & (btc_trend_z <= 1.2)
            & ((real_liq_z <= -1.8) | (liq_sc <= -2.3))
            & (ret15_norm <= -1.2) & (v_surge >= 1.5) & (p_ofi5 <= -0.05)
        )
        cas_long = (
            in_win & (btc_trend_z >= -1.2)
            & ((real_liq_z >= 1.8) | (liq_sc >= 2.3))
            & (ret15_norm >= 1.2) & (v_surge >= 1.5) & (p_ofi5 >= 0.05)
        )
        m = cas_short | cas_long
        if np.any(m):
            idxs = np.flatnonzero(m) + extra_delay_bars  # shift entry forward by extra_delay_bars!
            sides = np.where(cas_long[np.flatnonzero(m)], 1, -1)
            strengths = np.abs(real_liq_z[np.flatnonzero(m)]) + np.abs(ret15_norm[np.flatnonzero(m)]) * 0.5
            # Use volatility from actual signal time (shifted array lookup uses entry-1 safely)
            trades.extend(
                simulate_single_asset(
                    df=df, symbol=sym, strategy="R2_CASCADE_FOLLOW", horizon="medium",
                    signal_indices=idxs, signal_sides=sides, signal_strengths=strengths,
                    vol_series=vol240 * 3.8, tp_mult=2.3, sl_mult=1.6,
                    max_hold_mins=120, min_tp=0.0050, cooldown_mins=45,
                )
            )
    return trades


def main() -> None:
    data_map = load_universe()
    check_timestamp_alignment(data_map)

    feat_map, _ = compute_universe_features(data_map)
    btc_trend_z = compute_macro_btc_regime(data_map, feat_map).to_numpy(dtype=np.float64)

    n_total = len(data_map["BTCUSDT"])
    usable = n_total - WARMUP_BARS
    oos_start = WARMUP_BARS + int(usable * 0.35)
    oos_days = (n_total - oos_start) / float(BARS_PER_DAY)

    print("=== CHECK 2: EXECUTION LATENCY STRESS TEST (20d OOS) ===")
    for delay in [0, 1, 2, 3, 6]:
        tr = run_cascade_with_delay(data_map, feat_map, btc_trend_z, oos_start, n_total, extra_delay_bars=delay)
        mz = evaluate_trades(tr, oos_days, "zero_cost", max_concurrent=6)
        mm = evaluate_trades(tr, oos_days, "maker", max_concurrent=6)
        mt = evaluate_trades(tr, oos_days, "taker", max_concurrent=6)
        print(
            f"  Delay +{delay*5:>2}m (bar+{delay+1}): n={mm['n_trades']:>3} | "
            f"0c={mz['exp_bps']:+6.2f} | Maker(6bps)={mm['exp_bps']:+6.2f} [{mm['ci95_low_bps']:+.1f},{mm['ci95_high_bps']:+.1f}] | "
            f"Taker(14bps)={mt['exp_bps']:+6.2f} | Win={mm['win_rate_pct']:4.1f}% | PF={mm['profit_factor']:.2f}"
        )

    print("\n=== CHECK 3: EXTREME SLIPPAGE STRESS TEST (DELAY +0m AND +5m) ===")
    for delay in [0, 1]:
        tr = apply_portfolio_constraints(
            run_cascade_with_delay(data_map, feat_map, btc_trend_z, oos_start, n_total, extra_delay_bars=delay),
            max_concurrent=6,
        )
        raw = np.array([t.gross_ret + t.funding_pnl for t in tr]) * 10000.0
        print(f"  [Delay +{delay*5}m, n={len(tr)}] Raw={raw.mean():+.2f} bps -> "
              f"at 6bps={raw.mean()-6:+.2f} | at 14bps={raw.mean()-14:+.2f} | "
              f"at 20bps={raw.mean()-20:+.2f} | at 30bps={raw.mean()-30:+.2f} bps")

    print("\n=== CHECK 4: FULL TIMELINE STABILITY (4 CHRONOLOGICAL QUARTERS, DELAY +0m vs +5m) ===")
    q_cuts = [WARMUP_BARS + int(usable * p) for p in [0.0, 0.25, 0.50, 0.75, 1.0]]
    btc_c = data_map["BTCUSDT"]["close"]
    for qi in range(4):
        s, e = q_cuts[qi], q_cuts[qi + 1]
        q_days = (e - s) / float(BARS_PER_DAY)
        b_ret = (btc_c.iloc[e - 1] / btc_c.iloc[s] - 1.0) * 100.0
        t0 = evaluate_trades(run_cascade_with_delay(data_map, feat_map, btc_trend_z, s, e, 0), q_days, "maker", 6, 300)
        t1 = evaluate_trades(run_cascade_with_delay(data_map, feat_map, btc_trend_z, s, e, 1), q_days, "maker", 6, 300)
        print(
            f"  Q{qi+1} ({q_days:.1f}d, BTC {b_ret:+5.1f}%): "
            f"Delay+0m={t0['exp_bps']:+6.2f}bps (n={t0['n_trades']:>3}, PF={t0['profit_factor']:.2f}) | "
            f"Delay+5m={t1['exp_bps']:+6.2f}bps (n={t1['n_trades']:>3}, PF={t1['profit_factor']:.2f})"
        )

    print("\n=== CHECK 5: PER-SYMBOL BREADTH ACROSS ALL 16 COINS (20d OOS, DELAY +0m & +5m) ===")
    tr0 = apply_portfolio_constraints(run_cascade_with_delay(data_map, feat_map, btc_trend_z, oos_start, n_total, 0), 6)
    tr1 = apply_portfolio_constraints(run_cascade_with_delay(data_map, feat_map, btc_trend_z, oos_start, n_total, 1), 6)
    pos0, pos1 = 0, 0
    for sym in sorted(data_map.keys()):
        s0 = [t.net_ret("maker") * 10000.0 for t in tr0 if t.symbol == sym]
        s1 = [t.net_ret("maker") * 10000.0 for t in tr1 if t.symbol == sym]
        m0 = float(np.mean(s0)) if s0 else 0.0
        m1 = float(np.mean(s1)) if s1 else 0.0
        if m0 > 0: pos0 += 1
        if m1 > 0: pos1 += 1
        print(f"  {sym:<10} | Delay+0m: n={len(s0):>3}, Mkr={m0:+6.2f} bps | Delay+5m: n={len(s1):>3}, Mkr={m1:+6.2f} bps")
    print(f"  Positive Symbols -> Delay+0m: {pos0}/16 | Delay+5m: {pos1}/16")


if __name__ == "__main__":
    main()

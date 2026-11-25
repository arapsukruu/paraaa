"""Brutal Anti-Illusion Audit for Round 2 Engines.

Separates TRUE structural alpha from bear-market drift illusion by testing every Round 2 engine against:
  1. Long-only vs Short-only performance breakdown
  2. Direction-Matched Random Null Benchmark (identical Long count, Short count, TP/SL & horizon on same OOS days!)
  3. BTC-Up-Day vs BTC-Down-Day regime slice analysis (does it survive when BTC rallies?)
  4. BTC-Direction-Adjusted Residual Alpha: net_ret - side * btc_ret_same_window
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from config import BARS_PER_DAY, REPORTS_DIR
from engine.backtester import Trade, apply_portfolio_constraints, evaluate_trades, simulate_single_asset
from features import compute_universe_features, load_universe
from run_round2_adaptive import WARMUP_BARS, compute_macro_btc_regime, generate_r2_trades
from strategies.cross_pair import select_cointegrated_pairs


def direction_matched_null(
    data_map: dict[str, pd.DataFrame],
    vol_map: dict[str, np.ndarray],
    start_idx: int,
    end_idx: int,
    n_long: int,
    n_short: int,
    tp_mult: float,
    sl_mult: float,
    max_hold: int,
    min_tp: float,
    n_trials: int = 40,
    seed: int = 99,
) -> tuple[float, float, float]:
    """Random entries with EXACT same Long/Short ratio and Triple-Barrier rules."""
    rng = np.random.default_rng(seed)
    syms = list(data_map.keys())
    tot = n_long + n_short
    if tot == 0:
        return 0.0, 0.0, 0.0
    p_long = n_long / float(tot)
    per_sym = max(1, int(np.ceil(tot / len(syms))))
    exps = []
    for _ in range(n_trials):
        t_all: list[Trade] = []
        for s in syms:
            df = data_map[s]
            hi = min(end_idx, len(df) - max_hold // 5 - 2)
            if hi <= start_idx + 10:
                continue
            idxs = np.sort(rng.integers(start_idx, hi, size=per_sym))
            sides = np.where(rng.random(per_sym) < p_long, 1, -1)
            t_all.extend(
                simulate_single_asset(
                    df, s, "NULL", "null", idxs, sides, np.ones(per_sym),
                    vol_map[s], tp_mult, sl_mult, max_hold, min_tp=min_tp, cooldown_mins=45,
                )
            )
        if t_all:
            sel = apply_portfolio_constraints(t_all, max_concurrent=6)
            exps.append(np.mean([t.net_ret("maker") for t in sel]) * 10000.0)
    return float(np.mean(exps)), float(np.percentile(exps, 95)), float(np.std(exps))


def main() -> None:
    data_map = load_universe()
    feat_map, _ = compute_universe_features(data_map)
    btc_trend_z = compute_macro_btc_regime(data_map, feat_map).to_numpy(dtype=np.float64)
    btc_df = data_map["BTCUSDT"]
    btc_open = btc_df["open"].to_numpy(dtype=np.float64)
    btc_close = btc_df["close"].to_numpy(dtype=np.float64)

    # BTC daily returns to split UP days vs DOWN days
    btc_daily = btc_df["close"].resample("1D").last().pct_change()
    up_dates = set(btc_daily[btc_daily > 0].index.date)
    down_dates = set(btc_daily[btc_daily <= 0].index.date)

    n_total = len(btc_df)
    usable = n_total - WARMUP_BARS
    oos_start = WARMUP_BARS + int(usable * 0.35)
    oos_days = (n_total - oos_start) / float(BARS_PER_DAY)

    cuts = [
        WARMUP_BARS,
        WARMUP_BARS + int(usable * 0.35),
        WARMUP_BARS + int(usable * 0.57),
        WARMUP_BARS + int(usable * 0.78),
        n_total,
    ]
    folds = [(cuts[0], cuts[1], cuts[1], cuts[2]), (cuts[0], cuts[2], cuts[2], cuts[3]), (cuts[0], cuts[3], cuts[3], cuts[4])]

    oos_eng: dict[str, list[Trade]] = {}
    for tr_s, tr_e, te_s, te_e in folds:
        pairs = select_cointegrated_pairs(data_map, train_end_idx=tr_e, max_pairs=8)
        eng = generate_r2_trades(data_map, feat_map, btc_trend_z, pairs, te_s, te_e)
        for k, v in eng.items():
            oos_eng.setdefault(k, []).extend(v)

    barrier_cfg = {
        "R2_CASCADE_FOLLOW": (3.8, 2.3, 1.6, 120, 0.0050),
        "R2_VIB_MOMENTUM": (3.8, 2.4, 1.6, 150, 0.0055),
        "R2_CROWDED_TRAP": (4.5, 2.6, 1.8, 240, 0.0060),
        "R2_SPOT_WHALE": (5.8, 3.6, 2.3, 420, 0.0085),
        "R2_POST_LIQ_SNAP": (6.2, 3.6, 2.4, 480, 0.0090),
    }

    print(f"=== BRUTAL ANTI-ILLUSION AUDIT ({oos_days:.1f}d OOS | BTC UP days: {len(up_dates)}, DOWN days: {len(down_dates)}) ===\n")
    audit_report = {}

    for ename in sorted(oos_eng.keys()):
        sel = apply_portfolio_constraints(oos_eng[ename], max_concurrent=6)
        if not sel:
            continue
        longs = [t for t in sel if t.side == 1]
        shorts = [t for t in sel if t.side == -1]
        m_all = evaluate_trades(oos_eng[ename], oos_days, "maker", max_concurrent=6, n_bootstrap=500)
        m_long = evaluate_trades(longs, oos_days, "maker", max_concurrent=6, n_bootstrap=300)
        m_short = evaluate_trades(shorts, oos_days, "maker", max_concurrent=6, n_bootstrap=300)

        # BTC-UP days vs BTC-DOWN days
        up_t = [t for t in sel if t.entry_ts.date() in up_dates]
        dn_t = [t for t in sel if t.entry_ts.date() in down_dates]
        up_bps = float(np.mean([t.net_ret("maker") for t in up_t]) * 10000.0) if up_t else 0.0
        dn_bps = float(np.mean([t.net_ret("maker") for t in dn_t]) * 10000.0) if dn_t else 0.0

        # Pure Excess Alpha over concurrent BTC move in the trade's direction
        if ename == "R2_PAIRS_NEUTRAL":
            excess_bps = m_all["exp_bps"]  # already market-neutral!
            null_mean, null_p95, z_vs_null = -6.0, 0.0, 0.0
        else:
            excess_arr = []
            for t in sel:
                btc_r = (btc_close[t.exit_idx] - btc_open[t.entry_idx]) / btc_open[t.entry_idx]
                excess_arr.append((t.net_ret("maker") - t.side * btc_r) * 10000.0)
            excess_bps = float(np.mean(excess_arr))

            vs, tpm, slm, mh, mtp = barrier_cfg[ename]
            vmap = {s: feat_map[s]["vol_240m"].to_numpy(dtype=np.float64) * vs for s in data_map}
            null_mean, null_p95, null_sd = direction_matched_null(
                data_map, vmap, oos_start, n_total, len(longs), len(shorts), tpm, slm, mh, mtp, n_trials=40
            )
            z_vs_null = (m_all["exp_bps"] - null_mean) / max(null_sd, 1e-6)

        alpha_over_null = m_all["exp_bps"] - null_mean
        beats_p95 = m_all["exp_bps"] > null_p95

        print(f"[{ename}] Total={m_all['n_trades']} (Long={len(longs)}, Short={len(shorts)}) | Maker={m_all['exp_bps']:+.2f} bps")
        print(f"  Long-only : n={len(longs):>3} | Maker={m_long['exp_bps']:+6.2f} bps (Win {m_long['win_rate_pct']:4.1f}%)")
        print(f"  Short-only: n={len(shorts):>3} | Maker={m_short['exp_bps']:+6.2f} bps (Win {m_short['win_rate_pct']:4.1f}%)")
        print(f"  BTC UP days (n={len(up_t):>3}): {up_bps:+6.2f} bps | BTC DOWN days (n={len(dn_t):>3}): {dn_bps:+6.2f} bps")
        print(
            f"  Dir-Matched Random Null = {null_mean:+.2f} bps (p95={null_p95:+.2f}) -> "
            f"TRUE ALPHA OVER NULL = {alpha_over_null:+.2f} bps (z={z_vs_null:+.2f}, beats p95={beats_p95}) | "
            f"BTC-Adjusted Excess = {excess_bps:+.2f} bps\n"
        )

        audit_report[ename] = {
            "n_total": m_all["n_trades"],
            "n_long": len(longs),
            "n_short": len(shorts),
            "maker_bps": m_all["exp_bps"],
            "long_bps": m_long["exp_bps"],
            "short_bps": m_short["exp_bps"],
            "btc_up_days_bps": round(up_bps, 2),
            "btc_down_days_bps": round(dn_bps, 2),
            "dir_matched_null_bps": round(null_mean, 2),
            "dir_matched_null_p95_bps": round(null_p95, 2),
            "true_alpha_over_null_bps": round(alpha_over_null, 2),
            "z_vs_null": round(z_vs_null, 2),
            "beats_null_p95": bool(beats_p95),
            "btc_adjusted_excess_bps": round(excess_bps, 2),
        }

    out_p = REPORTS_DIR / "round2_audit.json"
    out_p.write_text(json.dumps(audit_report, indent=2), encoding="utf-8")
    print(f"[audit] Saved -> {out_p}")


if __name__ == "__main__":
    main()

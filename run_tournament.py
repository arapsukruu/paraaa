"""Full 4-Method x 3-Horizon Walk-Forward Benchmark Tournament.

Runs strictly leak-free chronological evaluation:
  - Train (50%): Pair selection, ML Meta-Labeler training & preprocessing fit
  - Val   (25%): Strategy selection & probability threshold calibration
  - Test  (25%): Untouched Out-of-Sample Holdout + Bootstrap 95% CI + Random Null test

Saves:
  - quant_lab/reports/tournament_results.json
  - quant_lab/reports/tournament_report.html
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import BARS_PER_DAY, PROCESSED_DIR, REPORTS_DIR, TEST_FRAC, TRAIN_FRAC, VAL_FRAC
from engine.backtester import (
    Trade,
    apply_portfolio_constraints,
    evaluate_trades,
    random_entry_baseline,
)
from features import compute_universe_features, load_universe
from strategies.cross_pair import run_cross_pair_strategies, select_cointegrated_pairs
from strategies.event_bars import run_event_bar_strategies
from strategies.meta_engine import apply_regime_killswitch, run_meta_labeling_experiment
from strategies.microstructure import run_microstructure_strategies

WARMUP_BARS = 432  # 36 hours warmup (432 x 5m bars) for all 24h-36h rolling indicators


def run_all_base_strategies(
    data_map: dict[str, pd.DataFrame],
    feat_map: dict[str, pd.DataFrame],
    start_idx: int,
    end_idx: int,
    selected_pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], list[Trade]]:
    out: dict[tuple[str, str], list[Trade]] = {}
    out.update(run_event_bar_strategies(data_map, feat_map, start_idx, end_idx))
    out.update(run_microstructure_strategies(data_map, feat_map, start_idx, end_idx))
    out.update(run_cross_pair_strategies(data_map, feat_map, start_idx, end_idx, selected_pairs))
    return out


def build_svg_equity_curve(trades: list[Trade], cost_mode: str = "maker", max_concurrent: int = 6) -> str:
    """Generate a clean inline SVG equity curve for the HTML report."""
    sel = apply_portfolio_constraints(trades, max_concurrent=max_concurrent)
    if len(sel) < 2:
        return "<div class='muted'>Yeterli islem yok</div>"
    slot_w = 1.0 / float(max_concurrent)
    rets = [t.net_ret(cost_mode) * slot_w for t in sel]
    eq = [100.0]
    for r in rets:
        eq.append(eq[-1] * (1.0 + r))
    w, h, pad = 640, 180, 20
    mn, mx = min(eq), max(eq)
    span = max(mx - mn, 1e-6)
    pts = []
    for i, v in enumerate(eq):
        x = pad + (w - 2 * pad) * (i / (len(eq) - 1))
        y = h - pad - (h - 2 * pad) * ((v - mn) / span)
        pts.append(f"{x:.1f},{y:.1f}")
    y100 = h - pad - (h - 2 * pad) * ((100.0 - mn) / span)
    color = "#22c55e" if eq[-1] >= 100.0 else "#ef4444"
    return (
        f"<svg viewBox='0 0 {w} {h}' width='100%' height='{h}' style='background:#0f172a;border-radius:8px'>"
        f"<line x1='{pad}' y1='{y100:.1f}' x2='{w-pad}' y2='{y100:.1f}' stroke='#475569' stroke-dasharray='4,4'/>"
        f"<polyline fill='none' stroke='{color}' stroke-width='2.2' points='{' '.join(pts)}'/>"
        f"<text x='{pad+4}' y='16' fill='#94a3b8' font-size='11'>Min: {mn:.2f} | Max: {mx:.2f} | Son: {eq[-1]:.2f}</text>"
        f"</svg>"
    )


def write_html_report(results: dict[str, Any], ensemble_trades: list[Trade], meta_best_trades: list[Trade], out_path: Path) -> None:
    rows_html = []
    for row in results["base_table"]:
        exp_m = row["test_maker"]["exp_bps"]
        cls = "pos" if exp_m > 0 else "neg"
        rows_html.append(
            f"<tr>"
            f"<td><b>{row['strategy']}</b></td>"
            f"<td>{row['horizon']}</td>"
            f"<td>{row['test_maker']['n_trades']} ({row['test_maker']['trades_per_day']}/g)</td>"
            f"<td>{row['test_maker']['avg_hold_mins']:.0f} dk</td>"
            f"<td>{row['test_zero']['exp_bps']:+.2f}</td>"
            f"<td class='{cls}'><b>{exp_m:+.2f}</b> [{row['test_maker']['ci95_low_bps']:+.1f}, {row['test_maker']['ci95_high_bps']:+.1f}]</td>"
            f"<td>{row['test_taker']['exp_bps']:+.2f}</td>"
            f"<td>{row['test_maker']['win_rate_pct']:.1f}%</td>"
            f"<td>{row['test_maker']['profit_factor']:.2f}</td>"
            f"<td>{row['test_maker']['total_ret_pct']:+.2f}%</td>"
            f"<td>{row['test_maker']['max_dd_pct']:.2f}%</td>"
            f"<td>{row['test_maker']['sharpe_daily']:+.2f}</td>"
            f"<td>{row['ks_test_maker']['exp_bps']:+.2f}</td>"
            f"</tr>"
        )

    horizon_rows = []
    for h, hm in results["horizon_summary"].items():
        horizon_rows.append(
            f"<tr><td><b>{h.upper()}</b></td>"
            f"<td>{hm['avg_zero_bps']:+.2f} bps</td>"
            f"<td>{hm['avg_maker_bps']:+.2f} bps</td>"
            f"<td>{hm['avg_taker_bps']:+.2f} bps</td>"
            f"<td>{hm['null_maker_bps']:+.2f} bps (p95: {hm['null_p95_bps']:+.2f})</td>"
            f"<td>{hm['avg_hold_mins']:.1f} dk</td>"
            f"<td>{hm['n_positive_maker']}/{hm['n_strats']}</td></tr>"
        )

    meta_block = ""
    meta = results.get("meta_all", {})
    if "models" in meta:
        m_rows = []
        uf = meta["unfiltered_test"]
        ks = meta["killswitch_test"]
        m_rows.append(
            f"<tr><td>Ham Havuz (Filtresiz)</td><td>-</td><td>{uf['n_trades']} ({uf['trades_per_day']}/g)</td>"
            f"<td>{uf['win_rate_pct']:.1f}%</td><td>{uf['exp_bps']:+.2f}</td><td>{uf['profit_factor']:.2f}</td>"
            f"<td>{uf['total_ret_pct']:+.2f}%</td><td>{uf['max_dd_pct']:.2f}%</td><td>{uf['sharpe_daily']:+.2f}</td></tr>"
        )
        m_rows.append(
            f"<tr><td>Sadece Rejim + Sok Kill-Switch (D1)</td><td>-</td><td>{ks['n_trades']} ({ks['trades_per_day']}/g)</td>"
            f"<td>{ks['win_rate_pct']:.1f}%</td><td>{ks['exp_bps']:+.2f}</td><td>{ks['profit_factor']:.2f}</td>"
            f"<td>{ks['total_ret_pct']:+.2f}%</td><td>{ks['max_dd_pct']:.2f}%</td><td>{ks['sharpe_daily']:+.2f}</td></tr>"
        )
        for mname, md in meta["models"].items():
            mm = md["test_metrics_maker"]
            m_rows.append(
                f"<tr><td><b>{mname}</b> (AUC={md['test_roc_auc']:.3f})</td><td>p>={md['calibrated_threshold']:.2f}</td>"
                f"<td>{mm['n_trades']} ({mm['trades_per_day']}/g)</td>"
                f"<td>{mm['win_rate_pct']:.1f}%</td><td><b>{mm['exp_bps']:+.2f}</b></td><td>{mm['profit_factor']:.2f}</td>"
                f"<td>{mm['total_ret_pct']:+.2f}%</td><td>{mm['max_dd_pct']:.2f}%</td><td>{mm['sharpe_daily']:+.2f}</td></tr>"
            )
            mk = md["test_metrics_plus_killswitch_maker"]
            m_rows.append(
                f"<tr><td>{mname} + Kill-Switch</td><td>p>={md['calibrated_threshold']:.2f}</td>"
                f"<td>{mk['n_trades']} ({mk['trades_per_day']}/g)</td>"
                f"<td>{mk['win_rate_pct']:.1f}%</td><td>{mk['exp_bps']:+.2f}</td><td>{mk['profit_factor']:.2f}</td>"
                f"<td>{mk['total_ret_pct']:+.2f}%</td><td>{mk['max_dd_pct']:.2f}%</td><td>{mk['sharpe_daily']:+.2f}</td></tr>"
            )
        meta_block = f"""
        <h2>Metod D: Rejim Kill-Switch & Walk-Forward ML Meta-Labeling (Untouched Test)</h2>
        <table>
          <thead><tr><th>Model / Filtre</th><th>Esik</th><th>Islem</th><th>Win%</th><th>Beklenti (Maker bps)</th><th>PF</th><th>Getiri</th><th>MaxDD</th><th>Sharpe</th></tr></thead>
          <tbody>{''.join(m_rows)}</tbody>
        </table>
        """

    ens_svg = build_svg_equity_curve(ensemble_trades, "maker")
    meta_svg = build_svg_equity_curve(meta_best_trades, "maker")

    html = f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><title>Quant Lab Turnuva Raporu</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background:#020617; color:#e2e8f0; margin:24px; line-height:1.45; }}
  h1, h2 {{ color:#f8fafc; margin-top:28px; }}
  .meta {{ color:#94a3b8; font-size:13px; margin-bottom:16px; }}
  table {{ width:100%; border-collapse:collapse; font-size:12.5px; margin:12px 0 24px; background:#0f172a; border-radius:8px; overflow:hidden; }}
  th, td {{ padding:8px 10px; border-bottom:1px solid #1e293b; text-align:right; }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align:left; }}
  th {{ background:#1e293b; color:#cbd5e1; font-weight:600; }}
  .pos {{ color:#4ade80; }} .neg {{ color:#f87171; }} .muted {{ color:#64748b; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
  .card {{ background:#0f172a; padding:16px; border-radius:10px; border:1px solid #1e293b; }}
</style></head><body>
<h1>Kripto Vadeli Quant Lab — 4 Metod x 3 Vade Walk-Forward Turnuvasi</h1>
<div class="meta">
  Evren: {len(results['symbols'])} parite | Donem: {results['date_range']['start']} -> {results['date_range']['end']} |
  Train: {results['splits']['train_days']:.1f}g | Val: {results['splits']['val_days']:.1f}g |
  <b>Test (OOS): {results['splits']['test_days']:.1f}g</b>
</div>

<h2>Vade Karsilastirmasi (Kisa vs Orta vs Uzun Gun Ici — Test OOS)</h2>
<table>
  <thead><tr><th>Vade</th><th>Ort. Ham Alpha (0 Cost)</th><th>Ort. Maker (6 bps RT)</th><th>Ort. Taker (14 bps RT)</th><th>Rastgele Giris Null (Maker)</th><th>Ort. Sure</th><th>Pozitif Strateji</th></tr></thead>
  <tbody>{''.join(horizon_rows)}</tbody>
</table>

<h2>Tum Stratejiler x Tum Vadeler (Untouched Out-of-Sample Test)</h2>
<table>
  <thead>
    <tr>
      <th>Strateji</th><th>Vade</th><th>Islem (Gunluk)</th><th>Ort. Sure</th>
      <th>Ham (0c bps)</th><th>Maker bps [95% CI]</th><th>Taker bps</th>
      <th>Win%</th><th>PF</th><th>Getiri</th><th>MaxDD</th><th>Sharpe</th><th>+KillSwitch bps</th>
    </tr>
  </thead>
  <tbody>{''.join(rows_html)}</tbody>
</table>

{meta_block}

<div class="grid">
  <div class="card"><h3>Val-Secilmis Top Ensemble (Test OOS Equity - Maker)</h3>{ens_svg}</div>
  <div class="card"><h3>En Iyi ML Meta-Labeler (Test OOS Equity - Maker)</h3>{meta_svg}</div>
</div>
</body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    t0 = time.time()
    print("[tournament] Loading aligned 1m microstructure universe...", flush=True)
    data_map = load_universe()
    symbols = sorted(data_map.keys())
    sample_df = data_map[symbols[0]]
    n_total = len(sample_df)

    # Data quality / missingness audit (ML best practice)
    spot_miss = float(np.mean([df["spot_close"].isna().mean() for df in data_map.values()]) * 100.0)
    oi_miss = float(np.mean([df["oi_usdt"].isna().mean() for df in data_map.values()]) * 100.0)
    print(
        f"[tournament] Loaded {len(symbols)} symbols x {n_total:,} bars "
        f"({sample_df.index[0]} -> {sample_df.index[-1]}) | Missingness: Spot={spot_miss:.2f}% OI={oi_miss:.2f}%",
        flush=True,
    )

    print("[tournament] Computing strictly causal microstructure, derivatives & Pair-DNA features...", flush=True)
    feat_map, mkt_frame = compute_universe_features(data_map)

    usable = n_total - WARMUP_BARS
    train_start = WARMUP_BARS
    train_end = WARMUP_BARS + int(usable * TRAIN_FRAC)
    val_start = train_end
    val_end = WARMUP_BARS + int(usable * (TRAIN_FRAC + VAL_FRAC))
    test_start = val_end
    test_end = n_total

    train_days = (train_end - train_start) / float(BARS_PER_DAY)
    val_days = (val_end - val_start) / float(BARS_PER_DAY)
    test_days = (test_end - test_start) / float(BARS_PER_DAY)

    print(
        f"[tournament] Splits -> Train: {train_days:.1f}d | Val: {val_days:.1f}d | Test (OOS): {test_days:.1f}d",
        flush=True,
    )

    # Select cointegrated pairs strictly on TRAIN split only
    selected_pairs = select_cointegrated_pairs(data_map, train_end_idx=train_end, max_pairs=8)
    print(f"[tournament] Train-selected cointegrated pairs: {selected_pairs}", flush=True)

    print("[tournament] Running Methods A, B, C across Train / Val / Test...", flush=True)
    tr_map = run_all_base_strategies(data_map, feat_map, train_start, train_end, selected_pairs)
    va_map = run_all_base_strategies(data_map, feat_map, val_start, val_end, selected_pairs)
    te_map = run_all_base_strategies(data_map, feat_map, test_start, test_end, selected_pairs)

    # Random Entry Null Baseline on Test for each horizon (5m-native vol_240m scale)
    vol_map_short = {s: feat_map[s]["vol_240m"].to_numpy(dtype=np.float64) * 2.2 for s in symbols}
    vol_map_med = {s: feat_map[s]["vol_240m"].to_numpy(dtype=np.float64) * 3.8 for s in symbols}
    vol_map_long = {s: feat_map[s]["vol_240m"].to_numpy(dtype=np.float64) * 6.2 for s in symbols}

    null_baselines = {
        "short": random_entry_baseline(data_map, vol_map_short, test_start, test_end, 150, 1.6, 1.2, 35, "maker", 30),
        "medium": random_entry_baseline(data_map, vol_map_med, test_start, test_end, 120, 2.4, 1.7, 150, "maker", 30),
        "long": random_entry_baseline(data_map, vol_map_long, test_start, test_end, 80, 3.8, 2.4, 480, "maker", 30),
    }

    base_table = []
    val_selected_keys = []
    print("\n=== UNTOUCHED OUT-OF-SAMPLE TEST RESULTS (ALL METHODS x HORIZONS) ===", flush=True)
    header = (
        f"{'Strategy':<20} {'Hor':<6} {'N(t/d)':<11} {'Hold':<6} "
        f"{'0c_bps':>7} {'Mkr_bps':>8} {'95% CI':>15} {'Tkr_bps':>8} "
        f"{'Win%':>6} {'PF':>5} {'Ret%':>7} {'DD%':>7} {'Shrp':>6} {'KS_mkr':>7}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for key in sorted(te_map.keys()):
        strat, horizon = key
        tr_m = evaluate_trades(tr_map[key], train_days, "maker", n_bootstrap=200)
        va_m = evaluate_trades(va_map[key], val_days, "maker", n_bootstrap=300)
        te_z = evaluate_trades(te_map[key], test_days, "zero_cost", n_bootstrap=500)
        te_m = evaluate_trades(te_map[key], test_days, "maker", n_bootstrap=1000)
        te_t = evaluate_trades(te_map[key], test_days, "taker", n_bootstrap=500)
        ks_m = evaluate_trades(apply_regime_killswitch(te_map[key]), test_days, "maker", n_bootstrap=500)

        # Validation-only selection rule (strictly NO peeking at Test!)
        if va_m["n_trades"] >= 10 and va_m["exp_bps"] > 0 and tr_m["exp_bps"] > -3.0:
            val_selected_keys.append((key, va_m["exp_bps"], va_m["sharpe_daily"]))

        base_table.append({
            "strategy": strat,
            "horizon": horizon,
            "train_maker": tr_m,
            "val_maker": va_m,
            "test_zero": te_z,
            "test_maker": te_m,
            "test_taker": te_t,
            "ks_test_maker": ks_m,
        })

        ci_str = f"[{te_m['ci95_low_bps']:+.1f},{te_m['ci95_high_bps']:+.1f}]"
        print(
            f"{strat:<20} {horizon:<6} {te_m['n_trades']:>3}({te_m['trades_per_day']:>4.1f}) "
            f"{te_m['avg_hold_mins']:>5.0f}m {te_z['exp_bps']:>+7.2f} {te_m['exp_bps']:>+8.2f} "
            f"{ci_str:>15} {te_t['exp_bps']:>+8.2f} {te_m['win_rate_pct']:>5.1f}% "
            f"{te_m['profit_factor']:>5.2f} {te_m['total_ret_pct']:>+6.2f}% {te_m['max_dd_pct']:>6.2f}% "
            f"{te_m['sharpe_daily']:>+6.2f} {ks_m['exp_bps']:>+7.2f}",
            flush=True,
        )

    # Horizon comparison summary
    horizon_summary = {}
    print("\n=== HORIZON COMPARISON (SHORT vs MEDIUM vs LONG INTRADAY - TEST OOS) ===", flush=True)
    for h in ["short", "medium", "long"]:
        sub = [r for r in base_table if r["horizon"] == h and r["test_maker"]["n_trades"] >= 5]
        avg_z = float(np.mean([r["test_zero"]["exp_bps"] for r in sub])) if sub else 0.0
        avg_m = float(np.mean([r["test_maker"]["exp_bps"] for r in sub])) if sub else 0.0
        avg_t = float(np.mean([r["test_taker"]["exp_bps"] for r in sub])) if sub else 0.0
        avg_h = float(np.mean([r["test_maker"]["avg_hold_mins"] for r in sub])) if sub else 0.0
        n_pos = sum(1 for r in sub if r["test_maker"]["exp_bps"] > 0)
        horizon_summary[h] = {
            "n_strats": len(sub),
            "n_positive_maker": n_pos,
            "avg_zero_bps": round(avg_z, 2),
            "avg_maker_bps": round(avg_m, 2),
            "avg_taker_bps": round(avg_t, 2),
            "avg_hold_mins": round(avg_h, 1),
            "null_maker_bps": null_baselines[h]["null_mean_bps"],
            "null_p95_bps": null_baselines[h]["null_p95_bps"],
        }
        print(
            f"  {h.upper():<7}: 0-cost={avg_z:+.2f} bps | Maker={avg_m:+.2f} bps | Taker={avg_t:+.2f} bps | "
            f"Null(Maker)={null_baselines[h]['null_mean_bps']:+.2f} (p95={null_baselines[h]['null_p95_bps']:+.2f}) | "
            f"AvgHold={avg_h:.1f}m | Pos={n_pos}/{len(sub)}",
            flush=True,
        )

    # Validation-selected Ensemble evaluated strictly on untouched Test
    val_selected_keys.sort(key=lambda x: x[1], reverse=True)
    top_keys = [k[0] for k in val_selected_keys[:6]]
    ens_test_trades: list[Trade] = []
    for k in top_keys:
        ens_test_trades.extend(te_map[k])
    ens_test_ks = apply_regime_killswitch(ens_test_trades)
    ens_metrics = {
        "selected_on_val": [f"{k[0]}:{k[1]}" for k in top_keys],
        "test_maker": evaluate_trades(ens_test_trades, test_days, "maker", max_concurrent=8),
        "test_taker": evaluate_trades(ens_test_trades, test_days, "taker", max_concurrent=8),
        "test_zero": evaluate_trades(ens_test_trades, test_days, "zero_cost", max_concurrent=8),
        "test_ks_maker": evaluate_trades(ens_test_ks, test_days, "maker", max_concurrent=8),
    }
    print(f"\n=== VALIDATION-SELECTED ENSEMBLE (STRICT OOS TEST) ===", flush=True)
    print(f"  Selected strictly on Val: {ens_metrics['selected_on_val']}", flush=True)
    print(f"  Ensemble Maker : {ens_metrics['test_maker']}", flush=True)
    print(f"  Ensemble +KS   : {ens_metrics['test_ks_maker']}", flush=True)
    print(f"  Ensemble Taker : {ens_metrics['test_taker']}", flush=True)

    # Method D: Walk-Forward ML Meta-Labeling across ALL candidates AND per horizon
    all_tr = [t for lst in tr_map.values() for t in lst]
    all_va = [t for lst in va_map.values() for t in lst]
    all_te = [t for lst in te_map.values() for t in lst]

    # Method-Family Scoreboard on Untouched Test OOS (Maker)
    family_summary = {}
    print("\n=== METHOD FAMILY COMPARISON (A vs B vs C - UNTOUCHED TEST OOS) ===", flush=True)
    for fam_prefix, fam_label in [
        ("A_", "Method A: Event/Dollar/Imbalance Bars"),
        ("B_", "Method B: Derivatives & Microstructure"),
        ("C_", "Method C: Cross-Pair Beta/Pairs/LeadLag"),
    ]:
        fam_trades = [t for (st, hor), lst in te_map.items() if st.startswith(fam_prefix) for t in lst]
        fam_ks = apply_regime_killswitch(fam_trades)
        fm_z = evaluate_trades(fam_trades, test_days, "zero_cost", max_concurrent=8)
        fm_m = evaluate_trades(fam_trades, test_days, "maker", max_concurrent=8)
        fm_t = evaluate_trades(fam_trades, test_days, "taker", max_concurrent=8)
        fm_ks = evaluate_trades(fam_ks, test_days, "maker", max_concurrent=8)
        family_summary[fam_label] = {
            "test_zero": fm_z,
            "test_maker": fm_m,
            "test_taker": fm_t,
            "test_ks_maker": fm_ks,
        }
        print(
            f"  {fam_label:<40}: n={fm_m['n_trades']:>3} ({fm_m['trades_per_day']:>4.1f}/d) | "
            f"0c={fm_z['exp_bps']:>+6.2f} | Maker={fm_m['exp_bps']:>+6.2f} [{fm_m['ci95_low_bps']:+.1f},{fm_m['ci95_high_bps']:+.1f}] | "
            f"Taker={fm_t['exp_bps']:>+6.2f} | +KS={fm_ks['exp_bps']:>+6.2f} | Shrp={fm_m['sharpe_daily']:>+5.2f}",
            flush=True,
        )

    # Full-Sample (Val + Test = 50% OOS, ~15 days) Diagnostic for Base Rule-Based Strategies
    # (Pair selection & parameters used ZERO Val/Test data, so Val+Test is 100% OOS for A, B, C!)
    oos50_days = val_days + test_days
    oos50_table = []
    print(f"\n=== EXPANDED 50% OOS ({oos50_days:.1f} DAYS: VAL+TEST) FOR ZERO-FIT STRATEGIES (A/B/C) ===", flush=True)
    for key in sorted(te_map.keys()):
        strat, horizon = key
        comb = va_map[key] + te_map[key]
        m_z = evaluate_trades(comb, oos50_days, "zero_cost", n_bootstrap=500)
        m_m = evaluate_trades(comb, oos50_days, "maker", n_bootstrap=1000)
        m_t = evaluate_trades(comb, oos50_days, "taker", n_bootstrap=500)
        m_ks = evaluate_trades(apply_regime_killswitch(comb), oos50_days, "maker", n_bootstrap=500)
        oos50_table.append({
            "strategy": strat,
            "horizon": horizon,
            "oos50_zero": m_z,
            "oos50_maker": m_m,
            "oos50_taker": m_t,
            "oos50_ks_maker": m_ks,
        })
        ci_str = f"[{m_m['ci95_low_bps']:+.1f},{m_m['ci95_high_bps']:+.1f}]"
        print(
            f"  {strat:<20} {horizon:<6} n={m_m['n_trades']:>4}({m_m['trades_per_day']:>4.1f}/d) "
            f"0c={m_z['exp_bps']:>+6.2f} Mkr={m_m['exp_bps']:>+6.2f} {ci_str:>15} "
            f"Tkr={m_t['exp_bps']:>+6.2f} PF={m_m['profit_factor']:>4.2f} Shrp={m_m['sharpe_daily']:>+5.2f} KS={m_ks['exp_bps']:>+6.2f}",
            flush=True,
        )

    print("\n=== METHOD D: PAIR-DNA + KILL-SWITCH + WALK-FORWARD ML META-LABELING ===", flush=True)
    meta_all, meta_selected_map = run_meta_labeling_experiment(
        all_tr, all_va, all_te, val_days, test_days, cost_mode="maker", return_selected=True
    )
    if "models" in meta_all:
        print(f"  Unfiltered Pool (Test Maker): {meta_all['unfiltered_test']}", flush=True)
        print(f"  Kill-Switch D1  (Test Maker): {meta_all['killswitch_test']}", flush=True)
        for mname, md in meta_all["models"].items():
            print(
                f"  {mname:<14} (ValAUC={md['val_roc_auc']:.3f}, TestAUC={md['test_roc_auc']:.3f}, "
                f"th={md['calibrated_threshold']:.2f}): {md['test_metrics_maker']}",
                flush=True,
            )
            print(f"    Top features: {md['top_features'][:8]}", flush=True)

    meta_by_horizon = {}
    for h in ["short", "medium", "long"]:
        h_tr = [t for (st, hor), lst in tr_map.items() if hor == h for t in lst]
        h_va = [t for (st, hor), lst in va_map.items() if hor == h for t in lst]
        h_te = [t for (st, hor), lst in te_map.items() if hor == h for t in lst]
        meta_by_horizon[h] = run_meta_labeling_experiment(h_tr, h_va, h_te, val_days, test_days, cost_mode="maker")
        if "models" in meta_by_horizon[h]:
            lgb_m = meta_by_horizon[h]["models"]["D_META_LGBM"]["test_metrics_maker"]
            lr_m = meta_by_horizon[h]["models"]["D_META_LOGREG"]["test_metrics_maker"]
            print(
                f"  Meta[{h.upper():<6}] LogReg={lr_m['exp_bps']:+.2f}bps (n={lr_m['n_trades']}, shrp={lr_m['sharpe_daily']:+.2f}) | "
                f"LGBM={lgb_m['exp_bps']:+.2f}bps (n={lgb_m['n_trades']}, shrp={lgb_m['sharpe_daily']:+.2f})",
                flush=True,
            )

    results = {
        "symbols": symbols,
        "date_range": {"start": str(sample_df.index[0]), "end": str(sample_df.index[-1])},
        "data_quality": {"spot_missing_pct": round(spot_miss, 3), "oi_missing_pct": round(oi_miss, 3)},
        "splits": {
            "train_start": str(sample_df.index[train_start]),
            "train_end": str(sample_df.index[train_end - 1]),
            "val_start": str(sample_df.index[val_start]),
            "val_end": str(sample_df.index[val_end - 1]),
            "test_start": str(sample_df.index[test_start]),
            "test_end": str(sample_df.index[test_end - 1]),
            "train_days": round(train_days, 2),
            "val_days": round(val_days, 2),
            "test_days": round(test_days, 2),
            "oos50_days": round(oos50_days, 2),
        },
        "selected_pairs": selected_pairs,
        "base_table": base_table,
        "oos50_table": oos50_table,
        "family_summary": family_summary,
        "horizon_summary": horizon_summary,
        "ensemble": ens_metrics,
        "meta_all": meta_all,
        "meta_by_horizon": meta_by_horizon,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "tournament_results.json"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # Best ML Meta-Labeler trades for SVG equity curve
    meta_svg_trades = meta_selected_map.get("D_META_LGBM") or meta_selected_map.get("D_META_LOGREG") or apply_regime_killswitch(all_te)
    html_path = REPORTS_DIR / "tournament_report.html"
    write_html_report(results, ens_test_trades, meta_svg_trades, html_path)

    print(f"\n[tournament] Completed in {time.time() - t0:.1f}s.", flush=True)
    print(f"[tournament] JSON saved : {json_path}", flush=True)
    print(f"[tournament] HTML saved : {html_path}", flush=True)


if __name__ == "__main__":
    main()

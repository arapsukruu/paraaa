"""Round 3 Strict Out-of-Sample Validation of the 3-Sleeve Hardened Core Engine.

Strict zero-leakage split:
  - TRAIN (Q1 + Q2, first 50% = 15.3d): Where physics rules (OI deleveraging, Pair DNA, Vol floor) were diagnosed.
  - UNTOUCHED OOS (Q3 + Q4, final 50% = 15.3d): NEVER inspected during Round 3 diagnosis!

Validates:
  1. Did Round 3 physics gates fix Q2 chop AND improve untouched Q3+Q4 OOS?
  2. Exact exit-aware Realistic Hybrid Cost (10 bps TP / 15 bps SL), Full Taker (14 bps), Extreme Stress (20 bps), Maker (6 bps)
  3. Execution Delay +5m stress test on untouched OOS
  4. Direction-Matched Random Null + Long-only vs Short-only + BTC UP vs DOWN days on untouched OOS
  5. Generates final visual HTML report (`quant_lab/reports/round3_core_report.html`).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audit_round2 import direction_matched_null
from config import BARS_PER_DAY, REPORTS_DIR
from core_strategy import (
    ManagedTrade,
    compute_macro_btc_trend_z,
    evaluate_managed_portfolio,
    generate_core_sleeves,
)
from features import compute_universe_features, load_universe
from strategies.cross_pair import select_cointegrated_pairs

WARMUP_BARS = 432


def build_svg_equity(trades: list[ManagedTrade], cost_mode: str = "hybrid") -> str:
    if len(trades) < 2:
        return "<div>Yeterli islem yok</div>"
    eq = [100.0]
    for t in trades:
        r = t.net_ret_hybrid() if cost_mode == "hybrid" else t.net_ret(cost_mode)
        eq.append(eq[-1] * (1.0 + r * t.risk_weight))
    w, h, pad = 760, 220, 24
    mn, mx = min(eq), max(eq)
    span = max(mx - mn, 1e-6)
    pts = [f"{pad + (w-2*pad)*(i/(len(eq)-1)):.1f},{h-pad-(h-2*pad)*((v-mn)/span):.1f}" for i, v in enumerate(eq)]
    y100 = h - pad - (h - 2 * pad) * ((100.0 - mn) / span)
    col = "#22c55e" if eq[-1] >= 100 else "#ef4444"
    return (
        f"<svg viewBox='0 0 {w} {h}' width='100%' height='{h}' style='background:#0f172a;border-radius:10px'>"
        f"<line x1='{pad}' y1='{y100:.1f}' x2='{w-pad}' y2='{y100:.1f}' stroke='#475569' stroke-dasharray='4,4'/>"
        f"<polyline fill='none' stroke='{col}' stroke-width='2.4' points='{' '.join(pts)}'/>"
        f"<text x='{pad+6}' y='18' fill='#94a3b8' font-size='12'>Baslangic: 100.0 | Min: {mn:.2f} | Max: {mx:.2f} | Son: {eq[-1]:.2f}</text>"
        f"</svg>"
    )


def main() -> None:
    t0 = time.time()
    data_map = load_universe()
    feat_map, _ = compute_universe_features(data_map)
    btc_trend_z = compute_macro_btc_trend_z(data_map, feat_map)
    btc_df = data_map["BTCUSDT"]
    n_total = len(btc_df)
    usable = n_total - WARMUP_BARS

    # Strictly 50% TRAIN (Q1+Q2) vs 50% UNTOUCHED OOS (Q3+Q4)
    q_cuts = [WARMUP_BARS + int(usable * p) for p in [0.0, 0.25, 0.50, 0.75, 1.0]]
    train_s, train_e = q_cuts[0], q_cuts[2]
    oos_s, oos_e = q_cuts[2], q_cuts[4]
    train_days = (train_e - train_s) / float(BARS_PER_DAY)
    oos_days = (oos_e - oos_s) / float(BARS_PER_DAY)

    # Pairs selected strictly on TRAIN only (zero OOS peeking!)
    pairs = select_cointegrated_pairs(data_map, train_end_idx=train_e, max_pairs=8)

    tr_sleeves = generate_core_sleeves(data_map, feat_map, btc_trend_z, pairs, train_s, train_e, enable_be_trail=True)
    oos_sleeves = generate_core_sleeves(data_map, feat_map, btc_trend_z, pairs, oos_s, oos_e, enable_be_trail=True)
    oos_notrail = generate_core_sleeves(data_map, feat_map, btc_trend_z, pairs, oos_s, oos_e, enable_be_trail=False)
    oos_delay5 = generate_core_sleeves(data_map, feat_map, btc_trend_z, pairs, oos_s, oos_e, enable_be_trail=True, extra_delay_bars=1)

    tr_all = [t for lst in tr_sleeves.values() for t in lst]
    oos_all = [t for lst in oos_sleeves.values() for t in lst]
    oos_all_d5 = [t for lst in oos_delay5.values() for t in lst]

    print(f"=== ROUND 3 HARDENED 3-SLEEVE CORE ENGINE (TRAIN={train_days:.1f}d vs UNTOUCHED OOS={oos_days:.1f}d) ===\n")

    print("--- 1) SLEEVE & PORTFOLIO BREAKDOWN ON UNTOUCHED OOS (Q3+Q4, 15.3 FULL DAYS) ---")
    hdr = f"{'Sleeve / Portfolio':<24} {'N(t/d)':<12} {'Hold':<6} {'Mkr6':>7} {'Hyb10-15':>9} {'95% CI (Hyb)':>15} {'Tkr14':>7} {'Str20':>7} {'Win%':>6} {'PF':>5} {'Ret%':>7} {'DD%':>7} {'Shrp':>6}"
    print(hdr)
    print("-" * len(hdr))

    report: dict[str, Any] = {"train_days": round(train_days, 2), "oos_days": round(oos_days, 2), "pairs": pairs, "oos_sleeves": {}, "quarters": []}

    for name, lst in [
        ("CORE_OI_CASCADE", oos_sleeves["CORE_OI_CASCADE"]),
        ("CORE_OI_CASCADE (noBE)", oos_notrail["CORE_OI_CASCADE"]),
        ("CORE_SPOT_WHALE", oos_sleeves["CORE_SPOT_WHALE"]),
        ("CORE_PAIRS_NEUT", oos_sleeves["CORE_PAIRS_NEUT"]),
        ("PORTFOLIO_3SLEEVE", oos_all),
        ("PORTFOLIO_DELAY_+5m", oos_all_d5),
    ]:
        mm, _ = evaluate_managed_portfolio(lst, oos_days, "maker")
        mh, acc = evaluate_managed_portfolio(lst, oos_days, "hybrid")
        mt, _ = evaluate_managed_portfolio(lst, oos_days, "taker")
        ms, _ = evaluate_managed_portfolio(lst, oos_days, "stress_20")
        report["oos_sleeves"][name] = {"maker": mm, "hybrid": mh, "taker": mt, "stress_20": ms}
        ci = f"[{mh['ci95_low_bps']:+.1f},{mh['ci95_high_bps']:+.1f}]"
        print(
            f"{name:<24} {mh['n_trades']:>4}({mh['trades_per_day']:>4.1f}) {mh['avg_hold_mins']:>5.0f}m "
            f"{mm['exp_bps']:>+7.2f} {mh['exp_bps']:>+9.2f} {ci:>15} {mt['exp_bps']:>+7.2f} {ms['exp_bps']:>+7.2f} "
            f"{mh['win_rate_pct']:>5.1f}% {mh['profit_factor']:>5.2f} {mh['total_ret_pct']:>+6.2f}% "
            f"{mh['max_dd_pct']:>6.2f}% {mh['sharpe_daily']:>+6.2f}"
        )

    # Check 4-Quarter Stability (Q1 Train, Q2 Chop Train, Q3 Crash OOS, Q4 V-Bounce OOS!)
    print("\n--- 2) 4-QUARTER STABILITY CHECK (DID Q2 CHOP FIX ITSELF & DO Q3/Q4 HOLD?) ---")
    btc_c = btc_df["close"]
    for qi in range(4):
        s, e = q_cuts[qi], q_cuts[qi + 1]
        qd = (e - s) / float(BARS_PER_DAY)
        bret = (btc_c.iloc[e - 1] / btc_c.iloc[s] - 1.0) * 100.0
        tag = "TRAIN" if qi < 2 else "OOS  "
        q_sl = generate_core_sleeves(data_map, feat_map, btc_trend_z, pairs, s, e, enable_be_trail=True)
        q_cas, _ = evaluate_managed_portfolio(q_sl["CORE_OI_CASCADE"], qd, "hybrid", n_bootstrap=300)
        q_all, _ = evaluate_managed_portfolio([t for lst in q_sl.values() for t in lst], qd, "hybrid", n_bootstrap=300)
        report["quarters"].append({"q": f"Q{qi+1}", "split": tag.strip(), "btc_pct": round(bret, 2), "cascade_hyb": q_cas, "port_hyb": q_all})
        print(
            f"  Q{qi+1} [{tag}] ({qd:.1f}d, BTC {bret:+5.1f}%) | "
            f"Cascade(Hyb): {q_cas['exp_bps']:+6.2f}bps (n={q_cas['n_trades']:>3}, PF={q_cas['profit_factor']:.2f}, Ret={q_cas['total_ret_pct']:+5.2f}%) | "
            f"3-Sleeve(Hyb): {q_all['exp_bps']:+6.2f}bps (n={q_all['n_trades']:>3}, PF={q_all['profit_factor']:.2f}, Ret={q_all['total_ret_pct']:+5.2f}%, DD={q_all['max_dd_pct']:5.2f}%)"
        )

    # Direction, BTC Up/Down Days & Direction-Matched Random Null Audit on UNTOUCHED OOS (Q3+Q4)
    mh_port, acc_port = evaluate_managed_portfolio(oos_all, oos_days, "hybrid")
    mh_cas, acc_cas = evaluate_managed_portfolio(oos_sleeves["CORE_OI_CASCADE"], oos_days, "hybrid")

    btc_daily = btc_df["close"].resample("1D").last().pct_change()
    up_dates = set(btc_daily[btc_daily > 0].index.date)
    dn_dates = set(btc_daily[btc_daily <= 0].index.date)

    print("\n--- 3) ANTI-ILLUSION AUDIT ON UNTOUCHED OOS (Q3+Q4, HYBRID REALISTIC COSTS) ---")
    for label, acc in [("CORE_OI_CASCADE", acc_cas), ("PORTFOLIO_3SLEEVE", acc_port)]:
        longs = [t for t in acc if t.side == 1]
        shorts = [t for t in acc if t.side == -1]
        up_t = [t for t in acc if t.entry_ts.date() in up_dates]
        dn_t = [t for t in acc if t.entry_ts.date() in dn_dates]
        l_bps = float(np.mean([t.net_ret_hybrid() for t in longs]) * 10000.0) if longs else 0.0
        s_bps = float(np.mean([t.net_ret_hybrid() for t in shorts]) * 10000.0) if shorts else 0.0
        u_bps = float(np.mean([t.net_ret_hybrid() for t in up_t]) * 10000.0) if up_t else 0.0
        d_bps = float(np.mean([t.net_ret_hybrid() for t in dn_t]) * 10000.0) if dn_t else 0.0
        print(
            f"  {label:<18} | Long(n={len(longs):>3}): {l_bps:+6.2f}bps | Short(n={len(shorts):>3}): {s_bps:+6.2f}bps | "
            f"BTC UP(n={len(up_t):>3}): {u_bps:+6.2f}bps | BTC DOWN(n={len(dn_t):>3}): {d_bps:+6.2f}bps"
        )

    # Direction-Matched Random Null on CORE_OI_CASCADE OOS
    vmap = {s: feat_map[s]["vol_240m"].to_numpy(dtype=np.float64) * 3.8 for s in data_map}
    n_l = sum(1 for t in acc_cas if t.side == 1)
    n_s = sum(1 for t in acc_cas if t.side == -1)
    null_m, null_p95, null_sd = direction_matched_null(data_map, vmap, oos_s, oos_e, n_l, n_s, 2.4, 1.6, 135, 0.0055, n_trials=40)
    # Adjust null from maker(6bp) to hybrid(~12bp avg) -> null_hyb is ~6 bps lower than null_m
    null_hyb = null_m - 6.0
    z_null = (mh_cas["exp_bps"] - null_hyb) / max(null_sd, 1e-6)
    print(f"  CORE_OI_CASCADE vs Dir-Matched Random Null (Hybrid): Signal={mh_cas['exp_bps']:+.2f}bps vs Null={null_hyb:+.2f}bps -> Excess={mh_cas['exp_bps']-null_hyb:+.2f}bps (z={z_null:+.2f})")
    print(f"  Exit Distribution (PORTFOLIO_3SLEEVE): {mh_port['exit_dist']} | Circuit Breaker Days: {mh_port['circuit_breaker_days']}")

    report["null_audit"] = {
        "cascade_hyb_bps": mh_cas["exp_bps"],
        "dir_matched_null_hyb_bps": round(null_hyb, 2),
        "excess_bps": round(mh_cas["exp_bps"] - null_hyb, 2),
        "z_vs_null": round(z_null, 2),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_p = REPORTS_DIR / "round3_core_results.json"
    json_p.write_text(json.dumps(report, indent=2, default=lambda x: float(x)), encoding="utf-8")

    # Write clean HTML dashboard for Round 3 Hardened Core
    svg_port = build_svg_equity(acc_port, "hybrid")
    svg_cas = build_svg_equity(acc_cas, "hybrid")
    q_rows = "".join(
        f"<tr><td><b>{q['q']}</b> ({q['split']})</td><td>{q['btc_pct']:+.1f}%</td>"
        f"<td>{q['cascade_hyb']['n_trades']}</td><td class='{'pos' if q['cascade_hyb']['exp_bps']>0 else 'neg'}'><b>{q['cascade_hyb']['exp_bps']:+.2f} bps</b></td>"
        f"<td>{q['cascade_hyb']['profit_factor']:.2f}</td>"
        f"<td>{q['port_hyb']['n_trades']}</td><td class='{'pos' if q['port_hyb']['exp_bps']>0 else 'neg'}'><b>{q['port_hyb']['exp_bps']:+.2f} bps</b></td>"
        f"<td>{q['port_hyb']['total_ret_pct']:+.2f}%</td><td>{q['port_hyb']['max_dd_pct']:.2f}%</td></tr>"
        for q in report["quarters"]
    )
    html = f"""<!doctype html><html lang="tr"><head><meta charset="utf-8"><title>Round 3 Cekirdek Motor Raporu</title>
<style>body{{font-family:system-ui,sans-serif;background:#020617;color:#e2e8f0;margin:24px;}} table{{width:100%;border-collapse:collapse;background:#0f172a;margin:14px 0 24px;font-size:13px;}} th,td{{padding:9px 11px;border-bottom:1px solid #1e293b;text-align:right;}} th:first-child,td:first-child{{text-align:left;}} th{{background:#1e293b;}} .pos{{color:#4ade80;}} .neg{{color:#f87171;}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px;}} .card{{background:#0f172a;padding:16px;border-radius:10px;border:1px solid #1e293b;}}</style></head><body>
<h1>Round 3 — 3-Kolonlu Guclendirilmis Cekirdek Strateji Motoru (Hibrit Maliyet &amp; OOS Dogrulama)</h1>
<p>Train (Q1+Q2): {train_days:.1f} gun | <b>Untouched OOS (Q3+Q4): {oos_days:.1f} gun</b> | Hibrit Maliyet: Taker Giris (8 bps) + TP Maker Limit (2 bps) / SL Taker (7 bps)</p>
<h2>4 Ceyrek Istikrar Tablosu (Hibrit Gercekci Maliyet)</h2>
<table><thead><tr><th>Ceyrek</th><th>BTC Degisim</th><th>Cascade N</th><th>Cascade Beklenti</th><th>Cascade PF</th><th>3-Kolon N</th><th>3-Kolon Beklenti</th><th>Getiri</th><th>MaxDD</th></tr></thead><tbody>{q_rows}</tbody></table>
<div class="grid">
  <div class="card"><h3>3-Kolonlu Portfoy — OOS Equity (Hibrit Maliyet)</h3>{svg_port}</div>
  <div class="card"><h3>CORE_OI_CASCADE — OOS Equity (Hibrit Maliyet)</h3>{svg_cas}</div>
</div></body></html>"""
    html_p = REPORTS_DIR / "round3_core_report.html"
    html_p.write_text(html, encoding="utf-8")
    print(f"\n[round3] Done in {time.time()-t0:.1f}s -> {json_p} | {html_p}")


if __name__ == "__main__":
    main()

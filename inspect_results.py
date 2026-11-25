"""Diagnostic inspector for Train / Val / Test consistency and BTC regime breakdown."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
r = json.loads((BASE / "reports" / "tournament_results.json").read_text(encoding="utf-8"))
btc = pd.read_pickle(BASE / "data" / "processed" / "BTCUSDT.pkl.gz", compression="gzip")
sp = r["splits"]

print("=== BTC PRICE REGIME BY SPLIT ===")
for k in ["train", "val", "test"]:
    sub = btc.loc[sp[f"{k}_start"] : sp[f"{k}_end"], "close"]
    ret = (sub.iloc[-1] / sub.iloc[0] - 1.0) * 100.0
    hi, lo = sub.max(), sub.min()
    print(f"  {k.upper():<5} ({sp[k+'_days']:>4.1f}d): {sub.iloc[0]:>7.0f} -> {sub.iloc[-1]:>7.0f} ({ret:+6.2f}%) | Range: [{lo:.0f}, {hi:.0f}]")

print("\n=== TRAIN vs VAL vs TEST CONSISTENCY (MAKER bps | 0-COST bps) ===")
print(f"{'Strategy':<19} {'Hor':<6} | {'Train Mkr (n)':>15} | {'Val Mkr (n)':>14} | {'Test Mkr (n)':>14} | {'Test 0c':>8} | {'Tr0c':>7}")
print("-" * 98)
for x in r["base_table"]:
    tr = x["train_maker"]
    va = x["val_maker"]
    te = x["test_maker"]
    tz = x["test_zero"]
    tr_0c = tr["exp_bps"] + 6.0 if tr["n_trades"] > 0 else 0.0
    print(
        f"{x['strategy']:<19} {x['horizon']:<6} | "
        f"{tr['exp_bps']:>+7.1f} ({tr['n_trades']:>4}) | "
        f"{va['exp_bps']:>+6.1f} ({va['n_trades']:>4}) | "
        f"{te['exp_bps']:>+6.1f} ({te['n_trades']:>4}) | "
        f"{tz['exp_bps']:>+8.1f} | {tr_0c:>+7.1f}"
    )

print("\n=== META-LABELER SLICE ANALYSIS (D_META_LOGREG on UNTOUCHED TEST) ===")
lr = r["meta_all"]["models"]["D_META_LOGREG"]
print("  Confusion Matrix (TN, FP / FN, TP):", lr["test_confusion_matrix"])
print("  By Direction:")
for side, m in lr["slice_by_side"].items():
    print(f"    {side:<6}: n={m['n_trades']:>3} | Win={m['win_rate_pct']:>5.1f}% | Maker={m['exp_bps']:>+6.2f} bps | PF={m['profit_factor']:.2f}")
print("  By Strategy:")
for st, m in lr["slice_by_strategy"].items():
    print(f"    {st:<20}: n={m['n_trades']:>3} | Win={m['win_rate_pct']:>5.1f}% | Maker={m['exp_bps']:>+6.2f} bps | PF={m['profit_factor']:.2f}")

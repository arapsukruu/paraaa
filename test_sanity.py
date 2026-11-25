"""Unit tests ensuring zero lookahead, conservative intra-bar tie-break, and funding accounting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.backtester import simulate_pair_trades, simulate_single_asset


def test_conservative_sl_tiebreak_and_next_bar_entry() -> None:
    idx = pd.date_range("2026-01-01", periods=10, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            "high": [100, 100, 105, 100, 100, 100, 100, 100, 100, 100],  # bar 2 hits BOTH +5% TP and -5% SL
            "low": [100, 100, 95, 100, 100, 100, 100, 100, 100, 100],
            "close": [100, 100, 101, 100, 100, 100, 100, 100, 100, 100],
            "funding_event": [0, 0, 0.0001, 0, 0, 0, 0, 0, 0, 0],
        },
        index=idx,
        dtype=np.float64,
    )
    # Signal at bar 0 -> MUST enter at bar 1 open (idx=1)
    trades = simulate_single_asset(
        df=df,
        symbol="TEST",
        strategy="T",
        horizon="short",
        signal_indices=np.array([0]),
        signal_sides=np.array([1]),
        signal_strengths=np.array([1.0]),
        vol_series=np.full(10, 0.01),
        tp_mult=2.0,   # +2%
        sl_mult=2.0,   # -2%
        max_hold_mins=25,
        min_tp=0.002,
    )
    assert len(trades) == 1
    tr = trades[0]
    assert tr.entry_idx == 1, f"Expected entry at bar 1, got {tr.entry_idx}"
    assert tr.exit_idx == 2, f"Expected exit at bar 2, got {tr.exit_idx}"
    assert tr.exit_reason == "SL", f"Conservative tie-break must pick SL, got {tr.exit_reason}"
    assert tr.hold_mins == 10, f"Expected 2 bars x 5m = 10 mins, got {tr.hold_mins}"
    assert np.isclose(tr.gross_ret, -0.02, atol=1e-6)
    assert np.isclose(tr.funding_pnl, -0.0001, atol=1e-9)  # Long pays positive funding


def test_pair_hedge_neutrality_under_market_crash() -> None:
    idx = pd.date_range("2026-01-01", periods=20, freq="1min", tz="UTC")
    # Both A and B crash -10% together, while A recovers +1% relative to B -> Long A / Short B makes +0.5% on gross
    pa = np.full(20, 100.0)
    pb = np.full(20, 100.0)
    pa[4:] = 90.9   # -9.1%
    pb[4:] = 90.0   # -10.0%
    df_a = pd.DataFrame({"open": pa, "close": pa, "funding_event": np.zeros(20)}, index=idx)
    df_b = pd.DataFrame({"open": pb, "close": pb, "funding_event": np.zeros(20)}, index=idx)
    z = np.zeros(20)
    z[1] = -2.5  # Long A / Short B signal at bar 1 -> entry at bar 2
    z[4] = 0.0   # Reverted at bar 4 -> exit at bar 5 open
    beta = np.ones(20)
    trades = simulate_pair_trades(
        df_a, df_b, "A", "B", "PAIR", "short", z, beta,
        entry_z=2.0, exit_z=0.2, stop_z=4.0, max_hold_mins=50, start_idx=1, end_idx=19,
    )
    assert len(trades) == 1
    # Expected gross return = +1 * (0.5 * (-0.091) - 0.5 * (-0.10)) = +0.0045 (+45 bps) despite -10% crash!
    assert np.isclose(trades[0].gross_ret, 0.0045, atol=1e-5), f"Got {trades[0].gross_ret}"


if __name__ == "__main__":
    test_conservative_sl_tiebreak_and_next_bar_entry()
    test_pair_hedge_neutrality_under_market_crash()
    print("ALL SANITY TESTS PASSED!")

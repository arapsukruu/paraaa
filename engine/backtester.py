"""Realistic, leak-free perpetual futures backtester and statistical validator.

Key guarantees:
  1. Strict causality: signal at bar t executes at bar t+1 open (zero lookahead).
  2. Conservative intra-bar tie-breaking: if both TP and SL touch in the same 1m bar, SL wins.
  3. Exact funding accounting across holding windows for both single-asset and 2-leg pairs.
  4. Multi-cost evaluation (zero_cost / maker / taker) for every trade.
  5. Portfolio-level concurrency limits + daily Sharpe + Bootstrap 95% CI + Random Null test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from config import BAR_MINS, COST_SCENARIOS


@dataclass
class Trade:
    strategy: str
    horizon: str
    symbol: str
    side: int  # +1 long (or long spread A-B), -1 short
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_idx: int
    exit_idx: int
    gross_ret: float
    funding_pnl: float
    hold_mins: int
    exit_reason: str
    vol_at_entry: float
    signal_strength: float = 0.0
    meta_features: dict[str, float] = field(default_factory=dict)

    def net_ret(self, cost_mode: str = "maker") -> float:
        c = COST_SCENARIOS[cost_mode]
        rt_cost = 2.0 * (c["fee_one_way"] + c["slippage_one_way"])
        return self.gross_ret + self.funding_pnl - rt_cost


def simulate_single_asset(
    df: pd.DataFrame,
    symbol: str,
    strategy: str,
    horizon: str,
    signal_indices: np.ndarray,
    signal_sides: np.ndarray,
    signal_strengths: np.ndarray,
    vol_series: np.ndarray,
    tp_mult: float,
    sl_mult: float,
    max_hold_mins: int,
    min_tp: float = 0.0020,
    max_tp: float = 0.0800,
    cooldown_mins: int = 5,
    exit_signal_fn=None,
    meta_feature_df: pd.DataFrame | None = None,
) -> list[Trade]:
    """Simulate non-overlapping single-asset perpetual trades with Triple-Barrier exits."""
    opens = df["open"].to_numpy(dtype=np.float64)
    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)
    funding = df["funding_event"].to_numpy(dtype=np.float64)
    timestamps = df.index
    n = len(df)

    meta_cols = list(meta_feature_df.columns) if meta_feature_df is not None else []
    meta_vals = meta_feature_df.to_numpy(dtype=np.float64) if meta_feature_df is not None else None

    trades: list[Trade] = []
    next_allowed_idx = 0

    for k in range(len(signal_indices)):
        sig_i = int(signal_indices[k])
        if sig_i < next_allowed_idx or sig_i >= n - 2:
            continue

        side = int(signal_sides[k])
        if side == 0:
            continue

        vol = float(vol_series[sig_i])
        if not np.isfinite(vol) or vol <= 1e-6:
            continue

        entry_i = sig_i + 1
        entry_px = opens[entry_i]
        if not np.isfinite(entry_px) or entry_px <= 0:
            continue

        tp_dist = float(np.clip(tp_mult * vol, min_tp, max_tp))
        sl_dist = float(np.clip(sl_mult * vol, min_tp * 0.75, max_tp))

        if side == 1:
            tp_px = entry_px * (1.0 + tp_dist)
            sl_px = entry_px * (1.0 - sl_dist)
        else:
            tp_px = entry_px * (1.0 - tp_dist)
            sl_px = entry_px * (1.0 + sl_dist)

        max_hold_bars = max(1, int(round(max_hold_mins / BAR_MINS)))
        cooldown_bars = max(1, int(round(cooldown_mins / BAR_MINS)))
        end_j = min(entry_i + max_hold_bars, n - 1)
        exit_j = end_j
        exit_px = closes[end_j]
        reason = "TIME"

        for j in range(entry_i, end_j + 1):
            if side == 1:
                hit_sl = lows[j] <= sl_px
                hit_tp = highs[j] >= tp_px
            else:
                hit_sl = highs[j] >= sl_px
                hit_tp = lows[j] <= tp_px

            # Conservative tie-break: SL always wins if both touched in same bar
            if hit_sl:
                exit_j = j
                exit_px = sl_px
                reason = "SL"
                break
            if hit_tp:
                exit_j = j
                exit_px = tp_px
                reason = "TP"
                break
            if exit_signal_fn is not None and j > entry_i and exit_signal_fn(j, side):
                exit_j = j
                exit_px = closes[j]
                reason = "SIGNAL"
                break

        gross_ret = side * ((exit_px - entry_px) / entry_px)
        # Funding paid by longs (+1) when funding_event > 0
        fund_pnl = -side * float(np.sum(funding[entry_i : exit_j + 1]))

        mfeat = {}
        if meta_vals is not None:
            row = meta_vals[sig_i]
            mfeat = {col: float(row[c_i]) for c_i, col in enumerate(meta_cols)}

        trades.append(
            Trade(
                strategy=strategy,
                horizon=horizon,
                symbol=symbol,
                side=side,
                entry_ts=timestamps[entry_i],
                exit_ts=timestamps[exit_j],
                entry_idx=entry_i,
                exit_idx=exit_j,
                gross_ret=float(gross_ret),
                funding_pnl=float(fund_pnl),
                hold_mins=int((exit_j - entry_i + 1) * BAR_MINS),
                exit_reason=reason,
                vol_at_entry=vol,
                signal_strength=float(signal_strengths[k]),
                meta_features=mfeat,
            )
        )
        next_allowed_idx = exit_j + cooldown_bars

    return trades


def simulate_pair_trades(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    sym_a: str,
    sym_b: str,
    strategy: str,
    horizon: str,
    z_series: np.ndarray,
    beta_series: np.ndarray,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    max_hold_mins: int,
    start_idx: int,
    end_idx: int,
    cooldown_mins: int = 10,
    valid_mask: np.ndarray | None = None,
    meta_feature_df: pd.DataFrame | None = None,
) -> list[Trade]:
    """Simulate market-neutral 2-leg perpetual trades (unit gross notional: w_a + w_b = 1)."""
    open_a = df_a["open"].to_numpy(dtype=np.float64)
    close_a = df_a["close"].to_numpy(dtype=np.float64)
    fund_a = df_a["funding_event"].to_numpy(dtype=np.float64)

    open_b = df_b["open"].to_numpy(dtype=np.float64)
    close_b = df_b["close"].to_numpy(dtype=np.float64)
    fund_b = df_b["funding_event"].to_numpy(dtype=np.float64)

    timestamps = df_a.index
    n = min(len(df_a), len(df_b), end_idx)

    meta_cols = list(meta_feature_df.columns) if meta_feature_df is not None else []
    meta_vals = meta_feature_df.to_numpy(dtype=np.float64) if meta_feature_df is not None else None

    trades: list[Trade] = []
    i = max(start_idx, 1)

    while i < n - 2:
        if valid_mask is not None and not valid_mask[i]:
            i += 1
            continue

        z = z_series[i]
        beta = beta_series[i]
        if not np.isfinite(z) or not np.isfinite(beta) or beta <= 0.2 or beta >= 5.0:
            i += 1
            continue

        # Mean reversion on spread = log(A) - beta * log(B):
        # z > +entry_z -> A overpriced vs B -> Short A, Long B (side = -1)
        # z < -entry_z -> A underpriced vs B -> Long A, Short B (side = +1)
        side = 0
        if z >= entry_z and z < stop_z * 0.9:
            side = -1
        elif z <= -entry_z and z > -stop_z * 0.9:
            side = 1
        else:
            i += 1
            continue

        entry_i = i + 1
        pa_in = open_a[entry_i]
        pb_in = open_b[entry_i]
        if not (np.isfinite(pa_in) and np.isfinite(pb_in) and pa_in > 0 and pb_in > 0):
            i += 1
            continue

        # Beta-hedged weights normalized to unit gross notional (|w_a| + |w_b| = 1.0)
        w_a = 1.0 / (1.0 + beta)
        w_b = beta / (1.0 + beta)

        max_hold_bars = max(1, int(round(max_hold_mins / BAR_MINS)))
        cooldown_bars = max(1, int(round(cooldown_mins / BAR_MINS)))
        end_j = min(entry_i + max_hold_bars, n - 1)
        exit_j = end_j
        reason = "TIME"

        # Hard spread PnL stop (2.5% gross notional) in addition to z-score stop
        for j in range(entry_i + 1, end_j + 1):
            zj = z_series[j]
            ret_a_j = (close_a[j] - pa_in) / pa_in
            ret_b_j = (close_b[j] - pb_in) / pb_in
            cur_pnl = side * (w_a * ret_a_j - w_b * ret_b_j)

            if cur_pnl <= -0.025 or (np.isfinite(zj) and side * zj <= -stop_z):
                # Stop-loss: spread diverged further against us
                exit_j = j
                reason = "SL"
                break
            if np.isfinite(zj) and ((side == 1 and zj >= -exit_z) or (side == -1 and zj <= exit_z)):
                exit_j = j
                reason = "TP"
                break

        # Exit at next bar open if available (strictly causal exit after observing close z_j!)
        fill_exit_j = min(exit_j + 1, n - 1)
        pa_out = open_a[fill_exit_j] if fill_exit_j > exit_j else close_a[exit_j]
        pb_out = open_b[fill_exit_j] if fill_exit_j > exit_j else close_b[exit_j]

        ret_a = (pa_out - pa_in) / pa_in
        ret_b = (pb_out - pb_in) / pb_in
        gross_ret = side * (w_a * ret_a - w_b * ret_b)

        # Funding on both legs
        fa = float(np.sum(fund_a[entry_i : fill_exit_j + 1]))
        fb = float(np.sum(fund_b[entry_i : fill_exit_j + 1]))
        fund_pnl = (-side * w_a * fa) + (side * w_b * fb)

        mfeat = {}
        if meta_vals is not None:
            row = meta_vals[i]
            mfeat = {col: float(row[c_i]) for c_i, col in enumerate(meta_cols)}

        trades.append(
            Trade(
                strategy=strategy,
                horizon=horizon,
                symbol=f"{sym_a}/{sym_b}",
                side=side,
                entry_ts=timestamps[entry_i],
                exit_ts=timestamps[fill_exit_j],
                entry_idx=entry_i,
                exit_idx=fill_exit_j,
                gross_ret=float(gross_ret),
                funding_pnl=float(fund_pnl),
                hold_mins=int((fill_exit_j - entry_i + 1) * BAR_MINS),
                exit_reason=reason,
                vol_at_entry=float(abs(z)),
                signal_strength=float(abs(z)),
                meta_features=mfeat,
            )
        )
        i = fill_exit_j + cooldown_bars

    return trades


def apply_portfolio_constraints(
    trades: list[Trade],
    max_concurrent: int = 6,
    max_per_symbol: int = 1,
) -> list[Trade]:
    """Chronologically filter trades to enforce realistic portfolio concurrency limits."""
    if not trades:
        return []
    ordered = sorted(trades, key=lambda t: (t.entry_ts, -t.signal_strength))
    accepted: list[Trade] = []
    active: list[Trade] = []

    for tr in ordered:
        # Release closed positions
        active = [a for a in active if a.exit_ts > tr.entry_ts]
        if len(active) >= max_concurrent:
            continue
        sym_count = sum(1 for a in active if a.symbol == tr.symbol)
        if sym_count >= max_per_symbol:
            continue
        active.append(tr)
        accepted.append(tr)
    return accepted


def evaluate_trades(
    trades: list[Trade],
    period_days: float,
    cost_mode: str = "maker",
    max_concurrent: int = 6,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute rigorous portfolio & statistical metrics for a list of trades."""
    selected = apply_portfolio_constraints(trades, max_concurrent=max_concurrent)
    if not selected:
        return {
            "n_trades": 0,
            "trades_per_day": 0.0,
            "win_rate_pct": 0.0,
            "exp_bps": 0.0,
            "ci95_low_bps": 0.0,
            "ci95_high_bps": 0.0,
            "profit_factor": 0.0,
            "total_ret_pct": 0.0,
            "max_dd_pct": 0.0,
            "sharpe_daily": 0.0,
            "avg_hold_mins": 0.0,
            "tp_pct": 0.0,
            "sl_pct": 0.0,
            "time_pct": 0.0,
            "funding_bps": 0.0,
        }

    rets = np.array([t.net_ret(cost_mode) for t in selected], dtype=np.float64)
    funds = np.array([t.funding_pnl for t in selected], dtype=np.float64)
    holds = np.array([t.hold_mins for t in selected], dtype=np.float64)

    # Portfolio capital allocation per slot = 1 / max_concurrent (unlevered 1x gross cap)
    slot_weight = 1.0 / float(max_concurrent)
    port_trade_rets = rets * slot_weight

    # Daily PnL series by exit date for honest annualized Sharpe (365 days/year for crypto)
    exit_dates = pd.to_datetime([t.exit_ts for t in selected]).floor("D")
    daily_pnl = pd.Series(port_trade_rets, index=exit_dates).groupby(level=0).sum()
    if len(daily_pnl) > 0:
        full_days = pd.date_range(daily_pnl.index.min(), daily_pnl.index.max(), freq="D", tz=daily_pnl.index.tz)
        daily_pnl = daily_pnl.reindex(full_days, fill_value=0.0)

    equity = np.cumprod(1.0 + daily_pnl.to_numpy()) if len(daily_pnl) > 0 else np.array([1.0])
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
    max_dd_pct = float(np.min(dd) * 100.0)
    total_ret_pct = float((equity[-1] - 1.0) * 100.0)

    daily_std = float(daily_pnl.std(ddof=1)) if len(daily_pnl) > 2 else 0.0
    sharpe_daily = float((daily_pnl.mean() / daily_std) * np.sqrt(365.0)) if daily_std > 1e-12 else 0.0

    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    gross_win = float(np.sum(wins))
    gross_loss = float(np.abs(np.sum(losses)))
    pf = gross_win / gross_loss if gross_loss > 1e-12 else (99.0 if gross_win > 0 else 0.0)

    # Bootstrap 95% CI on mean trade expectancy (in bps)
    rng = np.random.default_rng(seed)
    n_t = len(rets)
    if n_t >= 5:
        idx = rng.integers(0, n_t, size=(n_bootstrap, n_t))
        boot_means = rets[idx].mean(axis=1) * 10000.0
        ci_lo, ci_hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
    else:
        m_bps = float(rets.mean() * 10000.0)
        ci_lo, ci_hi = m_bps, m_bps

    reasons = [t.exit_reason for t in selected]
    return {
        "n_trades": int(n_t),
        "trades_per_day": round(n_t / max(period_days, 1e-6), 2),
        "win_rate_pct": round(float(np.mean(rets > 0) * 100.0), 2),
        "exp_bps": round(float(rets.mean() * 10000.0), 2),
        "ci95_low_bps": round(ci_lo, 2),
        "ci95_high_bps": round(ci_hi, 2),
        "profit_factor": round(float(pf), 3),
        "total_ret_pct": round(total_ret_pct, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "sharpe_daily": round(sharpe_daily, 2),
        "avg_hold_mins": round(float(holds.mean()), 1),
        "tp_pct": round(100.0 * reasons.count("TP") / n_t, 1),
        "sl_pct": round(100.0 * reasons.count("SL") / n_t, 1),
        "time_pct": round(100.0 * reasons.count("TIME") / n_t, 1),
        "funding_bps": round(float(funds.mean() * 10000.0), 2),
    }


def random_entry_baseline(
    data_map: dict[str, pd.DataFrame],
    vol_map: dict[str, np.ndarray],
    start_idx: int,
    end_idx: int,
    target_trades: int,
    tp_mult: float,
    sl_mult: float,
    max_hold_mins: int,
    cost_mode: str = "maker",
    n_trials: int = 30,
    seed: int = 123,
) -> dict[str, float]:
    """Monte Carlo random-entry null baseline with identical horizon & barriers."""
    rng = np.random.default_rng(seed)
    syms = list(data_map.keys())
    per_sym = max(1, int(np.ceil(target_trades / max(len(syms), 1))))
    trial_exps = []

    for _ in range(n_trials):
        all_t: list[Trade] = []
        for sym in syms:
            df = data_map[sym]
            hi = min(end_idx, len(df) - max_hold_mins - 2)
            if hi <= start_idx + 10:
                continue
            idxs = np.sort(rng.integers(start_idx, hi, size=per_sym))
            sides = rng.choice([-1, 1], size=per_sym)
            t = simulate_single_asset(
                df=df,
                symbol=sym,
                strategy="RANDOM_NULL",
                horizon="null",
                signal_indices=idxs,
                signal_sides=sides,
                signal_strengths=np.ones(per_sym),
                vol_series=vol_map[sym],
                tp_mult=tp_mult,
                sl_mult=sl_mult,
                max_hold_mins=max_hold_mins,
                cooldown_mins=5,
            )
            all_t.extend(t)
        if all_t:
            trial_exps.append(np.mean([tr.net_ret(cost_mode) for tr in all_t]) * 10000.0)

    if not trial_exps:
        return {"null_mean_bps": 0.0, "null_p95_bps": 0.0}
    return {
        "null_mean_bps": round(float(np.mean(trial_exps)), 2),
        "null_p95_bps": round(float(np.percentile(trial_exps, 95)), 2),
    }

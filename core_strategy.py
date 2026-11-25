"""Production-grade 3-Sleeve Hardened Core Strategy Module.

Shared between:
  - Offline Walk-Forward Validation (run_round3_hardened.py)
  - Live Automated Gate.io Paper-Trading Engine (live_paper_engine.py)

Three orthogonal sleeves (zero classic 5m/15m RSI/MACD indicators):
  1. CORE_OI_CASCADE  : Real Liquidation + Actual OI Deleveraging Drift + Pair-DNA & Vol Floor
  2. CORE_SPOT_WHALE  : Spot Leadership + Top-Trader (Whale) Positioning Continuation (Up-Day Alpha)
  3. CORE_PAIRS_NEUT  : Fee-Aware Market-Neutral Sector Pairs (Crash/Chop Stabilizer)

Risk Management:
  - Volatility-normalized position sizing (equal risk per trade)
  - Conservative breakeven profit lock (+1.1R -> move SL to +12 bps) & trailing stop
  - Exact exit-aware cost model (Taker entry + Maker Limit TP vs Taker SL)
  - Daily drawdown circuit breaker (-2.5% UTC day cap)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from config import BAR_MINS, BARS_PER_DAY
from engine.backtester import Trade, simulate_pair_trades


@dataclass
class ManagedTrade(Trade):
    risk_weight: float = 0.1667  # Portfolio notional allocation for this trade (vol-targeted)
    sl_dist: float = 0.015
    tp_dist: float = 0.025
    be_triggered: bool = False

    def net_ret_hybrid(self) -> float:
        """Realistic automated momentum execution:
        - Entry: Taker 5 bps + 3 bps slippage = 8 bps
        - Exit : Resting Limit TP -> Maker 2 bps (0 slip) = 10 bps RT total
                 Market SL / TIME -> Taker 5 bps + 2 bps slip = 15 bps RT total
        """
        entry_cost = 0.0008
        exit_cost = 0.0002 if self.exit_reason == "TP" else 0.0007
        return self.gross_ret + self.funding_pnl - (entry_cost + exit_cost)

    def net_ret_stress(self, rt_bps: float = 20.0) -> float:
        return self.gross_ret + self.funding_pnl - (rt_bps / 10000.0)


def compute_macro_btc_trend_z(data_map: dict[str, pd.DataFrame], feat_map: dict[str, pd.DataFrame]) -> np.ndarray:
    """Strictly causal 24h BTC macro trend Z-score."""
    btc = data_map["BTCUSDT"]["close"].astype(np.float64)
    r24h = np.log(btc / btc.shift(BARS_PER_DAY)).fillna(0.0)
    vol5 = feat_map["BTCUSDT"]["vol_1440m"].astype(np.float64)
    z24h = (r24h / (vol5 * np.sqrt(float(BARS_PER_DAY)) + 1e-9)).clip(-6.0, 6.0).fillna(0.0)
    return z24h.to_numpy(dtype=np.float64)


def simulate_managed_single_asset(
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
    min_tp: float = 0.0050,
    max_tp: float = 0.0800,
    cooldown_mins: int = 45,
    enable_be_trail: bool = True,
    be_trigger_r: float = 1.05,
    trail_r: float = 1.15,
    target_risk_per_trade: float = 0.0050,  # 0.50% portfolio equity risked per stop-out
    min_notional_w: float = 0.08,
    max_notional_w: float = 0.28,
    extra_delay_bars: int = 0,
    meta_feature_df: pd.DataFrame | None = None,
) -> list[ManagedTrade]:
    """Conservative managed trade simulator with vol-targeted sizing & breakeven/trailing stop.

    Strict causal & conservative intra-bar ordering:
      1. Entry strictly at bar (sig_i + 1 + extra_delay_bars) OPEN.
      2. On each bar j, check existing active SL BEFORE checking TP or updating high-water mark!
         (Never assumes price hit favorable peak before hitting stop inside the same bar.)
    """
    opens = df["open"].to_numpy(dtype=np.float64)
    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)
    funding = df["funding_event"].to_numpy(dtype=np.float64)
    timestamps = df.index
    n = len(df)

    meta_cols = list(meta_feature_df.columns) if meta_feature_df is not None else []
    meta_vals = meta_feature_df.to_numpy(dtype=np.float64) if meta_feature_df is not None else None

    trades: list[ManagedTrade] = []
    next_allowed_idx = 0
    max_hold_bars = max(1, int(round(max_hold_mins / BAR_MINS)))
    cooldown_bars = max(1, int(round(cooldown_mins / BAR_MINS)))

    for k in range(len(signal_indices)):
        sig_i = int(signal_indices[k])
        entry_i = sig_i + 1 + extra_delay_bars
        if entry_i < next_allowed_idx or entry_i >= n - 2:
            continue

        side = int(signal_sides[k])
        if side == 0:
            continue

        vol = float(vol_series[sig_i])
        if not np.isfinite(vol) or vol <= 1e-6:
            continue

        entry_px = opens[entry_i]
        if not np.isfinite(entry_px) or entry_px <= 0:
            continue

        tp_dist = float(np.clip(tp_mult * vol, min_tp, max_tp))
        sl_dist = float(np.clip(sl_mult * vol, min_tp * 0.72, max_tp * 0.85))
        risk_w = float(np.clip(target_risk_per_trade / max(sl_dist, 1e-4), min_notional_w, max_notional_w))

        if side == 1:
            tp_px = entry_px * (1.0 + tp_dist)
            active_sl = entry_px * (1.0 - sl_dist)
            best_fav = entry_px
        else:
            tp_px = entry_px * (1.0 - tp_dist)
            active_sl = entry_px * (1.0 + sl_dist)
            best_fav = entry_px

        end_j = min(entry_i + max_hold_bars, n - 1)
        exit_j = end_j
        exit_px = closes[end_j]
        reason = "TIME"
        be_triggered = False

        for j in range(entry_i, end_j + 1):
            # Step 1: Check EXISTING stop-loss FIRST (conservative!)
            if side == 1:
                hit_sl = lows[j] <= active_sl
                hit_tp = highs[j] >= tp_px
            else:
                hit_sl = highs[j] >= active_sl
                hit_tp = lows[j] <= tp_px

            if hit_sl:
                exit_j = j
                exit_px = active_sl
                reason = "BE_TRAIL" if be_triggered else "SL"
                break
            if hit_tp:
                exit_j = j
                exit_px = tp_px
                reason = "TP"
                break

            # Step 2: AFTER bar j survives active_sl, update favorable watermark & trailing stop for bar j+1
            if enable_be_trail:
                if side == 1:
                    if closes[j] > best_fav:
                        best_fav = closes[j]
                    fav_move = (best_fav - entry_px) / entry_px
                    if fav_move >= be_trigger_r * sl_dist:
                        be_triggered = True
                        be_floor = entry_px * (1.0 + 0.0012)  # Lock +12 bps above entry
                        trail_px = best_fav * (1.0 - trail_r * sl_dist)
                        active_sl = max(active_sl, be_floor, trail_px)
                else:
                    if closes[j] < best_fav:
                        best_fav = closes[j]
                    fav_move = (entry_px - best_fav) / entry_px
                    if fav_move >= be_trigger_r * sl_dist:
                        be_triggered = True
                        be_floor = entry_px * (1.0 - 0.0012)
                        trail_px = best_fav * (1.0 + trail_r * sl_dist)
                        active_sl = min(active_sl, be_floor, trail_px)

        gross_ret = side * ((exit_px - entry_px) / entry_px)
        fund_pnl = -side * float(np.sum(funding[entry_i : exit_j + 1]))

        mfeat = {}
        if meta_vals is not None:
            row = meta_vals[sig_i]
            mfeat = {col: float(row[c_i]) for c_i, col in enumerate(meta_cols)}

        trades.append(
            ManagedTrade(
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
                risk_weight=risk_w,
                sl_dist=sl_dist,
                tp_dist=tp_dist,
                be_triggered=be_triggered,
            )
        )
        next_allowed_idx = exit_j + cooldown_bars

    return trades


def detect_live_signals_at_bar(
    feat_map: dict[str, pd.DataFrame],
    btc_trend_z_val: float,
    bar_idx: int,
) -> list[dict[str, Any]]:
    """Shared real-time signal scanner used by BOTH backtests and live paper-trading engine.

    Evaluates strictly the single completed bar `bar_idx` across all symbols in `feat_map`.
    Returns sorted list of candidate signals (highest signal_strength first).
    """
    signals: list[dict[str, Any]] = []

    for sym, f in feat_map.items():
        if bar_idx < 0 or bar_idx >= len(f):
            continue
        row = f.iloc[bar_idx]
        vol240 = float(row["vol_240m"])
        if not np.isfinite(vol240) or vol240 <= 1e-6:
            continue

        r15 = float(row["r15"])
        p_ofi5 = float(row["perp_ofi_5m"])
        p_ofi15 = float(row["perp_ofi_15m"])
        s_ofi15 = float(row["spot_ofi_15m"])
        sp_div15 = float(row["spot_perp_div_15m"])
        prem_z = float(row["prem_z_240m"])
        oi_chg15 = float(row["oi_chg_15m"])
        real_liq_z = float(row["real_liq_z"])
        liq_sc = float(row["liq_cascade_score"])
        v_surge = float(row["vol_surge_15m"])
        top_z = float(row["top_ls_z"])
        smart_z = float(row["smart_retail_z"])
        vr = float(row["vr_15"])

        ret15_norm = r15 / (vol240 * np.sqrt(3.0) + 1e-9)

        # --- Sleeve 1: CORE_OI_CASCADE (Train-verified physics gates!) ---
        # Physics Gate 1: Actual OI deleveraging (oi_chg15 <= -0.0055, i.e., >= 0.55% OI drop)
        # Physics Gate 2: Active volatility floor (vol240 >= 12 bps) & Non-choppy Pair DNA (vr >= 0.68)
        # Physics Gate 3: Anti-climax cap (|real_liq_z| <= 4.5)
        oi_delev = oi_chg15 <= -0.0055
        dna_active = (vr >= 0.68) and (vol240 >= 0.0012)
        not_climax = abs(real_liq_z) <= 4.5

        if oi_delev and dna_active and not_climax and v_surge >= 1.4:
            if (
                btc_trend_z_val <= 1.2
                and (real_liq_z <= -1.4 or liq_sc <= -1.9)
                and ret15_norm <= -1.05
                and p_ofi5 <= -0.04
            ):
                strength = abs(oi_chg15) * 100.0 + abs(real_liq_z) * 0.7 + abs(ret15_norm) * 0.4
                signals.append({
                    "strategy": "CORE_OI_CASCADE",
                    "symbol": sym,
                    "side": -1,
                    "strength": float(strength),
                    "vol": vol240 * 3.8,
                    "tp_mult": 2.4,
                    "sl_mult": 1.6,
                    "max_hold_mins": 135,
                    "min_tp": 0.0055,
                    "reason": f"LongLiqZ={real_liq_z:.2f}, OI15m={oi_chg15*100:.2f}%, VR={vr:.2f}, Vol={vol240*10000:.1f}bp",
                })
            elif (
                btc_trend_z_val >= -1.2
                and (real_liq_z >= 1.4 or liq_sc >= 1.9)
                and ret15_norm >= 1.05
                and p_ofi5 >= 0.04
            ):
                strength = abs(oi_chg15) * 100.0 + abs(real_liq_z) * 0.7 + abs(ret15_norm) * 0.4
                signals.append({
                    "strategy": "CORE_OI_CASCADE",
                    "symbol": sym,
                    "side": 1,
                    "strength": float(strength),
                    "vol": vol240 * 3.8,
                    "tp_mult": 2.4,
                    "sl_mult": 1.6,
                    "max_hold_mins": 135,
                    "min_tp": 0.0055,
                    "reason": f"ShortLiqZ={real_liq_z:.2f}, OI15m={oi_chg15*100:.2f}%, VR={vr:.2f}, Vol={vol240*10000:.1f}bp",
                })

        # --- Sleeve 2: CORE_SPOT_WHALE (Bull / Up-Day & Organic Demand Complement) ---
        if vr >= 0.88 and vol240 >= 0.0012 and v_surge >= 1.5 and oi_chg15 >= -0.004:
            if s_ofi15 >= 0.24 and sp_div15 >= 0.06 and p_ofi15 >= 0.03 and (smart_z >= 0.7 or top_z >= 0.9) and prem_z <= 1.1:
                strength = abs(s_ofi15) * 3.0 + abs(smart_z) * 0.7
                signals.append({
                    "strategy": "CORE_SPOT_WHALE",
                    "symbol": sym,
                    "side": 1,
                    "strength": float(strength),
                    "vol": vol240 * 5.8,
                    "tp_mult": 3.4,
                    "sl_mult": 2.1,
                    "max_hold_mins": 360,
                    "min_tp": 0.0085,
                    "reason": f"SpotOFI={s_ofi15:+.2f}, SmartZ={smart_z:+.2f}, VR={vr:.2f}, PremZ={prem_z:+.2f}",
                })
            elif s_ofi15 <= -0.24 and sp_div15 <= -0.06 and p_ofi15 <= -0.03 and (smart_z <= -0.7 or top_z <= -0.9) and prem_z >= -1.1:
                strength = abs(s_ofi15) * 3.0 + abs(smart_z) * 0.7
                signals.append({
                    "strategy": "CORE_SPOT_WHALE",
                    "symbol": sym,
                    "side": -1,
                    "strength": float(strength),
                    "vol": vol240 * 5.8,
                    "tp_mult": 3.4,
                    "sl_mult": 2.1,
                    "max_hold_mins": 360,
                    "min_tp": 0.0085,
                    "reason": f"SpotOFI={s_ofi15:+.2f}, SmartZ={smart_z:+.2f}, VR={vr:.2f}, PremZ={prem_z:+.2f}",
                })

    signals.sort(key=lambda x: x["strength"], reverse=True)
    return signals


def generate_core_sleeves(
    data_map: dict[str, pd.DataFrame],
    feat_map: dict[str, pd.DataFrame],
    btc_trend_z: np.ndarray,
    selected_pairs: list[tuple[str, str]],
    start_idx: int,
    end_idx: int,
    enable_be_trail: bool = True,
    extra_delay_bars: int = 0,
) -> dict[str, list[ManagedTrade]]:
    """Vectorized historical generation of all 3 hardened core sleeves on [start_idx, end_idx)."""
    sleeves: dict[str, list[ManagedTrade]] = {
        "CORE_OI_CASCADE": [],
        "CORE_SPOT_WHALE": [],
        "CORE_PAIRS_NEUT": [],
    }

    for sym, df in data_map.items():
        f = feat_map[sym]
        n = len(df)
        hi = min(end_idx, n)
        idx_arr = np.arange(n)
        in_win = (idx_arr >= start_idx) & (idx_arr < hi - extra_delay_bars - 2)

        vol240 = f["vol_240m"].to_numpy(dtype=np.float64)
        r15 = f["r15"].to_numpy(dtype=np.float64)
        p_ofi5 = f["perp_ofi_5m"].to_numpy(dtype=np.float64)
        p_ofi15 = f["perp_ofi_15m"].to_numpy(dtype=np.float64)
        s_ofi15 = f["spot_ofi_15m"].to_numpy(dtype=np.float64)
        sp_div15 = f["spot_perp_div_15m"].to_numpy(dtype=np.float64)
        prem_z = f["prem_z_240m"].to_numpy(dtype=np.float64)
        oi_chg15 = f["oi_chg_15m"].to_numpy(dtype=np.float64)
        real_liq_z = f["real_liq_z"].to_numpy(dtype=np.float64)
        liq_sc = f["liq_cascade_score"].to_numpy(dtype=np.float64)
        v_surge = f["vol_surge_15m"].to_numpy(dtype=np.float64)
        top_z = f["top_ls_z"].to_numpy(dtype=np.float64)
        smart_z = f["smart_retail_z"].to_numpy(dtype=np.float64)
        vr = f["vr_15"].to_numpy(dtype=np.float64)

        ret15_norm = r15 / (vol240 * np.sqrt(3.0) + 1e-9)

        # --- Sleeve 1: CORE_OI_CASCADE (Exact match to detect_live_signals_at_bar!) ---
        base_gate = in_win & (oi_chg15 <= -0.0055) & (vr >= 0.68) & (vol240 >= 0.0012) & (np.abs(real_liq_z) <= 4.5) & (v_surge >= 1.4)
        cas_short = base_gate & (btc_trend_z <= 1.2) & ((real_liq_z <= -1.4) | (liq_sc <= -1.9)) & (ret15_norm <= -1.05) & (p_ofi5 <= -0.04)
        cas_long = base_gate & (btc_trend_z >= -1.2) & ((real_liq_z >= 1.4) | (liq_sc >= 1.9)) & (ret15_norm >= 1.05) & (p_ofi5 >= 0.04)
        m_cas = cas_short | cas_long
        if np.any(m_cas):
            idxs = np.flatnonzero(m_cas)
            sides = np.where(cas_long[idxs], 1, -1)
            strengths = np.abs(oi_chg15[idxs]) * 100.0 + np.abs(real_liq_z[idxs]) * 0.7 + np.abs(ret15_norm[idxs]) * 0.4
            sleeves["CORE_OI_CASCADE"].extend(
                simulate_managed_single_asset(
                    df=df, symbol=sym, strategy="CORE_OI_CASCADE", horizon="medium",
                    signal_indices=idxs, signal_sides=sides, signal_strengths=strengths,
                    vol_series=vol240 * 3.8, tp_mult=2.4, sl_mult=1.6, max_hold_mins=135,
                    min_tp=0.0055, cooldown_mins=45, enable_be_trail=enable_be_trail,
                    target_risk_per_trade=0.0055, min_notional_w=0.10, max_notional_w=0.28,
                    extra_delay_bars=extra_delay_bars, meta_feature_df=f,
                )
            )

        # --- Sleeve 2: CORE_SPOT_WHALE ---
        wh_gate = in_win & (vr >= 0.88) & (vol240 >= 0.0012) & (v_surge >= 1.5) & (oi_chg15 >= -0.004)
        wh_long = wh_gate & (s_ofi15 >= 0.24) & (sp_div15 >= 0.06) & (p_ofi15 >= 0.03) & ((smart_z >= 0.7) | (top_z >= 0.9)) & (prem_z <= 1.1)
        wh_short = wh_gate & (s_ofi15 <= -0.24) & (sp_div15 <= -0.06) & (p_ofi15 <= -0.03) & ((smart_z <= -0.7) | (top_z <= -0.9)) & (prem_z >= -1.1)
        m_wh = wh_long | wh_short
        if np.any(m_wh):
            idxs = np.flatnonzero(m_wh)
            sides = np.where(wh_long[idxs], 1, -1)
            strengths = np.abs(s_ofi15[idxs]) * 3.0 + np.abs(smart_z[idxs]) * 0.7
            sleeves["CORE_SPOT_WHALE"].extend(
                simulate_managed_single_asset(
                    df=df, symbol=sym, strategy="CORE_SPOT_WHALE", horizon="long",
                    signal_indices=idxs, signal_sides=sides, signal_strengths=strengths,
                    vol_series=vol240 * 5.8, tp_mult=3.4, sl_mult=2.1, max_hold_mins=360,
                    min_tp=0.0085, cooldown_mins=90, enable_be_trail=enable_be_trail,
                    target_risk_per_trade=0.0040, min_notional_w=0.08, max_notional_w=0.20,
                    extra_delay_bars=extra_delay_bars, meta_feature_df=f,
                )
            )

    # --- Sleeve 3: CORE_PAIRS_NEUT (Market-Neutral Cointegrated Sector Pairs) ---
    win = 432
    for sym_a, sym_b in selected_pairs:
        df_a, df_b = data_map[sym_a], data_map[sym_b]
        la = np.log(df_a["close"].astype(np.float64))
        lb = np.log(df_b["close"].astype(np.float64))
        beta = (la.rolling(win, min_periods=win // 2).cov(lb) / (lb.rolling(win, min_periods=win // 2).var() + 1e-12)).clip(0.3, 3.0)
        beta_lag = beta.shift(1)
        roll_corr = la.diff(3).rolling(win, min_periods=win // 2).corr(lb.diff(3)).fillna(0.0)
        spread = la - beta_lag * lb
        sp_mean = spread.rolling(win, min_periods=win // 2).mean()
        sp_std = spread.rolling(win, min_periods=win // 2).std()
        z_series = ((spread - sp_mean) / (sp_std + 1e-12)).to_numpy(dtype=np.float64)
        exp_edge = np.abs(z_series) * sp_std.fillna(0.0).to_numpy() / (1.0 + beta_lag.fillna(1.0).to_numpy())
        valid_mask = (roll_corr.to_numpy() >= 0.55) & (exp_edge >= 0.0045)

        raw_pairs = simulate_pair_trades(
            df_a, df_b, sym_a, sym_b, "CORE_PAIRS_NEUT", "long",
            z_series, beta_lag.to_numpy(), entry_z=2.1, exit_z=0.15, stop_z=4.5,
            max_hold_mins=720, start_idx=start_idx, end_idx=end_idx, cooldown_mins=90, valid_mask=valid_mask,
        )
        for tr in raw_pairs:
            sleeves["CORE_PAIRS_NEUT"].append(
                ManagedTrade(
                    strategy=tr.strategy, horizon=tr.horizon, symbol=tr.symbol, side=tr.side,
                    entry_ts=tr.entry_ts, exit_ts=tr.exit_ts, entry_idx=tr.entry_idx, exit_idx=tr.exit_idx,
                    gross_ret=tr.gross_ret, funding_pnl=tr.funding_pnl, hold_mins=tr.hold_mins,
                    exit_reason=tr.exit_reason, vol_at_entry=tr.vol_at_entry, signal_strength=tr.signal_strength,
                    risk_weight=0.18, sl_dist=0.025, tp_dist=0.015, be_triggered=False,
                )
            )

    return sleeves


def evaluate_managed_portfolio(
    trades: list[ManagedTrade],
    period_days: float,
    cost_mode: str = "hybrid",
    max_concurrent: int = 6,
    max_gross_exposure: float = 1.35,
    daily_dd_kill_pct: float = -0.025,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[dict[str, Any], list[ManagedTrade]]:
    """Simulate realistic live portfolio with concurrency, gross exposure cap & daily DD circuit breaker."""
    if not trades:
        return {"n_trades": 0, "exp_bps": 0.0}, []

    ordered = sorted(trades, key=lambda t: (t.entry_ts, -t.signal_strength))
    accepted: list[ManagedTrade] = []
    active: list[ManagedTrade] = []
    daily_realized: dict[Any, float] = {}
    killed_days = set()

    def _ret(tr: ManagedTrade) -> float:
        if cost_mode == "hybrid":
            return tr.net_ret_hybrid()
        if cost_mode == "stress_20":
            return tr.net_ret_stress(20.0)
        return tr.net_ret(cost_mode)

    for tr in ordered:
        # Close finished trades and update daily realized PnL for circuit breaker
        still_open = []
        for a in active:
            if a.exit_ts <= tr.entry_ts:
                d_key = a.exit_ts.date()
                daily_realized[d_key] = daily_realized.get(d_key, 0.0) + _ret(a) * a.risk_weight
                if daily_realized[d_key] <= daily_dd_kill_pct:
                    killed_days.add(d_key)
            else:
                still_open.append(a)
        active = still_open

        # Circuit breaker: pause directional entries if today hit daily DD cap (market-neutral pairs allowed)
        if tr.entry_ts.date() in killed_days and tr.strategy != "CORE_PAIRS_NEUT":
            continue

        if len(active) >= max_concurrent:
            continue
        if any(a.symbol == tr.symbol for a in active):
            continue
        cur_exp = sum(a.risk_weight for a in active)
        if cur_exp + tr.risk_weight > max_gross_exposure:
            continue

        active.append(tr)
        accepted.append(tr)

    if not accepted:
        return {"n_trades": 0, "exp_bps": 0.0}, []

    rets = np.array([_ret(t) for t in accepted], dtype=np.float64)
    port_rets = np.array([_ret(t) * t.risk_weight for t in accepted], dtype=np.float64)
    holds = np.array([t.hold_mins for t in accepted], dtype=np.float64)

    exit_dates = pd.to_datetime([t.exit_ts for t in accepted]).floor("D")
    daily_pnl = pd.Series(port_rets, index=exit_dates).groupby(level=0).sum()
    full_days = pd.date_range(daily_pnl.index.min(), daily_pnl.index.max(), freq="D", tz=daily_pnl.index.tz)
    daily_pnl = daily_pnl.reindex(full_days, fill_value=0.0)

    equity = np.cumprod(1.0 + daily_pnl.to_numpy())
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
    max_dd_pct = float(np.min(dd) * 100.0)
    total_ret_pct = float((equity[-1] - 1.0) * 100.0)

    d_std = float(daily_pnl.std(ddof=1)) if len(daily_pnl) > 2 else 0.0
    sharpe = float((daily_pnl.mean() / d_std) * np.sqrt(365.0)) if d_std > 1e-12 else 0.0

    wins = port_rets[port_rets > 0]
    losses = port_rets[port_rets <= 0]
    pf = float(np.sum(wins) / abs(np.sum(losses))) if np.sum(losses) < -1e-12 else 99.0

    rng = np.random.default_rng(seed)
    n_t = len(rets)
    idx = rng.integers(0, n_t, size=(n_bootstrap, n_t))
    boot = rets[idx].mean(axis=1) * 10000.0
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    reasons = [t.exit_reason for t in accepted]
    metrics = {
        "n_trades": int(n_t),
        "trades_per_day": round(n_t / max(period_days, 1e-6), 2),
        "win_rate_pct": round(float(np.mean(rets > 0) * 100.0), 2),
        "exp_bps": round(float(rets.mean() * 10000.0), 2),
        "ci95_low_bps": round(ci_lo, 2),
        "ci95_high_bps": round(ci_hi, 2),
        "profit_factor": round(pf, 3),
        "total_ret_pct": round(total_ret_pct, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "sharpe_daily": round(sharpe, 2),
        "avg_hold_mins": round(float(holds.mean()), 1),
        "avg_risk_w_pct": round(float(np.mean([t.risk_weight for t in accepted]) * 100.0), 1),
        "circuit_breaker_days": len(killed_days),
        "exit_dist": {k: round(100.0 * reasons.count(k) / n_t, 1) for k in ["TP", "BE_TRAIL", "SL", "TIME"]},
    }
    return metrics, accepted

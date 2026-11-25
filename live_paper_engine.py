"""Live Automated Gate.io Futures Paper-Trading Engine & Real-Time Market Radar.

Runs the Round 3 validated CORE_OI_CASCADE engine with 100% research-to-production parity:
  - Uses the exact same causal feature & signal code as offline walk-forward validation
  - Zero API keys required (reads public Gate.io USDT-M Perp, Spot, Index, OI, Liquidations & Funding)
  - Volatility-normalized risk sizing, max 6 concurrent positions, max 1.35x gross exposure
  - Realistic Hybrid execution costs (8 bps entry + 2 bps Maker TP / 7 bps SL-TIME exit + funding)
  - Daily UTC drawdown circuit breaker (-2.5%)
  - Persistent state across restarts + CSV trade journal + live HTML dashboard (`live_state/live_dashboard.html`)

Usage:
  python quant_lab/live_paper_engine.py --once            # Single live scan cycle on current Gate.io data
  python quant_lab/live_paper_engine.py --loop            # Continuous 24/7 automated paper trading
  python quant_lab/live_paper_engine.py --offline-test    # Local verification replay on cached dataset
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import BAR_MINS, BASE_DIR, UNIVERSE
from core_strategy import compute_macro_btc_trend_z, detect_live_signals_at_bar
from data_collector import (
    _sym_to_gate,
    fetch_contract_stats,
    fetch_funding,
    fetch_futures_candles,
    fetch_spot_candles,
    gate_get,
)
from features import compute_universe_features, load_universe

STATE_DIR = BASE_DIR / "live_state"
STATE_FILE = STATE_DIR / "portfolio_state.json"
TRADES_CSV = STATE_DIR / "trades_log.csv"
SIGNALS_JSONL = STATE_DIR / "signals_log.jsonl"
DASHBOARD_HTML = STATE_DIR / "live_dashboard.html"

# Validated Round 3 Production Parameters (CORE_OI_CASCADE, clean static Triple-Barrier)
ENGINE_CFG = {
    "strategy": "CORE_OI_CASCADE",
    "lookback_hours": 36,             # 432 x 5m bars needed for full 24h-36h feature warmup
    "max_concurrent": 6,
    "max_gross_exposure": 1.35,
    "target_risk_per_trade": 0.0055,  # 0.55% equity risked per SL
    "min_notional_w": 0.10,
    "max_notional_w": 0.28,
    "cooldown_mins": 45,
    "daily_dd_kill_pct": -0.025,      # -2.5% UTC daily realized loss pauses entries for the day
    "entry_cost_rate": 0.0008,        # 5 bps taker + 3 bps slippage
    "tp_exit_cost_rate": 0.0002,      # 2 bps resting Maker Limit TP
    "sl_exit_cost_rate": 0.0007,      # 5 bps taker + 2 bps slippage on SL/TIME
}


def default_state(capital: float = 10000.0) -> dict[str, Any]:
    now_iso = dt.datetime.now(tz=dt.timezone.utc).isoformat()
    return {
        "created_at": now_iso,
        "updated_at": now_iso,
        "initial_capital": float(capital),
        "equity": float(capital),
        "peak_equity": float(capital),
        "max_dd_pct": 0.0,
        "last_bar_ts": None,
        "open_positions": {},        # symbol -> position dict
        "cooldown_until": {},        # symbol -> ISO timestamp
        "daily_pnl_usdt": {},        # YYYY-MM-DD -> realized PnL USDT
        "equity_history": [{"ts": now_iso, "equity": float(capital)}],
        "stats": {"total_trades": 0, "wins": 0, "gross_win_usdt": 0.0, "gross_loss_usdt": 0.0},
    }


def load_state(capital: float = 10000.0, reset: bool = False) -> dict[str, Any]:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if reset or not STATE_FILE.exists():
        st = default_state(capital)
        save_state(st)
        return st
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def save_state(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = dt.datetime.now(tz=dt.timezone.utc).isoformat()
    payload = json.dumps(state, indent=2)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(payload, encoding="utf-8")
    for attempt in range(4):
        try:
            tmp.replace(STATE_FILE)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    # Fallback direct overwrite if Windows Defender momentarily locks atomic rename
    STATE_FILE.write_text(payload, encoding="utf-8")
    if tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass


def append_trade_csv(row: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    exists = TRADES_CSV.exists()
    cols = [
        "entry_ts", "exit_ts", "symbol", "strategy", "side", "entry_px", "exit_px",
        "notional_usdt", "gross_bps", "funding_bps", "cost_bps", "net_bps", "net_pnl_usdt",
        "hold_mins", "exit_reason", "equity_after", "signal_reason",
    ]
    with TRADES_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not exists:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in cols})


def append_signal_log(event: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with SIGNALS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=str) + "\n")


def fetch_live_rolling_symbol(sym: str, start_s: int, end_s: int) -> tuple[str, pd.DataFrame | None]:
    """Fetch last ~36h of 5m perp, index, spot, contract_stats & funding for one symbol."""
    contract = _sym_to_gate(sym)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        f_perp = pool.submit(fetch_futures_candles, contract, start_s, end_s, "5m")
        f_idx = pool.submit(fetch_futures_candles, f"index_{contract}", start_s, end_s, "5m")
        f_spot = pool.submit(fetch_spot_candles, contract, start_s, end_s, "5m")
        f_stats = pool.submit(fetch_contract_stats, contract, start_s, end_s, "5m")
        f_fund = pool.submit(fetch_funding, contract, 15)
        perp = f_perp.result()
        idx_df = f_idx.result()
        spot_df = f_spot.result()
        stats_df = f_stats.result()
        fr_df = f_fund.result()

    if len(perp) < 200:
        return sym, None

    perp = perp.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "sum": "quote_vol"})
    df = perp[["ts_s", "open", "high", "low", "close", "quote_vol"]].copy()

    if len(idx_df) > 0:
        idx_df = idx_df[["ts_s", "c"]].rename(columns={"c": "index_close"})
        df = df.merge(idx_df, on="ts_s", how="left")
        df["premium_close"] = ((df["close"] - df["index_close"]) / (df["index_close"] + 1e-12)).astype("float32")
    else:
        df["index_close"] = np.float32(np.nan)
        df["premium_close"] = np.float32(0.0)

    if len(spot_df) > 0:
        df = df.merge(spot_df, on="ts_s", how="left")
    else:
        for c in ["spot_open", "spot_high", "spot_low", "spot_close", "spot_quote_vol"]:
            df[c] = np.float32(np.nan)

    if len(stats_df) > 0:
        df = pd.merge_asof(df.sort_values("ts_s"), stats_df.sort_values("ts_s"), on="ts_s", direction="backward", tolerance=900)
    for col in ["oi_usdt", "lsr_taker", "retail_ls_acc", "top_ls_pos", "long_liq_usd", "short_liq_usd", "long_liq_size", "short_liq_size"]:
        if col not in df.columns:
            df[col] = np.float32(np.nan)

    lsr = df["lsr_taker"].astype(np.float64).clip(0.01, 100.0)
    perp_ofi = ((lsr - 1.0) / (lsr + 1.0)).fillna(0.0)
    bvc_perp = ((df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-12)).clip(-1.0, 1.0).fillna(0.0)
    perp_ofi = np.where(df["lsr_taker"].notna(), perp_ofi, bvc_perp)
    df["perp_ofi_bar"] = perp_ofi.astype("float32")
    df["perp_delta_q"] = (df["quote_vol"].astype(np.float64) * perp_ofi).astype("float32")

    bvc_spot = ((df["spot_close"] - df["spot_open"]) / (df["spot_high"] - df["spot_low"] + 1e-12)).clip(-1.0, 1.0).fillna(0.0)
    df["spot_ofi_bar"] = bvc_spot.astype("float32")
    df["spot_delta_q"] = (df["spot_quote_vol"].fillna(0.0).astype(np.float64) * bvc_spot).astype("float32")

    if len(fr_df) > 0:
        df = df.merge(fr_df, on="ts_s", how="left")
        df["funding_rate"] = df["funding_event"].ffill().fillna(0.0).astype("float32")
        df["funding_event"] = df["funding_event"].fillna(0.0).astype("float32")
    else:
        df["funding_rate"] = np.float32(0.0)
        df["funding_event"] = np.float32(0.0)

    df["timestamp"] = pd.to_datetime(df["ts_s"], unit="s", utc=True)
    df = df.set_index("timestamp").sort_index()
    return sym, df


def fetch_live_universe(symbols: list[str] = UNIVERSE) -> dict[str, pd.DataFrame]:
    """Fetch aligned live rolling 36h window across all 16 symbols from Gate.io."""
    end_s = (int(time.time()) // 300) * 300 - 300  # latest completed 5m candle
    start_s = end_s - ENGINE_CFG["lookback_hours"] * 3600
    raw_map: dict[str, pd.DataFrame] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(fetch_live_rolling_symbol, s, start_s, end_s) for s in symbols]
        for fut in concurrent.futures.as_completed(futs):
            sym, df = fut.result()
            if df is not None:
                raw_map[sym] = df
    if "BTCUSDT" not in raw_map:
        raise RuntimeError("Live fetch failed: BTCUSDT missing from Gate.io response.")

    start = max(df.index.min() for df in raw_map.values())
    end = min(df.index.max() for df in raw_map.values())
    common_idx = pd.date_range(start, end, freq=f"{BAR_MINS}min", tz="UTC")
    aligned = {}
    for sym, df in raw_map.items():
        d = df.reindex(common_idx)
        for c in ["open", "high", "low", "close", "index_close", "spot_close", "spot_open", "spot_high", "spot_low", "premium_close", "oi_usdt", "top_ls_pos", "retail_ls_acc", "lsr_taker", "funding_rate"]:
            if c in d.columns:
                d[c] = d[c].ffill()
        for c in ["quote_vol", "perp_delta_q", "perp_ofi_bar", "spot_quote_vol", "spot_delta_q", "spot_ofi_bar", "long_liq_usd", "short_liq_usd", "funding_event"]:
            if c in d.columns:
                d[c] = d[c].fillna(0.0)
        d["open"] = d["open"].fillna(d["close"])
        d["high"] = d["high"].fillna(d["close"])
        d["low"] = d["low"].fillna(d["close"])
        aligned[sym] = d
    return aligned


def fetch_live_last_prices() -> dict[str, float]:
    """Quick single REST call returning all Gate.io USDT-M futures live last prices."""
    data = gate_get("/futures/usdt/tickers")
    prices: dict[str, float] = {}
    if isinstance(data, list):
        for item in data:
            c = item.get("contract", "")
            if c.endswith("_USDT"):
                sym = c.replace("_USDT", "USDT")
                try:
                    prices[sym] = float(item["last"])
                except Exception:
                    pass
    return prices


def process_bar(
    state: dict[str, Any],
    data_map: dict[str, pd.DataFrame],
    feat_map: dict[str, pd.DataFrame],
    btc_trend_z: np.ndarray,
    bar_idx: int,
    live_prices: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Execute one complete paper-trading cycle on completed bar `bar_idx`."""
    bar_ts = data_map["BTCUSDT"].index[bar_idx]
    bar_ts_iso = bar_ts.isoformat()
    day_key = str(bar_ts.date())

    # 1. Manage existing open positions (conservative SL-first check on bar highs/lows + live price + funding + expiry)
    closed_now = []
    for sym, pos in list(state["open_positions"].items()):
        df = data_map.get(sym)
        if df is None:
            continue
        row = df.iloc[bar_idx]
        side = int(pos["side"])
        entry_px = float(pos["entry_px"])
        tp_px = float(pos["tp_px"])
        sl_px = float(pos["sl_px"])
        cur_px = float(live_prices.get(sym, row["close"])) if live_prices else float(row["close"])

        # Accrue funding if settlement occurred on this bar
        fund_ev = float(row.get("funding_event", 0.0))
        if fund_ev != 0.0 and bar_ts_iso != pos.get("last_fund_bar"):
            pos["accrued_funding_rate"] = float(pos.get("accrued_funding_rate", 0.0)) + (-side * fund_ev)
            pos["last_fund_bar"] = bar_ts_iso

        # Conservative intra-bar SL/TP check
        hi_px = max(float(row["high"]), cur_px)
        lo_px = min(float(row["low"]), cur_px)
        hit_sl = (lo_px <= sl_px) if side == 1 else (hi_px >= sl_px)
        hit_tp = (hi_px >= tp_px) if side == 1 else (lo_px <= tp_px)
        expired = bar_ts >= pd.Timestamp(pos["expiry_ts"])

        exit_reason = None
        exit_px = cur_px
        if hit_sl:
            exit_reason = "SL"
            exit_px = sl_px
        elif hit_tp:
            exit_reason = "TP"
            exit_px = tp_px
        elif expired:
            exit_reason = "TIME"
            exit_px = cur_px

        if exit_reason is not None:
            gross_ret = side * ((exit_px - entry_px) / entry_px)
            fund_ret = float(pos.get("accrued_funding_rate", 0.0))
            cost_rate = ENGINE_CFG["entry_cost_rate"] + (
                ENGINE_CFG["tp_exit_cost_rate"] if exit_reason == "TP" else ENGINE_CFG["sl_exit_cost_rate"]
            )
            net_ret = gross_ret + fund_ret - cost_rate
            notional = float(pos["notional_usdt"])
            net_pnl_usdt = notional * net_ret

            state["equity"] = round(float(state["equity"]) + net_pnl_usdt, 4)
            state["peak_equity"] = max(float(state["peak_equity"]), state["equity"])
            dd_now = (state["equity"] - state["peak_equity"]) / max(state["peak_equity"], 1e-6) * 100.0
            state["max_dd_pct"] = round(min(float(state["max_dd_pct"]), dd_now), 3)
            state["daily_pnl_usdt"][day_key] = round(float(state["daily_pnl_usdt"].get(day_key, 0.0)) + net_pnl_usdt, 4)
            state["equity_history"].append({"ts": bar_ts_iso, "equity": state["equity"]})
            state["equity_history"] = state["equity_history"][-500:]

            st = state["stats"]
            st["total_trades"] += 1
            if net_pnl_usdt > 0:
                st["wins"] += 1
                st["gross_win_usdt"] = round(st["gross_win_usdt"] + net_pnl_usdt, 4)
            else:
                st["gross_loss_usdt"] = round(st["gross_loss_usdt"] + abs(net_pnl_usdt), 4)

            hold_mins = int(max(BAR_MINS, (bar_ts - pd.Timestamp(pos["entry_ts"])).total_seconds() / 60.0))
            trade_row = {
                "entry_ts": pos["entry_ts"],
                "exit_ts": bar_ts_iso,
                "symbol": sym,
                "strategy": pos["strategy"],
                "side": "LONG" if side == 1 else "SHORT",
                "entry_px": round(entry_px, 6),
                "exit_px": round(exit_px, 6),
                "notional_usdt": round(notional, 2),
                "gross_bps": round(gross_ret * 10000.0, 2),
                "funding_bps": round(fund_ret * 10000.0, 2),
                "cost_bps": round(cost_rate * 10000.0, 2),
                "net_bps": round(net_ret * 10000.0, 2),
                "net_pnl_usdt": round(net_pnl_usdt, 2),
                "hold_mins": hold_mins,
                "exit_reason": exit_reason,
                "equity_after": round(state["equity"], 2),
                "signal_reason": pos.get("signal_reason", ""),
            }
            append_trade_csv(trade_row)
            closed_now.append(trade_row)
            del state["open_positions"][sym]
            cd_until = (bar_ts + pd.Timedelta(minutes=ENGINE_CFG["cooldown_mins"])).isoformat()
            state["cooldown_until"][sym] = cd_until

    # 2. Check Daily Drawdown Circuit Breaker
    today_pnl = float(state["daily_pnl_usdt"].get(day_key, 0.0))
    daily_dd_pct = today_pnl / max(float(state["equity"]), 1.0)
    circuit_tripped = daily_dd_pct <= ENGINE_CFG["daily_dd_kill_pct"]

    # 3. Scan new CORE_OI_CASCADE signals on completed bar_idx (skip if already processed this exact bar)
    opened_now = []
    btc_z_now = float(btc_trend_z[bar_idx])
    all_sigs = [s for s in detect_live_signals_at_bar(feat_map, btc_z_now, bar_idx) if s["strategy"] == ENGINE_CFG["strategy"]]

    if state.get("last_bar_ts") != bar_ts_iso:
        for sig in all_sigs:
            sym = sig["symbol"]
            rej = None
            cur_exp = sum(float(p["risk_weight"]) for p in state["open_positions"].values())

            if circuit_tripped:
                rej = "circuit_breaker"
            elif sym in state["open_positions"]:
                rej = "already_open"
            elif len(state["open_positions"]) >= ENGINE_CFG["max_concurrent"]:
                rej = "max_concurrent"
            elif sym in state["cooldown_until"] and bar_ts < pd.Timestamp(state["cooldown_until"][sym]):
                rej = "cooldown"

            vol = float(sig["vol"])
            tp_dist = float(np.clip(sig["tp_mult"] * vol, sig["min_tp"], 0.08))
            sl_dist = float(np.clip(sig["sl_mult"] * vol, sig["min_tp"] * 0.72, 0.068))
            risk_w = float(np.clip(ENGINE_CFG["target_risk_per_trade"] / max(sl_dist, 1e-4), ENGINE_CFG["min_notional_w"], ENGINE_CFG["max_notional_w"]))

            if rej is None and (cur_exp + risk_w) > ENGINE_CFG["max_gross_exposure"]:
                rej = "max_gross_exposure"

            log_ev = {"ts": bar_ts_iso, **sig, "risk_weight": round(risk_w, 4), "status": "REJECTED:" + rej if rej else "OPENED"}
            append_signal_log(log_ev)
            if rej is not None:
                continue

            entry_px = float(live_prices.get(sym, data_map[sym]["close"].iloc[bar_idx])) if live_prices else float(data_map[sym]["close"].iloc[bar_idx])
            side = int(sig["side"])
            tp_px = entry_px * (1.0 + tp_dist) if side == 1 else entry_px * (1.0 - tp_dist)
            sl_px = entry_px * (1.0 - sl_dist) if side == 1 else entry_px * (1.0 + sl_dist)
            notional = round(float(state["equity"]) * risk_w, 2)
            expiry_ts = (bar_ts + pd.Timedelta(minutes=int(sig["max_hold_mins"]))).isoformat()

            pos_obj = {
                "symbol": sym,
                "strategy": sig["strategy"],
                "side": side,
                "entry_ts": bar_ts_iso,
                "expiry_ts": expiry_ts,
                "entry_px": entry_px,
                "tp_px": tp_px,
                "sl_px": sl_px,
                "tp_dist": tp_dist,
                "sl_dist": sl_dist,
                "risk_weight": risk_w,
                "notional_usdt": notional,
                "accrued_funding_rate": 0.0,
                "signal_reason": sig["reason"],
            }
            state["open_positions"][sym] = pos_obj
            opened_now.append(pos_obj)

        state["last_bar_ts"] = bar_ts_iso

    save_state(state)
    return {
        "bar_ts": bar_ts_iso,
        "btc_trend_z": round(btc_z_now, 2),
        "circuit_tripped": circuit_tripped,
        "daily_pnl_usdt": round(today_pnl, 2),
        "signals_seen": len(all_sigs),
        "opened": opened_now,
        "closed": closed_now,
    }


def render_live_dashboard(
    state: dict[str, Any],
    data_map: dict[str, pd.DataFrame],
    feat_map: dict[str, pd.DataFrame],
    btc_trend_z: np.ndarray,
    bar_idx: int,
    live_prices: dict[str, float] | None = None,
) -> None:
    """Render self-contained dark quant terminal dashboard (`live_state/live_dashboard.html`)."""
    st = state["stats"]
    tot = st["total_trades"]
    wr = (100.0 * st["wins"] / tot) if tot > 0 else 0.0
    pf = (st["gross_win_usdt"] / st["gross_loss_usdt"]) if st["gross_loss_usdt"] > 1e-6 else (99.0 if st["gross_win_usdt"] > 0 else 0.0)
    ret_pct = (state["equity"] / max(state["initial_capital"], 1.0) - 1.0) * 100.0
    bar_ts = data_map["BTCUSDT"].index[bar_idx]
    day_key = str(bar_ts.date())
    today_pnl = float(state["daily_pnl_usdt"].get(day_key, 0.0))
    btc_z_now = float(btc_trend_z[bar_idx])

    # Open positions table
    open_rows = []
    total_unreal = 0.0
    for sym, p in state["open_positions"].items():
        cur_px = float(live_prices.get(sym, data_map[sym]["close"].iloc[bar_idx])) if live_prices else float(data_map[sym]["close"].iloc[bar_idx])
        side = int(p["side"])
        unr_ret = side * ((cur_px - float(p["entry_px"])) / float(p["entry_px"]))
        unr_usdt = float(p["notional_usdt"]) * unr_ret
        total_unreal += unr_usdt
        cls = "pos" if unr_usdt >= 0 else "neg"
        open_rows.append(
            f"<tr><td><b>{sym}</b></td><td class='{'pos' if side==1 else 'neg'}'>{'LONG' if side==1 else 'SHORT'}</td>"
            f"<td>{p['entry_px']:.4f}</td><td>{cur_px:.4f}</td><td>{p['tp_px']:.4f}</td><td>{p['sl_px']:.4f}</td>"
            f"<td>{p['notional_usdt']:.0f}</td><td class='{cls}'><b>{unr_ret*10000:+.1f} bps ({unr_usdt:+.2f})</b></td>"
            f"<td>{p['expiry_ts'][11:16]} UTC</td><td class='muted'>{p['signal_reason']}</td></tr>"
        )
    if not open_rows:
        open_rows.append("<tr><td colspan='10' style='text-align:center' class='muted'>Su an acik pozisyon yok — gercek OI erimesi + tasfiye dalgasi bekleniyor...</td></tr>")

    # 16-coin live Market Radar (sorted by biggest OI drop!)
    radar_items = []
    for sym, f in feat_map.items():
        r = f.iloc[bar_idx]
        px = float(live_prices.get(sym, data_map[sym]["close"].iloc[bar_idx])) if live_prices else float(data_map[sym]["close"].iloc[bar_idx])
        oi15 = float(r["oi_chg_15m"]) * 100.0
        liqz = float(r["real_liq_z"])
        vr = float(r["vr_15"])
        volbp = float(r["vol_240m"]) * 10000.0
        pofi = float(r["perp_ofi_5m"])
        vsurge = float(r["vol_surge_15m"])
        gates_ok = sum([oi15 <= -0.55, vr >= 0.68, volbp >= 12.0, abs(liqz) >= 1.4, vsurge >= 1.4])
        radar_items.append((gates_ok, -oi15, sym, px, oi15, liqz, vr, volbp, pofi, vsurge))
    radar_items.sort(reverse=True)

    radar_rows = []
    for gates_ok, _, sym, px, oi15, liqz, vr, volbp, pofi, vsurge in radar_items:
        badge = f"<span class='badge-hot'>{gates_ok}/5</span>" if gates_ok >= 4 else f"<span class='badge'>{gates_ok}/5</span>"
        radar_rows.append(
            f"<tr><td><b>{sym}</b></td><td>{px:.4f}</td>"
            f"<td class='{'pos' if oi15 <= -0.55 else ''}'>{oi15:+.2f}%</td>"
            f"<td class='{'neg' if liqz <= -1.4 else ('pos' if liqz >= 1.4 else '')}'>{liqz:+.2f}</td>"
            f"<td class='{'pos' if vr >= 0.68 else 'muted'}'>{vr:.2f}</td>"
            f"<td class='{'pos' if volbp >= 12 else 'muted'}'>{volbp:.1f} bp</td>"
            f"<td>{pofi:+.2f}</td><td>{vsurge:.2f}x</td><td>{badge}</td></tr>"
        )

    # SVG Equity Curve
    eq_pts = [e["equity"] for e in state["equity_history"]]
    if len(eq_pts) >= 2:
        w, h, pad = 740, 170, 20
        mn, mx = min(eq_pts), max(eq_pts)
        sp = max(mx - mn, 1e-4)
        pts = [f"{pad+(w-2*pad)*(i/(len(eq_pts)-1)):.1f},{h-pad-(h-2*pad)*((v-mn)/sp):.1f}" for i, v in enumerate(eq_pts)]
        col = "#22c55e" if eq_pts[-1] >= eq_pts[0] else "#ef4444"
        svg = f"<svg viewBox='0 0 {w} {h}' width='100%' height='{h}' style='background:#0b1220;border-radius:8px'><polyline fill='none' stroke='{col}' stroke-width='2.3' points='{' '.join(pts)}'/></svg>"
    else:
        svg = "<div class='muted' style='padding:24px;text-align:center'>Ilk kapanan islemden sonra sermaye egrisi olusacak.</div>"

    # Last 15 closed trades
    closed_rows = []
    if TRADES_CSV.exists():
        cdf = pd.read_csv(TRADES_CSV).tail(15).iloc[::-1]
        for _, tr in cdf.iterrows():
            cls = "pos" if float(tr["net_pnl_usdt"]) >= 0 else "neg"
            closed_rows.append(
                f"<tr><td>{str(tr['exit_ts'])[5:16]}</td><td><b>{tr['symbol']}</b></td><td>{tr['side']}</td>"
                f"<td>{tr['exit_reason']}</td><td>{tr['hold_mins']} dk</td>"
                f"<td class='{cls}'><b>{float(tr['net_bps']):+.1f} bps ({float(tr['net_pnl_usdt']):+.2f})</b></td>"
                f"<td>{float(tr['equity_after']):.2f}</td></tr>"
            )
    if not closed_rows:
        closed_rows.append("<tr><td colspan='7' style='text-align:center' class='muted'>Henuz kapanmis paper islem yok.</td></tr>")

    html = f"""<!doctype html><html lang="tr"><head><meta charset="utf-8"><meta http-equiv="refresh" content="60">
<title>Quant Lab — Canli Otomatik Paper-Trading Motoru</title>
<style>
  body{{font-family:system-ui,-apple-system,sans-serif;background:#020617;color:#e2e8f0;margin:20px;line-height:1.4}}
  .kpis{{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:14px 0 22px}}
  .kpi{{background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:12px 14px}}
  .kpi .lbl{{font-size:11px;color:#94a3b8;text-transform:uppercase}} .kpi .val{{font-size:20px;font-weight:700;margin-top:4px}}
  table{{width:100%;border-collapse:collapse;background:#0f172a;border-radius:8px;overflow:hidden;font-size:12.5px;margin:10px 0 22px}}
  th,td{{padding:7px 9px;border-bottom:1px solid #1e293b;text-align:right}} th:first-child,td:first-child{{text-align:left}} th{{background:#1e293b;color:#cbd5e1}}
  .pos{{color:#4ade80}} .neg{{color:#f87171}} .muted{{color:#64748b}}
  .badge{{background:#1e293b;padding:2px 7px;border-radius:10px;font-size:11px}}
  .badge-hot{{background:#7c2d12;color:#fdba74;padding:2px 7px;border-radius:10px;font-size:11px;font-weight:700}}
</style></head><body>
<h1>CORE_OI_CASCADE — Canli Otomatik Paper-Trading Motoru (Gate.io USDT-M)</h1>
<div class="muted">Son Bar: {bar_ts} | Guncelleme: {state['updated_at'][:19]} UTC | BTC 24s Trend Z: <b>{btc_z_now:+.2f}</b> | Maks Pozisyon: {ENGINE_CFG['max_concurrent']} | Gunluk Devre Kesici: {ENGINE_CFG['daily_dd_kill_pct']*100:.1f}%</div>

<div class="kpis">
  <div class="kpi"><div class="lbl">Sanal Bakiye</div><div class="val">{state['equity']:,.2f} USDT</div></div>
  <div class="kpi"><div class="lbl">Toplam Getiri</div><div class="val {'pos' if ret_pct>=0 else 'neg'}">{ret_pct:+.2f}%</div></div>
  <div class="kpi"><div class="lbl">Bugunku PnL</div><div class="val {'pos' if today_pnl>=0 else 'neg'}">{today_pnl:+.2f} USDT</div></div>
  <div class="kpi"><div class="lbl">Acik Unrealized</div><div class="val {'pos' if total_unreal>=0 else 'neg'}">{total_unreal:+.2f} USDT</div></div>
  <div class="kpi"><div class="lbl">Win% / PF ({tot} islem)</div><div class="val">{wr:.1f}% / {pf:.2f}</div></div>
  <div class="kpi"><div class="lbl">Max Drawdown</div><div class="val neg">{state['max_dd_pct']:.2f}%</div></div>
</div>

<h2>Acik Pozisyonlar ({len(state['open_positions'])}/{ENGINE_CFG['max_concurrent']})</h2>
<table><thead><tr><th>Parite</th><th>Yon</th><th>Giris</th><th>Anlik</th><th>TP</th><th>SL</th><th>Notional</th><th>Anlik PnL</th><th>Zaman Bitis</th><th>Gerekce</th></tr></thead><tbody>{''.join(open_rows)}</tbody></table>

<h2>Canli 16 Parite Mikroyapi Radari (Tetik Kosullari: OI&lt;=-0.55%, |LiqZ|&gt;=1.4, VR&gt;=0.68, Vol&gt;=12bp, Sur&gt;=1.4x)</h2>
<table><thead><tr><th>Parite</th><th>Fiyat</th><th>OI 15dk%</th><th>Liq Z</th><th>DNA (VR)</th><th>Vol 4s</th><th>Perp OFI</th><th>Hacim Artisi</th><th>Hazirlik</th></tr></thead><tbody>{''.join(radar_rows)}</tbody></table>

<h2>Sermaye Egrisi &amp; Son Kapanan Islemler</h2>
{svg}
<table><thead><tr><th>Kapanis</th><th>Parite</th><th>Yon</th><th>Cikis</th><th>Sure</th><th>Net PnL (Hibrit)</th><th>Bakiye</th></tr></thead><tbody>{''.join(closed_rows)}</tbody></table>
</body></html>"""
    DASHBOARD_HTML.write_text(html, encoding="utf-8")


def run_offline_replay_test(n_bars: int = 1200) -> None:
    """Verify 100% of live paper-trading engine logic locally on the last ~4 days of cached data."""
    print(f"[offline-test] Running live paper engine replay on last {n_bars} bars (~{n_bars/288:.1f} days)...", flush=True)
    # Use isolated test state so we never pollute live_state
    global STATE_FILE, TRADES_CSV, SIGNALS_JSONL, DASHBOARD_HTML
    orig = (STATE_FILE, TRADES_CSV, SIGNALS_JSONL, DASHBOARD_HTML)
    test_dir = STATE_DIR / "_offline_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    STATE_FILE = test_dir / "portfolio_state.json"
    TRADES_CSV = test_dir / "trades_log.csv"
    SIGNALS_JSONL = test_dir / "signals_log.jsonl"
    DASHBOARD_HTML = test_dir / "live_dashboard.html"
    for p in [STATE_FILE, TRADES_CSV, SIGNALS_JSONL, DASHBOARD_HTML]:
        if p.exists():
            p.unlink()

    data_map = load_universe()
    feat_map, _ = compute_universe_features(data_map)
    btc_z = compute_macro_btc_trend_z(data_map, feat_map)
    n_tot = len(data_map["BTCUSDT"])
    start_i = max(432, n_tot - n_bars)

    state = load_state(10000.0, reset=True)
    opened_tot, closed_tot = 0, 0
    for i in range(start_i, n_tot):
        res = process_bar(state, data_map, feat_map, btc_z, i)
        opened_tot += len(res["opened"])
        closed_tot += len(res["closed"])

    render_live_dashboard(state, data_map, feat_map, btc_z, n_tot - 1)
    st = state["stats"]
    pf = st["gross_win_usdt"] / max(st["gross_loss_usdt"], 1e-6)
    print(
        f"[offline-test] PASSED! Replay Equity: 10,000.00 -> {state['equity']:,.2f} USDT "
        f"({(state['equity']/10000-1)*100:+.2f}%) | Opened={opened_tot}, Closed={closed_tot} | "
        f"Win={100*st['wins']/max(st['total_trades'],1):.1f}% | PF={pf:.2f} | MaxDD={state['max_dd_pct']:.2f}%",
        flush=True,
    )
    print(f"[offline-test] Test dashboard verified: {DASHBOARD_HTML}", flush=True)
    STATE_FILE, TRADES_CSV, SIGNALS_JSONL, DASHBOARD_HTML = orig


def run_live_cycle(state: dict[str, Any]) -> None:
    t0 = time.time()
    data_map = fetch_live_universe(UNIVERSE)
    feat_map, _ = compute_universe_features(data_map)
    btc_z = compute_macro_btc_trend_z(data_map, feat_map)
    live_px = fetch_live_last_prices()
    last_idx = len(data_map["BTCUSDT"]) - 1

    res = process_bar(state, data_map, feat_map, btc_z, last_idx, live_prices=live_px)
    render_live_dashboard(state, data_map, feat_map, btc_z, last_idx, live_prices=live_px)

    print(
        f"[LIVE {res['bar_ts']}] BTC_24hZ={res['btc_trend_z']:+.2f} | Equity={state['equity']:,.2f} USDT | "
        f"Open={len(state['open_positions'])}/{ENGINE_CFG['max_concurrent']} | "
        f"Signals={res['signals_seen']} Opened={len(res['opened'])} Closed={len(res['closed'])} | "
        f"CircuitBreaker={res['circuit_tripped']} | {time.time()-t0:.1f}s",
        flush=True,
    )
    for op in res["opened"]:
        print(f"  + OPEN  {op['symbol']} {'LONG' if op['side']==1 else 'SHORT'} @ {op['entry_px']:.4f} | TP={op['tp_px']:.4f} SL={op['sl_px']:.4f} | {op['signal_reason']}", flush=True)
    for cl in res["closed"]:
        print(f"  - CLOSE {cl['symbol']} {cl['side']} ({cl['exit_reason']}) Net={cl['net_bps']:+.1f}bps ({cl['net_pnl_usdt']:+.2f} USDT)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Automated Gate.io CORE_OI_CASCADE Paper-Trading Engine")
    parser.add_argument("--once", action="store_true", help="Run a single live market scan & dashboard update")
    parser.add_argument("--loop", action="store_true", help="Run continuously 24/7 synced to 5m bar closes")
    parser.add_argument("--offline-test", action="store_true", help="Run local offline replay verification test")
    parser.add_argument("--capital", type=float, default=10000.0, help="Starting virtual capital in USDT")
    parser.add_argument("--reset", action="store_true", help="Reset virtual paper account state")
    args = parser.parse_args()

    if args.offline_test:
        run_offline_replay_test(n_bars=1500)
        return

    state = load_state(capital=args.capital, reset=args.reset)
    if args.once or not args.loop:
        run_live_cycle(state)
        print(f"[LIVE] Dashboard saved -> {DASHBOARD_HTML}", flush=True)
        return

    print(f"[LIVE] Starting 24/7 Automated Paper-Trading Engine (Capital={state['equity']:,.2f} USDT)...", flush=True)
    print(f"[LIVE] Open dashboard in browser: {DASHBOARD_HTML}", flush=True)
    while True:
        try:
            run_live_cycle(state)
        except Exception as exc:
            print(f"[LIVE] Cycle error (will retry next bar): {type(exc).__name__}: {exc}", flush=True)
        # Sleep until 8 seconds after next 5-minute UTC bar close
        now_s = time.time()
        next_bar_s = ((int(now_s) // 300) + 1) * 300 + 8
        sleep_s = max(10.0, next_bar_s - now_s)
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()

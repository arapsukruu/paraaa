"""24/7 Live Automated Demo Trading Bot + Real-Time Local Web Dashboard (5,000 USDT).

Runs the Round 3 validated CORE_OI_CASCADE strategy live on Gate.io public USDT-M Futures data:
  - Fast Price & TP/SL Loop (every 5s): live prices across all 16 pairs, real-time unrealized PnL, instant TP/SL fills
  - Strategy Scan Loop (every completed 5m UTC bar + 6s): incremental OI, real liquidations, Spot/Perp OFI & Pair DNA
  - Built-in Local Web Server (http://127.0.0.1:8787): zero-flicker live dashboard with Pause/Resume, Manual Close, Scan Now & Reset

Usage:
  python quant_lab/demo_bot_server.py --capital 5000 --port 8787
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import json
import os
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
from features import compute_universe_features
from telegram_notifier import TelegramNotifier, load_local_env

ENV_PATHS = [BASE_DIR / ".env", BASE_DIR.parent / ".env"]
load_local_env(ENV_PATHS)

STATE_DIR = BASE_DIR / "live_state"
STATE_FILE = STATE_DIR / "demo_5000_state.json"
TRADES_CSV = STATE_DIR / "demo_5000_trades.csv"
EVENTS_JSONL = STATE_DIR / "demo_5000_events.jsonl"
TEMPLATE_HTML = BASE_DIR / "templates" / "dashboard.html"

BOT_CFG = {
    "strategy": "CORE_OI_CASCADE",
    "lookback_hours": 36,             # 432 x 5m bars for 24h-36h rolling features
    "max_concurrent": 6,
    "max_gross_exposure": 1.35,
    "target_risk_per_trade": 0.0055,  # 0.55% equity risked per SL (~27.5 USDT on 5,000 USDT)
    "min_notional_w": 0.10,           # ~500 USDT min position on 5,000 USDT
    "max_notional_w": 0.28,           # ~1,400 USDT max position on 5,000 USDT
    "cooldown_mins": 45,
    "daily_dd_kill_pct": -0.025,      # -2.5% UTC daily loss circuit breaker (-125 USDT)
    "entry_cost_rate": 0.0008,        # 5 bps taker + 3 bps slippage
    "tp_exit_cost_rate": 0.0002,      # 2 bps Maker Limit TP
    "sl_exit_cost_rate": 0.0007,      # 5 bps taker + 2 bps slippage on SL / TIME / MANUAL
    "price_poll_sec": 5.0,
}


def utc_now_iso() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).isoformat()


def fetch_symbol_window(sym: str, start_s: int, end_s: int) -> tuple[str, pd.DataFrame | None]:
    contract = _sym_to_gate(sym)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        f_perp = pool.submit(fetch_futures_candles, contract, start_s, end_s, "5m")
        f_idx = pool.submit(fetch_futures_candles, f"index_{contract}", start_s, end_s, "5m")
        f_spot = pool.submit(fetch_spot_candles, contract, start_s, end_s, "5m")
        f_stats = pool.submit(fetch_contract_stats, contract, start_s, end_s, "5m")
        f_fund = pool.submit(fetch_funding, contract, 12)
        perp = f_perp.result()
        idx_df = f_idx.result()
        spot_df = f_spot.result()
        stats_df = f_stats.result()
        fr_df = f_fund.result()

    if len(perp) == 0:
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


def align_universe(raw_map: dict[str, pd.DataFrame], max_bars: int = 460) -> dict[str, pd.DataFrame]:
    start = max(df.index.min() for df in raw_map.values())
    end = min(df.index.max() for df in raw_map.values())
    common_idx = pd.date_range(start, end, freq=f"{BAR_MINS}min", tz="UTC")
    if len(common_idx) > max_bars:
        common_idx = common_idx[-max_bars:]

    aligned: dict[str, pd.DataFrame] = {}
    for sym, df in raw_map.items():
        d = df[~df.index.duplicated(keep="last")].reindex(common_idx)
        for c in ["open", "high", "low", "close", "index_close", "spot_close", "spot_open", "spot_high", "spot_low", "premium_close", "oi_usdt", "top_ls_pos", "retail_ls_acc", "lsr_taker", "funding_rate"]:
            if c in d.columns:
                d[c] = d[c].ffill()
        for c in ["quote_vol", "perp_delta_q", "perp_ofi_bar", "spot_quote_vol", "spot_delta_q", "spot_ofi_bar", "long_liq_usd", "short_liq_usd", "long_liq_size", "short_liq_size", "funding_event"]:
            if c in d.columns:
                d[c] = d[c].fillna(0.0)
        d["open"] = d["open"].fillna(d["close"])
        d["high"] = d["high"].fillna(d["close"])
        d["low"] = d["low"].fillna(d["close"])
        aligned[sym] = d
    return aligned


def fetch_all_tickers() -> dict[str, float]:
    data = gate_get("/futures/usdt/tickers")
    out: dict[str, float] = {}
    if isinstance(data, list):
        for item in data:
            c = item.get("contract", "")
            if c.endswith("_USDT"):
                sym = c.replace("_USDT", "USDT")
                if sym in UNIVERSE:
                    try:
                        out[sym] = float(item["last"])
                    except Exception:
                        pass
    return out


class LiveDemoBot:
    def __init__(self, initial_capital: float = 5000.0, reset: bool = False) -> None:
        self.lock = threading.RLock()
        self.initial_capital = float(initial_capital)
        self.paused = False
        self.boot_status = "GEÇMİŞ YÜKLENİYOR (36 SAAT)..."
        self.scan_in_progress = False
        self.next_scan_epoch = 0.0
        self.last_price_ts = ""
        self.btc_trend_z_now = 0.0
        self.raw_map: dict[str, pd.DataFrame] = {}
        self.data_map: dict[str, pd.DataFrame] = {}
        self.feat_map: dict[str, pd.DataFrame] = {}
        self.live_prices: dict[str, float] = {}
        self.radar_snapshot: list[dict[str, Any]] = []
        self.events: list[dict[str, str]] = []
        self._stop_event = threading.Event()
        self._force_scan_event = threading.Event()
        self.tg = TelegramNotifier()
        self._boot_notified = False
        self._cb_notified_day: str | None = None

        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.state = self._load_or_init_state(reset=reset)
        tg_status = "AKTİF" if self.tg.enabled else "KAPALI (TELEGRAM_BOT_TOKEN bekleniyor)"
        self._log("SYS", f"Demo Bot başlatıldı (Başlangıç: {self.state['initial_capital']:,.2f} USDT, Bakiye: {self.state['equity']:,.2f} USDT | Telegram: {tg_status})")

    def _default_state(self, cap: float) -> dict[str, Any]:
        now = utc_now_iso()
        return {
            "created_at": now,
            "updated_at": now,
            "initial_capital": float(cap),
            "equity": float(cap),
            "peak_equity": float(cap),
            "max_dd_pct": 0.0,
            "last_bar_ts": None,
            "open_positions": {},
            "cooldown_until": {},
            "daily_pnl_usdt": {},
            "equity_history": [{"ts": now, "equity": float(cap)}],
            "stats": {"total_trades": 0, "wins": 0, "gross_win_usdt": 0.0, "gross_loss_usdt": 0.0},
        }

    def _load_or_init_state(self, reset: bool = False) -> dict[str, Any]:
        if reset or not STATE_FILE.exists():
            if reset and TRADES_CSV.exists():
                TRADES_CSV.unlink()
            st = self._default_state(self.initial_capital)
            self._save_state_unlocked(st)
            return st
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))

    def _save_state_unlocked(self, st: dict[str, Any] | None = None) -> None:
        target = st if st is not None else self.state
        target["updated_at"] = utc_now_iso()
        payload = json.dumps(target, indent=2)
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(payload, encoding="utf-8")
        for attempt in range(4):
            try:
                tmp.replace(STATE_FILE)
                return
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))
        STATE_FILE.write_text(payload, encoding="utf-8")
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

    def _log(self, tag: str, msg: str) -> None:
        ev = {"ts": utc_now_iso(), "tag": tag, "msg": msg}
        with self.lock:
            self.events.append(ev)
            self.events = self.events[-80:]
        try:
            with EVENTS_JSONL.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        except OSError:
            pass
        print(f"[{ev['ts'][11:19]} UTC][{tag}] {msg}", flush=True)

    def _append_trade_csv(self, row: dict[str, Any]) -> None:
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

    def _close_position_unlocked(self, sym: str, exit_px: float, exit_reason: str, now_ts: pd.Timestamp) -> dict[str, Any]:
        pos = self.state["open_positions"].pop(sym)
        side = int(pos["side"])
        entry_px = float(pos["entry_px"])
        gross_ret = side * ((exit_px - entry_px) / entry_px)
        fund_ret = float(pos.get("accrued_funding_rate", 0.0))
        cost_rate = BOT_CFG["entry_cost_rate"] + (
            BOT_CFG["tp_exit_cost_rate"] if exit_reason == "TP" else BOT_CFG["sl_exit_cost_rate"]
        )
        net_ret = gross_ret + fund_ret - cost_rate
        notional = float(pos["notional_usdt"])
        net_pnl_usdt = notional * net_ret

        self.state["equity"] = round(float(self.state["equity"]) + net_pnl_usdt, 4)
        self.state["peak_equity"] = max(float(self.state["peak_equity"]), self.state["equity"])
        dd_now = (self.state["equity"] - self.state["peak_equity"]) / max(self.state["peak_equity"], 1e-6) * 100.0
        self.state["max_dd_pct"] = round(min(float(self.state["max_dd_pct"]), dd_now), 3)

        day_key = str(now_ts.date())
        self.state["daily_pnl_usdt"][day_key] = round(float(self.state["daily_pnl_usdt"].get(day_key, 0.0)) + net_pnl_usdt, 4)
        self.state["equity_history"].append({"ts": now_ts.isoformat(), "equity": self.state["equity"]})
        self.state["equity_history"] = self.state["equity_history"][-400:]

        st = self.state["stats"]
        st["total_trades"] += 1
        if net_pnl_usdt > 0:
            st["wins"] += 1
            st["gross_win_usdt"] = round(st["gross_win_usdt"] + net_pnl_usdt, 4)
        else:
            st["gross_loss_usdt"] = round(st["gross_loss_usdt"] + abs(net_pnl_usdt), 4)

        hold_mins = int(max(1, (now_ts - pd.Timestamp(pos["entry_ts"])).total_seconds() / 60.0))
        cd_until = (now_ts + pd.Timedelta(minutes=BOT_CFG["cooldown_mins"])).isoformat()
        self.state["cooldown_until"][sym] = cd_until

        row = {
            "entry_ts": pos["entry_ts"],
            "exit_ts": now_ts.isoformat(),
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
            "equity_after": round(self.state["equity"], 2),
            "signal_reason": pos.get("signal_reason", ""),
        }
        self._append_trade_csv(row)
        self._save_state_unlocked()
        return row

    def check_realtime_exits(self) -> None:
        """Check open positions against live 5-second Gate.io prices for instant TP/SL/TIME closure."""
        tickers = fetch_all_tickers()
        if not tickers:
            return
        now_ts = pd.Timestamp.now(tz="UTC")
        closed_msgs = []

        with self.lock:
            self.live_prices.update(tickers)
            self.last_price_ts = now_ts.isoformat()

            for sym, pos in list(self.state["open_positions"].items()):
                px = self.live_prices.get(sym)
                if px is None:
                    continue
                side = int(pos["side"])
                tp_px = float(pos["tp_px"])
                sl_px = float(pos["sl_px"])
                expired = now_ts >= pd.Timestamp(pos["expiry_ts"])

                hit_sl = (px <= sl_px) if side == 1 else (px >= sl_px)
                hit_tp = (px >= tp_px) if side == 1 else (px <= tp_px)

                reason = None
                fill_px = px
                if hit_sl:
                    reason = "SL"
                    fill_px = sl_px
                elif hit_tp:
                    reason = "TP"
                    fill_px = tp_px
                elif expired:
                    reason = "TIME"
                    fill_px = px

                if reason is not None:
                    row = self._close_position_unlocked(sym, fill_px, reason, now_ts)
                    closed_msgs.append(row)

            # Keep radar prices ticking live too!
            for r in self.radar_snapshot:
                if r["symbol"] in self.live_prices:
                    r["price"] = self.live_prices[r["symbol"]]

        for row in closed_msgs:
            tag = "CLOSE" if row["net_pnl_usdt"] >= 0 else "LOSS"
            self._log(tag, f"{row['symbol']} {row['side']} kapandı ({row['exit_reason']}) @ {row['exit_px']} | Net: {row['net_bps']:+.1f} bps ({row['net_pnl_usdt']:+.2f} $) -> Bakiye: {row['equity_after']:,.2f} $")
            self._notify_trade_close(row)

    def _notify_trade_open(self, p: dict[str, Any]) -> None:
        side_txt = "🟢 LONG" if int(p["side"]) == 1 else "🔴 SHORT"
        msg = (
            f"<b>⚡ YENİ İŞLEM AÇILDI — {p['symbol']}</b>\n"
            f"Yön: <b>{side_txt}</b> | Büyüklük: <b>{p['notional_usdt']:,.0f} USDT</b>\n"
            f"Giriş: <code>{p['entry_px']:.5g}</code>\n"
            f"TP: <code>{p['tp_px']:.5g}</code> | SL: <code>{p['sl_px']:.5g}</code>\n"
            f"Süre Limiti: 135 dk\n"
            f"Gerekçe: <i>{p['signal_reason']}</i>"
        )
        self.tg.send_async(msg)

    def _notify_trade_close(self, row: dict[str, Any]) -> None:
        win = float(row["net_pnl_usdt"]) >= 0
        icon = "✅ KÂR" if win else "❌ ZARAR"
        st = self.state["stats"]
        wr = (100.0 * st["wins"] / st["total_trades"]) if st["total_trades"] > 0 else 0.0
        pf = (st["gross_win_usdt"] / st["gross_loss_usdt"]) if st["gross_loss_usdt"] > 1e-6 else 0.0
        msg = (
            f"<b>{icon} — {row['symbol']} ({row['side']}) KAPANDI [{row['exit_reason']}]</b>\n"
            f"Giriş: <code>{row['entry_px']}</code> → Çıkış: <code>{row['exit_px']}</code> ({row['hold_mins']} dk)\n"
            f"Net Sonuç: <b>{row['net_bps']:+.1f} bps ({row['net_pnl_usdt']:+.2f} USDT)</b>\n"
            f"Güncel Bakiye: <b>{row['equity_after']:,.2f} USDT</b> | Win: {wr:.1f}% | PF: {pf:.2f}"
        )
        self.tg.send_async(msg)

    def send_test_telegram(self) -> tuple[bool, str]:
        self.tg.reload_from_env(ENV_PATHS)
        st = self.get_api_state()
        msg = (
            f"<b>🤖 CORE_OI_CASCADE Test Bildirimi</b>\n"
            f"Durum: <b>{st['boot_status']}</b>\n"
            f"Demo Bakiye: <b>{st['total_equity']:,.2f} USDT</b>\n"
            f"Açık Pozisyon: {len(st['open_positions'])}/{st['max_concurrent']} | BTC 24s Z: {st['btc_trend_z']:+.2f}\n"
            f"Telegram bildirim bağlantısı aktif!"
        )
        ok, info = self.tg.send_sync(msg)
        self._log("SYS", f"Telegram test: {info}")
        return ok, info

    def run_5m_strategy_scan(self, full_reload: bool = False) -> None:
        """Fetch latest completed 5m bars (full 36h on boot, incremental 75m thereafter) & run CORE_OI_CASCADE."""
        with self.lock:
            if self.scan_in_progress:
                return
            self.scan_in_progress = True

        t0 = time.time()
        try:
            end_s = (int(time.time()) // 300) * 300 - 300
            is_boot = full_reload or len(self.raw_map) < len(UNIVERSE)
            lookback_s = BOT_CFG["lookback_hours"] * 3600 if is_boot else 75 * 60
            start_s = end_s - lookback_s

            fetched: dict[str, pd.DataFrame] = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futs = [pool.submit(fetch_symbol_window, s, start_s, end_s) for s in UNIVERSE]
                for fut in concurrent.futures.as_completed(futs):
                    sym, df = fut.result()
                    if df is not None and len(df) > 0:
                        fetched[sym] = df

            if not fetched:
                self._log("SYS", "Gate.io 5dk taraması boş döndü, sonraki barda tekrar denenecek.")
                return

            with self.lock:
                for sym, df_new in fetched.items():
                    if sym in self.raw_map and not is_boot:
                        comb = pd.concat([self.raw_map[sym], df_new])
                        comb = comb[~comb.index.duplicated(keep="last")].sort_index().iloc[-460:]
                        self.raw_map[sym] = comb
                    else:
                        self.raw_map[sym] = df_new

                if "BTCUSDT" not in self.raw_map or len(self.raw_map) < 8:
                    return

                data_map = align_universe(self.raw_map, max_bars=450)
                feat_map, _ = compute_universe_features(data_map)
                btc_z_arr = compute_macro_btc_trend_z(data_map, feat_map)
                last_idx = len(data_map["BTCUSDT"]) - 1
                bar_ts = data_map["BTCUSDT"].index[last_idx]
                bar_ts_iso = bar_ts.isoformat()
                btc_z_now = float(btc_z_arr[last_idx])

                self.data_map = data_map
                self.feat_map = feat_map
                self.btc_trend_z_now = btc_z_now
                self.boot_status = "LIVE_RUNNING"

                # Accrue funding on open positions if this 5m bar had a settlement
                for sym, pos in self.state["open_positions"].items():
                    if sym in data_map:
                        fev = float(data_map[sym]["funding_event"].iloc[last_idx])
                        if fev != 0.0 and bar_ts_iso != pos.get("last_fund_bar"):
                            pos["accrued_funding_rate"] = float(pos.get("accrued_funding_rate", 0.0)) + (-int(pos["side"]) * fev)
                            pos["last_fund_bar"] = bar_ts_iso

                # Update live 16-coin Microstructure Radar
                radar = []
                for sym, f in feat_map.items():
                    r = f.iloc[last_idx]
                    px = float(self.live_prices.get(sym, data_map[sym]["close"].iloc[last_idx]))
                    oi15 = float(r["oi_chg_15m"]) * 100.0
                    liqz = float(r["real_liq_z"])
                    vr = float(r["vr_15"])
                    volbp = float(r["vol_240m"]) * 10000.0
                    pofi = float(r["perp_ofi_5m"])
                    vsurge = float(r["vol_surge_15m"])
                    r15_pct = float(r["r15"]) * 100.0
                    gates_ok = sum([
                        oi15 <= -0.55,
                        1.4 <= abs(liqz) <= 4.5,
                        vr >= 0.68,
                        volbp >= 12.0,
                        vsurge >= 1.4,
                    ])
                    radar.append({
                        "symbol": sym,
                        "price": px,
                        "r15_pct": round(r15_pct, 2),
                        "oi15_pct": round(oi15, 2),
                        "liq_z": round(liqz, 2),
                        "vr": round(vr, 2),
                        "vol_bps": round(volbp, 1),
                        "perp_ofi": round(pofi, 2),
                        "vol_surge": round(vsurge, 2),
                        "gates_ok": int(gates_ok),
                    })
                radar.sort(key=lambda x: (x["gates_ok"], -x["oi15_pct"]), reverse=True)
                self.radar_snapshot = radar

                # Evaluate CORE_OI_CASCADE signals on this completed 5m bar (skip duplicate bar)
                sigs = [s for s in detect_live_signals_at_bar(feat_map, btc_z_now, last_idx) if s["strategy"] == BOT_CFG["strategy"]]
                opened_msgs = []
                day_key = str(bar_ts.date())
                today_pnl = float(self.state["daily_pnl_usdt"].get(day_key, 0.0))
                circuit_tripped = (today_pnl / max(float(self.state["equity"]), 1.0)) <= BOT_CFG["daily_dd_kill_pct"]

                if self.state.get("last_bar_ts") != bar_ts_iso and not self.paused:
                    now_ts = pd.Timestamp.now(tz="UTC")
                    for sig in sigs:
                        sym = sig["symbol"]
                        if circuit_tripped or sym in self.state["open_positions"]:
                            continue
                        if len(self.state["open_positions"]) >= BOT_CFG["max_concurrent"]:
                            continue
                        cd = self.state["cooldown_until"].get(sym)
                        if cd and now_ts < pd.Timestamp(cd):
                            continue

                        vol = float(sig["vol"])
                        tp_dist = float(np.clip(sig["tp_mult"] * vol, sig["min_tp"], 0.08))
                        sl_dist = float(np.clip(sig["sl_mult"] * vol, sig["min_tp"] * 0.72, 0.068))
                        risk_w = float(np.clip(BOT_CFG["target_risk_per_trade"] / max(sl_dist, 1e-4), BOT_CFG["min_notional_w"], BOT_CFG["max_notional_w"]))
                        cur_exp = sum(float(p["risk_weight"]) for p in self.state["open_positions"].values())
                        if cur_exp + risk_w > BOT_CFG["max_gross_exposure"]:
                            continue

                        entry_px = float(self.live_prices.get(sym, data_map[sym]["close"].iloc[last_idx]))
                        side = int(sig["side"])
                        tp_px = entry_px * (1.0 + tp_dist) if side == 1 else entry_px * (1.0 - tp_dist)
                        sl_px = entry_px * (1.0 - sl_dist) if side == 1 else entry_px * (1.0 + sl_dist)
                        notional = round(float(self.state["equity"]) * risk_w, 2)
                        expiry_ts = (now_ts + pd.Timedelta(minutes=int(sig["max_hold_mins"]))).isoformat()

                        pos = {
                            "symbol": sym,
                            "strategy": sig["strategy"],
                            "side": side,
                            "entry_ts": now_ts.isoformat(),
                            "expiry_ts": expiry_ts,
                            "entry_px": entry_px,
                            "tp_px": tp_px,
                            "sl_px": sl_px,
                            "risk_weight": risk_w,
                            "notional_usdt": notional,
                            "accrued_funding_rate": 0.0,
                            "signal_reason": sig["reason"],
                        }
                        self.state["open_positions"][sym] = pos
                        opened_msgs.append(pos)

                    self.state["last_bar_ts"] = bar_ts_iso
                    self._save_state_unlocked()

            elapsed = time.time() - t0
            hot = [f"{r['symbol']}({r['gates_ok']}/5)" for r in self.radar_snapshot[:3]]
            self._log("SCAN", f"5dk bar {bar_ts_iso[11:16]} UTC tarandı ({elapsed:.1f}s) | BTC Z={btc_z_now:+.2f} | Sinyal={len(sigs)} Açılan={len(opened_msgs)} | En yakın: {', '.join(hot)}")
            if not self._boot_notified:
                self._boot_notified = True
                self.tg.send_async(
                    f"<b>🚀 CORE_OI_CASCADE Bot Çevrimiçi</b>\n"
                    f"Bakiye: <b>{self.state['equity']:,.2f} USDT</b> | 16 parite canlı izleniyor.\n"
                    f"BTC 24s Rejim Z: <code>{btc_z_now:+.2f}</code> | En yakın: {', '.join(hot)}"
                )
            if circuit_tripped and self._cb_notified_day != day_key:
                self._cb_notified_day = day_key
                self.tg.send_async(
                    f"<b>🛑 GÜNLÜK DEVRE KESİCİ AKTİF ({day_key})</b>\n"
                    f"Bugünkü PnL: <b>{today_pnl:+.2f} USDT</b> (Limit: {BOT_CFG['daily_dd_kill_pct']*100:.1f}%).\n"
                    f"Bugün yeni işlem açılışı durduruldu."
                )
            for p in opened_msgs:
                self._log("OPEN", f"{p['symbol']} {'LONG' if p['side']==1 else 'SHORT'} @ {p['entry_px']:.4f} ({p['notional_usdt']:.0f} $) | TP={p['tp_px']:.4f} SL={p['sl_px']:.4f} | {p['signal_reason']}")
                self._notify_trade_open(p)
        except Exception as exc:
            self._log("SYS", f"Tarama hatası: {type(exc).__name__}: {exc}")
        finally:
            with self.lock:
                self.scan_in_progress = False

    def manual_close(self, sym: str) -> bool:
        with self.lock:
            if sym not in self.state["open_positions"]:
                return False
            px = float(self.live_prices.get(sym, self.state["open_positions"][sym]["entry_px"]))
            row = self._close_position_unlocked(sym, px, "MANUAL", pd.Timestamp.now(tz="UTC"))
        self._log("CLOSE" if row["net_pnl_usdt"] >= 0 else "LOSS", f"[MANUEL] {sym} {row['side']} kapatıldı @ {px:.4f} | Net: {row['net_pnl_usdt']:+.2f} $")
        self._notify_trade_close(row)
        return True

    def reset_demo(self) -> None:
        with self.lock:
            self.state = self._load_or_init_state(reset=True)
            self.paused = False
        self._log("SYS", f"Demo hesap {self.initial_capital:,.2f} USDT bakiyeye sıfırlandı.")

    def get_api_state(self) -> dict[str, Any]:
        with self.lock:
            now_ts = pd.Timestamp.now(tz="UTC")
            open_list = []
            unreal_tot = 0.0
            gross_exp = 0.0
            for sym, p in self.state["open_positions"].items():
                cur_px = float(self.live_prices.get(sym, p["entry_px"]))
                side = int(p["side"])
                unr_ret = side * ((cur_px - float(p["entry_px"])) / float(p["entry_px"]))
                unr_usdt = float(p["notional_usdt"]) * unr_ret
                unreal_tot += unr_usdt
                gross_exp += float(p["risk_weight"])
                mins_left = max(0, int((pd.Timestamp(p["expiry_ts"]) - now_ts).total_seconds() / 60.0))
                open_list.append({
                    **p,
                    "cur_px": cur_px,
                    "unreal_bps": round(unr_ret * 10000.0, 2),
                    "unreal_usdt": round(unr_usdt, 2),
                    "mins_left": mins_left,
                })

            day_key = str(now_ts.date())
            today_pnl = float(self.state["daily_pnl_usdt"].get(day_key, 0.0))
            realized_eq = float(self.state["equity"])
            total_eq = round(realized_eq + unreal_tot, 2)
            circuit_tripped = (today_pnl / max(realized_eq, 1.0)) <= BOT_CFG["daily_dd_kill_pct"]

            st = self.state["stats"]
            pf = (st["gross_win_usdt"] / st["gross_loss_usdt"]) if st["gross_loss_usdt"] > 1e-6 else (99.0 if st["gross_win_usdt"] > 0 else 0.0)

            closed_trades = []
            if TRADES_CSV.exists():
                try:
                    cdf = pd.read_csv(TRADES_CSV).tail(30).iloc[::-1]
                    closed_trades = cdf.to_dict(orient="records")
                except Exception:
                    pass

            # Append live mark-to-market point at end of equity curve
            eq_hist = list(self.state["equity_history"])
            if eq_hist:
                eq_hist.append({"ts": now_ts.isoformat(), "equity": total_eq})

            return {
                "boot_status": self.boot_status,
                "paused": self.paused,
                "telegram_enabled": self.tg.enabled,
                "scan_in_progress": self.scan_in_progress,
                "next_scan_epoch": self.next_scan_epoch,
                "last_price_ts": self.last_price_ts,
                "btc_trend_z": round(self.btc_trend_z_now, 2),
                "initial_capital": self.state["initial_capital"],
                "realized_equity": round(realized_eq, 2),
                "total_equity": total_eq,
                "unrealized_pnl_usdt": round(unreal_tot, 2),
                "today_pnl_usdt": round(today_pnl, 2),
                "peak_equity": round(float(self.state["peak_equity"]), 2),
                "max_dd_pct": round(float(self.state["max_dd_pct"]), 2),
                "max_concurrent": BOT_CFG["max_concurrent"],
                "max_gross_exposure": BOT_CFG["max_gross_exposure"],
                "gross_exposure": round(gross_exp, 2),
                "daily_dd_kill_pct": BOT_CFG["daily_dd_kill_pct"],
                "circuit_tripped": circuit_tripped,
                "stats": {**st, "profit_factor": round(pf, 2)},
                "open_positions": open_list,
                "radar": list(self.radar_snapshot),
                "closed_trades": closed_trades,
                "equity_history": eq_hist,
                "events": list(self.events[-40:]),
            }

    def start_background_loops(self) -> None:
        # Thread 1: Fast 5-second live ticker + instant TP/SL execution loop
        def _price_loop() -> None:
            while not self._stop_event.is_set():
                try:
                    self.check_realtime_exits()
                except Exception as exc:
                    self._log("SYS", f"Fiyat akış uyarısı: {type(exc).__name__}")
                self._stop_event.wait(BOT_CFG["price_poll_sec"])

        # Thread 2: Initial 36h boot + synced 5m UTC bar strategy scanner
        def _scan_loop() -> None:
            self.run_5m_strategy_scan(full_reload=True)
            while not self._stop_event.is_set():
                now_s = time.time()
                next_bar_s = ((int(now_s) // 300) + 1) * 300 + 6
                self.next_scan_epoch = float(next_bar_s)
                wait_s = max(2.0, next_bar_s - now_s)
                triggered = self._force_scan_event.wait(timeout=wait_s)
                if self._stop_event.is_set():
                    break
                if triggered:
                    self._force_scan_event.clear()
                self.run_5m_strategy_scan(full_reload=False)

        threading.Thread(target=_price_loop, daemon=True, name="price-loop").start()
        threading.Thread(target=_scan_loop, daemon=True, name="scan-loop").start()


def make_handler(bot: LiveDemoBot):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return  # Keep terminal logs clean for bot events

        def _send_json(self, payload: Any, code: int = 200) -> None:
            raw = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            path = urllib.parse.urlparse(self.path).path
            if path in ("/", "/index.html"):
                html = TEMPLATE_HTML.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
            elif path == "/api/state":
                self._send_json(bot.get_api_state())
            elif path == "/health":
                self._send_json({"ok": True, "status": bot.boot_status, "equity": bot.state["equity"], "ts": utc_now_iso()})
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            path = urllib.parse.urlparse(self.path).path
            if path == "/api/pause":
                with bot.lock:
                    bot.paused = True
                bot._log("SYS", "Bot kullanıcı tarafından duraklatıldı (açık pozisyonlar korunur).")
                self._send_json({"ok": True, "paused": True})
            elif path == "/api/resume":
                with bot.lock:
                    bot.paused = False
                bot._log("SYS", "Bot tekrar aktif edildi.")
                self._send_json({"ok": True, "paused": False})
            elif path == "/api/reset":
                bot.reset_demo()
                self._send_json({"ok": True})
            elif path == "/api/scan-now":
                bot._force_scan_event.set()
                self._send_json({"ok": True})
            elif path == "/api/test-telegram":
                ok, info = bot.send_test_telegram()
                self._send_json({"ok": ok, "message": info})
            elif path.startswith("/api/close/"):
                sym = urllib.parse.unquote(path.split("/api/close/", 1)[1])
                ok = bot.manual_close(sym)
                self._send_json({"ok": ok})
            else:
                self._send_json({"error": "not found"}, 404)

    return Handler


def main() -> None:
    default_port = int(os.environ.get("PORT", "8787"))
    default_host = os.environ.get("HOST", "0.0.0.0" if "PORT" in os.environ else "127.0.0.1")
    default_cap = float(os.environ.get("DEMO_CAPITAL", "5000"))

    parser = argparse.ArgumentParser(description="Live 5,000 USDT Automated CORE_OI_CASCADE Demo Trading Bot & Dashboard")
    parser.add_argument("--capital", type=float, default=default_cap, help="Starting demo balance in USDT (default: 5000)")
    parser.add_argument("--port", type=int, default=default_port, help="HTTP dashboard port (reads $PORT in cloud)")
    parser.add_argument("--host", type=str, default=default_host, help="Bind host (0.0.0.0 in cloud, 127.0.0.1 locally)")
    parser.add_argument("--reset", action="store_true", help="Reset demo balance to starting capital")
    args = parser.parse_args()

    bot = LiveDemoBot(initial_capital=args.capital, reset=args.reset)
    bot.start_background_loops()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(bot))
    print(f"\n==================================================================", flush=True)
    print(f"  CORE_OI_CASCADE CANLI DEMO BOT AKTIF ({args.capital:,.2f} USDT)", flush=True)
    print(f"  HTML Panel Adresi: http://{args.host}:{args.port}", flush=True)
    print(f"==================================================================\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        bot._stop_event.set()
        server.shutdown()


if __name__ == "__main__":
    main()

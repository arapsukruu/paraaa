"""Gate.io Public USDT-M Perpetual & Spot Microstructure Data Collector.

Downloads 32 days of aligned 5-minute data for 16 liquid pairs (zero API keys required):
  1. USDT-M Perp 5m OHLCV + Quote Volume (USDT)
  2. Index Price 5m OHLC (for exact Perp-vs-Index Basis / Premium)
  3. Spot 5m OHLCV + Spot Quote Volume (USDT)
  4. 5m Contract Stats: Open Interest USD, Taker Buy/Sell Ratio (Perp OFI),
     Top Trader Long/Short Size Ratio, Retail Long/Short Account Ratio,
     and REAL realized Long & Short Liquidation Volume (USD)
  5. Historical Funding Rates (exact settlement timestamps & rates)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from config import BAR_MINS, PROCESSED_DIR, UNIVERSE

GATE_BASE = "https://api.gateio.ws/api/v4"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) quant-lab/2.0", "Accept": "application/json"}

# Thread-local keep-alive HTTP sessions (~10x faster than fresh TLS per request)
_TLS = threading.local()
_RL_LOCK = threading.Lock()
_LAST_REQ = 0.0
_MIN_INTERVAL = 0.025  # ~40 req/sec global (Gate public limit is 900/10s = 90 req/sec)
_ERR_LOGGED = 0


def _get_session() -> requests.Session:
    sess = getattr(_TLS, "sess", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update(HEADERS)
        adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16)
        sess.mount("https://", adapter)
        _TLS.sess = sess
    return sess


def _sym_to_gate(sym: str) -> str:
    return sym[:-4] + "_USDT" if sym.endswith("USDT") else sym


def gate_get(path: str, retries: int = 4) -> Any:
    global _LAST_REQ, _ERR_LOGGED
    url = GATE_BASE + path
    sess = _get_session()
    for attempt in range(retries):
        with _RL_LOCK:
            now = time.monotonic()
            wait = _MIN_INTERVAL - (now - _LAST_REQ)
            if wait > 0:
                time.sleep(wait)
            _LAST_REQ = time.monotonic()
        try:
            resp = sess.get(url, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (400, 404):
                if _ERR_LOGGED < 5:
                    _ERR_LOGGED += 1
                    print(f"[gate-collector] HTTP {resp.status_code} on {path[:90]} -> {resp.text[:160]}", flush=True)
                return None
            if _ERR_LOGGED < 5:
                _ERR_LOGGED += 1
                print(f"[gate-collector] HTTP {resp.status_code} retry {attempt+1} on {path[:90]}", flush=True)
            time.sleep(1.2 * (attempt + 1))
        except Exception as exc:
            # Fallback to urllib if requests session ever resets
            try:
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=15) as r2:
                    return json.loads(r2.read().decode("utf-8"))
            except Exception:
                pass
            if _ERR_LOGGED < 5 and attempt == retries - 1:
                _ERR_LOGGED += 1
                print(f"[gate-collector] ERR on {path[:90]} -> {type(exc).__name__}: {exc}", flush=True)
            time.sleep(1.0 * (attempt + 1))
    return None


def fetch_futures_candles(contract: str, start_s: int, end_s: int, interval: str = "5m") -> pd.DataFrame:
    step_s = BAR_MINS * 60
    chunk_s = 900 * step_s  # Safe under Gate 1000/2000 cap
    rows = []
    cur = start_s
    while cur < end_s:
        nxt = min(cur + chunk_s, end_s)
        data = gate_get(f"/futures/usdt/candlesticks?contract={contract}&interval={interval}&from={cur}&to={nxt}")
        if isinstance(data, list) and data:
            rows.extend(data)
        cur = nxt + step_s
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts_s"] = pd.to_numeric(df["t"], errors="coerce").astype("int64")
    for c in ["o", "h", "l", "c", "v", "sum"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df = df.drop_duplicates("ts_s").sort_values("ts_s").reset_index(drop=True)
    return df


def fetch_spot_candles(pair: str, start_s: int, end_s: int, interval: str = "5m") -> pd.DataFrame:
    step_s = BAR_MINS * 60
    chunk_s = 850 * step_s  # < 1000 spot limit
    rows = []
    cur = start_s
    while cur < end_s:
        nxt = min(cur + chunk_s, end_s)
        data = gate_get(f"/spot/candlesticks?currency_pair={pair}&interval={interval}&from={cur}&to={nxt}")
        if isinstance(data, list) and data:
            rows.extend(data)
        cur = nxt + step_s
    if not rows:
        return pd.DataFrame()
    # Gate spot array: [ts, quote_vol, close, high, low, open, base_vol, is_closed]
    df = pd.DataFrame(rows, columns=["ts_s", "spot_quote_vol", "spot_close", "spot_high", "spot_low", "spot_open", "spot_base_vol", "closed"])
    df["ts_s"] = pd.to_numeric(df["ts_s"], errors="coerce").astype("int64")
    for c in ["spot_quote_vol", "spot_close", "spot_high", "spot_low", "spot_open"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df = df[["ts_s", "spot_open", "spot_high", "spot_low", "spot_close", "spot_quote_vol"]].drop_duplicates("ts_s").sort_values("ts_s")
    return df


def fetch_contract_stats(contract: str, start_s: int, end_s: int, interval: str = "5m") -> pd.DataFrame:
    step_s = BAR_MINS * 60
    rows = []
    cur = start_s
    # Gate contract_stats max limit is 100 per call
    guard = 0
    while cur < end_s and guard < 120:
        guard += 1
        data = gate_get(f"/futures/usdt/contract_stats?contract={contract}&interval={interval}&from={cur}&limit=100")
        if not isinstance(data, list) or not data:
            cur += 100 * step_s
            continue
        rows.extend(data)
        max_t = max(int(item["time"]) for item in data if "time" in item)
        if max_t < cur:
            break
        cur = max_t + step_s
        if len(data) < 5 and cur >= end_s - 5 * step_s:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts_s"] = pd.to_numeric(df["time"], errors="coerce").astype("int64")
    # Standardize available columns
    col_map = {
        "open_interest_usd": "oi_usdt",
        "open_interest": "oi_contracts",
        "mark_price": "stats_mark",
        "lsr_taker": "lsr_taker",
        "lsr_account": "retail_ls_acc",
        "top_lsr_size": "top_ls_pos",
        "top_lsr_account": "top_ls_acc",
        "long_liq_usd": "long_liq_usd",
        "short_liq_usd": "short_liq_usd",
        "long_liq_size": "long_liq_size",
        "short_liq_size": "short_liq_size",
    }
    keep = ["ts_s"] + [k for k in col_map if k in df.columns]
    df = df[keep].rename(columns=col_map)
    for c in df.columns:
        if c != "ts_s":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    if "oi_usdt" not in df.columns and "oi_contracts" in df.columns and "stats_mark" in df.columns:
        df["oi_usdt"] = (df["oi_contracts"] * df["stats_mark"]).astype("float32")
    df = df.drop_duplicates("ts_s").sort_values("ts_s").reset_index(drop=True)
    return df


def fetch_funding(contract: str, limit: int = 180) -> pd.DataFrame:
    data = gate_get(f"/futures/usdt/funding_rate?contract={contract}&limit={limit}")
    if not isinstance(data, list) or not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["ts_s"] = pd.to_numeric(df["t"], errors="coerce").astype("int64")
    # Snap to nearest 5m bucket
    df["ts_s"] = (np.round(df["ts_s"] / 300.0) * 300).astype("int64")
    df["funding_event"] = pd.to_numeric(df["r"], errors="coerce").astype("float32")
    return df[["ts_s", "funding_event"]].drop_duplicates("ts_s").sort_values("ts_s")


def collect_symbol(sym: str, start_s: int, end_s: int, force: bool = False) -> tuple[str, dict[str, Any] | None]:
    out_path = PROCESSED_DIR / f"{sym}.pkl.gz"
    if out_path.exists() and not force:
        df = pd.read_pickle(out_path, compression="gzip")
        if len(df) > 4000:
            return sym, {"rows": len(df), "cached": True, "start": str(df.index.min()), "end": str(df.index.max())}

    contract = _sym_to_gate(sym)
    # Fetch perp, index, spot, contract_stats, funding concurrently for this symbol
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as subpool:
        f_perp = subpool.submit(fetch_futures_candles, contract, start_s, end_s, "5m")
        f_idx = subpool.submit(fetch_futures_candles, f"index_{contract}", start_s, end_s, "5m")
        f_spot = subpool.submit(fetch_spot_candles, contract, start_s, end_s, "5m")
        f_stats = subpool.submit(fetch_contract_stats, contract, start_s, end_s, "5m")
        f_fund = subpool.submit(fetch_funding, contract, 200)
        perp = f_perp.result()
        idx_df = f_idx.result()
        spot_df = f_spot.result()
        stats_df = f_stats.result()
        fr_df = f_fund.result()

    if len(perp) < 2000:
        return sym, None

    perp = perp.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "sum": "quote_vol", "v": "base_contracts"})
    df = perp[["ts_s", "open", "high", "low", "close", "quote_vol"]].copy()

    # Index candles -> exact Perp vs Index basis
    if len(idx_df) > 0:
        idx_df = idx_df[["ts_s", "c"]].rename(columns={"c": "index_close"})
        df = df.merge(idx_df, on="ts_s", how="left")
        df["premium_close"] = ((df["close"] - df["index_close"]) / (df["index_close"] + 1e-12)).astype("float32")
    else:
        df["index_close"] = np.float32(np.nan)
        df["premium_close"] = np.float32(0.0)

    # Spot candles -> Spot volume & Spot BVC order flow
    if len(spot_df) > 0:
        df = df.merge(spot_df, on="ts_s", how="left")
    else:
        for c in ["spot_open", "spot_high", "spot_low", "spot_close", "spot_quote_vol"]:
            df[c] = np.float32(np.nan)

    # Contract stats (OI, Taker L/S, Top Trader L/S, Retail L/S, REAL Liquidations!)
    if len(stats_df) > 0:
        df = pd.merge_asof(
            df.sort_values("ts_s"),
            stats_df.sort_values("ts_s"),
            on="ts_s",
            direction="backward",
            tolerance=900,
        )
    for col in ["oi_usdt", "lsr_taker", "retail_ls_acc", "top_ls_pos", "long_liq_usd", "short_liq_usd", "long_liq_size", "short_liq_size"]:
        if col not in df.columns:
            df[col] = np.float32(np.nan)

    # Reconstruct exact signed Perp aggressor delta from lsr_taker = buy_vol / sell_vol
    lsr = df["lsr_taker"].astype(np.float64).clip(0.01, 100.0)
    perp_ofi = ((lsr - 1.0) / (lsr + 1.0)).fillna(0.0)
    bvc_perp = ((df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-12)).clip(-1.0, 1.0).fillna(0.0)
    perp_ofi = np.where(df["lsr_taker"].notna(), perp_ofi, bvc_perp)
    df["perp_ofi_bar"] = perp_ofi.astype("float32")
    df["perp_delta_q"] = (df["quote_vol"].astype(np.float64) * perp_ofi).astype("float32")

    # Spot signed delta via Bulk Volume Classification
    bvc_spot = ((df["spot_close"] - df["spot_open"]) / (df["spot_high"] - df["spot_low"] + 1e-12)).clip(-1.0, 1.0).fillna(0.0)
    df["spot_ofi_bar"] = bvc_spot.astype("float32")
    df["spot_delta_q"] = (df["spot_quote_vol"].fillna(0.0).astype(np.float64) * bvc_spot).astype("float32")

    # Funding rate events
    if len(fr_df) > 0:
        df = df.merge(fr_df, on="ts_s", how="left")
        df["funding_rate"] = df["funding_event"].ffill().fillna(0.0).astype("float32")
        df["funding_event"] = df["funding_event"].fillna(0.0).astype("float32")
    else:
        df["funding_rate"] = np.float32(0.0)
        df["funding_event"] = np.float32(0.0)

    df["timestamp"] = pd.to_datetime(df["ts_s"], unit="s", utc=True)
    df = df.set_index("timestamp").sort_index()
    df.to_pickle(out_path, compression="gzip")

    info = {
        "rows": int(len(df)),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "spot_cov_pct": round(float(df["spot_close"].notna().mean() * 100.0), 1),
        "oi_cov_pct": round(float(df["oi_usdt"].notna().mean() * 100.0), 1),
        "lsr_cov_pct": round(float(df["lsr_taker"].notna().mean() * 100.0), 1),
        "liq_events": int(((df["long_liq_usd"].fillna(0) > 0) | (df["short_liq_usd"].fillna(0) > 0)).sum()),
        "funding_events": int((df["funding_event"] != 0).sum()),
    }
    return sym, info


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Gate.io 5m Futures+Spot+OI+Liq microstructure dataset.")
    parser.add_argument("--days", type=int, default=32, help="Lookback days (max 33 for Gate 5m 10,000-bar cap)")
    parser.add_argument("--symbols", nargs="*", default=UNIVERSE, help="Symbols to download")
    parser.add_argument("--workers", type=int, default=4, help="Parallel symbol workers")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    # Snap end time to last completed 5m candle
    end_s = (int(time.time()) // 300) * 300 - 300
    start_s = end_s - args.days * 86400
    print(
        f"[gate-collector] Fetching {args.days}d of 5m futures+spot+index+OI+liq data "
        f"({pd.to_datetime(start_s, unit='s', utc=True)} -> {pd.to_datetime(end_s, unit='s', utc=True)}) "
        f"for {len(args.symbols)} symbols...",
        flush=True,
    )

    t0 = time.time()
    summary = {"source": "gateio_usdt_perpetual_5m", "days": args.days, "symbols": {}}

    # Probe BTCUSDT first to verify fields & connectivity cleanly
    first_sym = args.symbols[0]
    sym, info = collect_symbol(first_sym, start_s, end_s, force=args.force)
    if info is None:
        print(f"[gate-collector] ERROR: Failed to fetch {first_sym}. Aborting.", flush=True)
        sys.exit(2)
    summary["symbols"][sym] = info
    print(f"[1/{len(args.symbols)}] {sym}: {info}", flush=True)

    rest = args.symbols[1:]
    done = 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(collect_symbol, s, start_s, end_s, args.force): s for s in rest}
        for fut in concurrent.futures.as_completed(futs):
            done += 1
            s_out, inf = fut.result()
            if inf is None:
                print(f"[{done}/{len(args.symbols)}] {s_out}: SKIPPED", flush=True)
            else:
                summary["symbols"][s_out] = inf
                print(f"[{done}/{len(args.symbols)}] {s_out}: {inf}", flush=True)

    meta_path = PROCESSED_DIR / "dataset_meta.json"
    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[gate-collector] Complete in {time.time() - t0:.1f}s -> {meta_path}", flush=True)


if __name__ == "__main__":
    main()

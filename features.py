"""Strictly causal 5m microstructure, derivatives, REAL liquidation, Pair-DNA, and shock features.

Zero lookahead guarantee:
  - Atomic resolution: BAR_MINS = 5 minutes (288 bars/day).
  - Every rolling window uses strictly past data <= timestamp t.
  - Exploits Gate.io real exchange liquidations (long_liq_usd, short_liq_usd),
    Open Interest USD, Taker Buy/Sell Ratio, Top Trader Position Ratio, Retail Account Ratio,
    Perp-vs-Index Basis, Spot vs Perp Order Flow, Pair DNA (Variance Ratio), and BTC Beta Residuals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import BAR_MINS, BARS_PER_DAY, PROCESSED_DIR, UNIVERSE


def load_universe(symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load all available processed symbol DataFrames and align to a common UTC 5m index."""
    syms = symbols or UNIVERSE
    raw_map: dict[str, pd.DataFrame] = {}
    for sym in syms:
        p = PROCESSED_DIR / f"{sym}.pkl.gz"
        if p.exists():
            df = pd.read_pickle(p, compression="gzip")
            if len(df) > 2000:
                raw_map[sym] = df

    if not raw_map:
        raise FileNotFoundError(f"No processed datasets found in {PROCESSED_DIR}")

    start = max(df.index.min() for df in raw_map.values())
    end = min(df.index.max() for df in raw_map.values())
    common_idx = pd.date_range(start, end, freq=f"{BAR_MINS}min", tz="UTC")

    aligned: dict[str, pd.DataFrame] = {}
    for sym, df in raw_map.items():
        d = df.reindex(common_idx)
        ffill_cols = [
            "open", "high", "low", "close", "index_close", "spot_close", "spot_open", "spot_high", "spot_low",
            "premium_close", "oi_usdt", "top_ls_pos", "retail_ls_acc", "lsr_taker", "funding_rate",
        ]
        for c in ffill_cols:
            if c in d.columns:
                d[c] = d[c].ffill()
        zero_cols = [
            "quote_vol", "perp_delta_q", "perp_ofi_bar", "spot_quote_vol", "spot_delta_q", "spot_ofi_bar",
            "long_liq_usd", "short_liq_usd", "long_liq_size", "short_liq_size", "funding_event",
        ]
        for c in zero_cols:
            if c in d.columns:
                d[c] = d[c].fillna(0.0)
        d["open"] = d["open"].fillna(d["close"])
        d["high"] = d["high"].fillna(d["close"])
        d["low"] = d["low"].fillna(d["close"])
        aligned[sym] = d
    return aligned


def _rolling_z(s: pd.Series, win: int, min_p: int | None = None) -> pd.Series:
    mp = min_p or max(8, win // 4)
    mu = s.rolling(win, min_periods=mp).mean()
    sd = s.rolling(win, min_periods=mp).std()
    return ((s - mu) / (sd + 1e-12)).clip(-8.0, 8.0)


def compute_universe_features(data_map: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Compute strictly causal features per symbol + universe-wide Market Shock & Regime frame."""
    # Step 1: Universe 5m return, volatility, and liquidation matrices
    r5_dict = {}
    vol1h_dict = {}
    liq_tot_dict = {}
    for sym, df in data_map.items():
        c = df["close"].astype(np.float64)
        r5 = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        r5_dict[sym] = r5
        vol1h_dict[sym] = r5.rolling(12, min_periods=4).std().fillna(0.0)  # 12 x 5m = 60m
        l_liq = df["long_liq_usd"].astype(np.float64).fillna(0.0)
        s_liq = df["short_liq_usd"].astype(np.float64).fillna(0.0)
        if (l_liq.sum() + s_liq.sum()) == 0.0 and "long_liq_size" in df.columns:
            # Fallback if USD column empty: use contract size * close
            l_liq = df["long_liq_size"].astype(np.float64).fillna(0.0) * c
            s_liq = df["short_liq_size"].astype(np.float64).fillna(0.0) * c
        liq_tot_dict[sym] = l_liq + s_liq

    r5_mat = pd.DataFrame(r5_dict)
    vol1h_mat = pd.DataFrame(vol1h_dict)
    liq_mat = pd.DataFrame(liq_tot_dict)

    # Universe Market Shock & Synchronization Index (strictly causal)
    mkt_median_vol = vol1h_mat.median(axis=1)
    mkt_vol_z = _rolling_z(mkt_median_vol, BARS_PER_DAY, min_p=48).fillna(0.0)
    # Cross-sectional unanimity over 15m (3 bars): near 1.0 when entire market moves together
    r15_mat = r5_mat.rolling(3, min_periods=1).sum()
    mkt_unanimity = r15_mat.apply(np.sign).mean(axis=1).abs().rolling(2, min_periods=1).mean().fillna(0.0)
    mkt_abs_r5 = r5_mat.median(axis=1).abs()
    mkt_ret_shock = (mkt_abs_r5 / (mkt_median_vol + 1e-9)).clip(0.0, 10.0).fillna(0.0)
    # Market-wide realized liquidation surge
    mkt_liq_sum = liq_mat.sum(axis=1)
    mkt_liq_z = _rolling_z(np.log1p(mkt_liq_sum), BARS_PER_DAY, min_p=48).fillna(0.0)

    mkt_shock_score = (
        0.40 * mkt_vol_z.clip(lower=0.0)
        + 0.35 * (mkt_ret_shock - 1.5).clip(lower=0.0) * mkt_unanimity
        + 0.25 * (mkt_liq_z - 1.0).clip(lower=0.0)
    ).fillna(0.0)

    mkt_frame = pd.DataFrame(
        {
            "mkt_vol_z": mkt_vol_z,
            "mkt_unanimity": mkt_unanimity,
            "mkt_ret_shock": mkt_ret_shock,
            "mkt_liq_z": mkt_liq_z,
            "mkt_shock_score": mkt_shock_score,
        },
        index=r5_mat.index,
    )

    btc_r5 = r5_mat["BTCUSDT"] if "BTCUSDT" in r5_mat else r5_mat.mean(axis=1)
    btc_r15 = btc_r5.rolling(3, min_periods=1).sum().fillna(0.0)
    btc_r60 = btc_r5.rolling(12, min_periods=4).sum().fillna(0.0)
    btc_var5_12h = btc_r5.rolling(144, min_periods=36).var().fillna(1e-9)

    idx = r5_mat.index
    hour_frac = idx.hour + idx.minute / 60.0
    hour_sin = np.sin(2.0 * np.pi * hour_frac / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hour_frac / 24.0)
    mins_to_funding = (480 - ((idx.hour % 8) * 60 + idx.minute)) / 480.0

    feat_map: dict[str, pd.DataFrame] = {}

    for sym, df in data_map.items():
        c = df["close"].astype(np.float64)
        h = df["high"].astype(np.float64)
        l = df["low"].astype(np.float64)
        qvol = df["quote_vol"].astype(np.float64)
        p_delta = df["perp_delta_q"].astype(np.float64)
        s_qvol = df["spot_quote_vol"].astype(np.float64)
        s_delta = df["spot_delta_q"].astype(np.float64)
        prem = df["premium_close"].astype(np.float64).fillna(0.0)
        oi = df["oi_usdt"].astype(np.float64)
        top_ls = df["top_ls_pos"].astype(np.float64)
        ret_ls = df["retail_ls_acc"].astype(np.float64)
        fund = df["funding_rate"].astype(np.float64).fillna(0.0)

        l_liq = df["long_liq_usd"].astype(np.float64).fillna(0.0)
        s_liq = df["short_liq_usd"].astype(np.float64).fillna(0.0)
        if (l_liq.sum() + s_liq.sum()) == 0.0 and "long_liq_size" in df.columns:
            l_liq = df["long_liq_size"].astype(np.float64).fillna(0.0) * c
            s_liq = df["short_liq_size"].astype(np.float64).fillna(0.0) * c

        r5 = r5_mat[sym]
        r15 = r5.rolling(3, min_periods=1).sum().fillna(0.0)
        r60 = r5.rolling(12, min_periods=4).sum().fillna(0.0)
        r240 = r5.rolling(48, min_periods=12).sum().fillna(0.0)

        # Realized Volatility (per-5m-bar std)
        vol_60m = vol1h_mat[sym]                                       # 1h window (12 bars)
        vol_240m = r5.rolling(48, min_periods=12).std().fillna(0.0)    # 4h window
        vol_1440m = r5.rolling(BARS_PER_DAY, min_periods=48).std().fillna(0.0)  # 24h window
        vol_ratio = (vol_60m / (vol_1440m + 1e-9)).clip(0.0, 10.0)
        bar_range_pct = ((h - l) / (c + 1e-12)).fillna(0.0)
        wick_shock = (bar_range_pct / (vol_240m + 1e-9)).clip(0.0, 15.0)

        # Order Flow Imbalance (Exact Perp Taker OFI + Spot BVC OFI)
        qvol_15m = qvol.rolling(3, min_periods=1).sum()
        qvol_60m = qvol.rolling(12, min_periods=3).sum()
        sqvol_15m = s_qvol.rolling(3, min_periods=1).sum()
        sqvol_60m = s_qvol.rolling(12, min_periods=3).sum()

        perp_ofi_5m = df["perp_ofi_bar"].astype(np.float64).clip(-1.0, 1.0)
        perp_ofi_15m = (p_delta.rolling(3, min_periods=1).sum() / (qvol_15m + 1.0)).clip(-1.0, 1.0)
        perp_ofi_60m = (p_delta.rolling(12, min_periods=3).sum() / (qvol_60m + 1.0)).clip(-1.0, 1.0)

        spot_ofi_5m = df["spot_ofi_bar"].astype(np.float64).clip(-1.0, 1.0)
        spot_ofi_15m = (s_delta.rolling(3, min_periods=1).sum() / (sqvol_15m + 1.0)).clip(-1.0, 1.0)
        spot_ofi_60m = (s_delta.rolling(12, min_periods=3).sum() / (sqvol_60m + 1.0)).clip(-1.0, 1.0)

        # Spot vs Perp Divergence (>0: Spot stronger than Perp; <0: Perp FOMO without Spot)
        spot_perp_div_15m = spot_ofi_15m - perp_ofi_15m
        spot_perp_div_60m = spot_ofi_60m - perp_ofi_60m
        perp_spot_vol_log = np.log((qvol_15m + 10.0) / (sqvol_15m + 10.0)).clip(-5.0, 5.0)

        # Volume Surge (15m vs trailing 24h mean)
        vol_surge_15m = (qvol_15m / (qvol_15m.rolling(BARS_PER_DAY, min_periods=48).mean() + 1.0)).clip(0.0, 20.0)

        # Derivatives: Basis (Perp vs Index), OI Delta, Real Liquidations, Smart vs Retail L/S
        prem_z_240m = _rolling_z(prem, 48, min_p=12).fillna(0.0)
        prem_z_1440m = _rolling_z(prem, BARS_PER_DAY, min_p=48).fillna(0.0)

        oi_chg_15m = np.log((oi + 1.0) / (oi.shift(3) + 1.0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        oi_chg_60m = np.log((oi + 1.0) / (oi.shift(12) + 1.0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        oi_z_360m = _rolling_z(oi_chg_15m, 72, min_p=18).fillna(0.0)

        # REAL Exchange Liquidations over 15m:
        # Signed net liquidation flow: + = Short liquidations (forced buying up), - = Long liquidations (forced selling down)
        net_liq_15m = (s_liq - l_liq).rolling(3, min_periods=1).sum()
        tot_liq_15m = (s_liq + l_liq).rolling(3, min_periods=1).sum()
        signed_log_liq = np.sign(net_liq_15m) * np.log1p(np.abs(net_liq_15m))
        real_liq_z = _rolling_z(signed_log_liq, BARS_PER_DAY, min_p=48).fillna(0.0)
        liq_vol_ratio = (tot_liq_15m / (qvol_15m + 100.0)).clip(0.0, 1.0)

        # Combined Liquidation Cascade Score (real liqs + OI drop + sharp move)
        oi_drop_boost = 1.0 + (-oi_z_360m).clip(lower=0.0) * 0.4
        ret15_norm = (r15 / (vol_240m * np.sqrt(3.0) + 1e-9)).clip(-10.0, 10.0)
        liq_cascade_score = np.where(
            np.abs(real_liq_z) > 0.5,
            real_liq_z * oi_drop_boost,
            np.sign(r15) * (np.abs(ret15_norm) - 1.5).clip(lower=0.0) * (-oi_z_360m).clip(lower=0.0),
        )

        # Smart Money (Top Trader Size L/S) vs Retail Crowd (Account L/S)
        retail_ls_z = _rolling_z(np.log(ret_ls.clip(lower=0.05)), BARS_PER_DAY, min_p=48).fillna(0.0)
        top_ls_z = _rolling_z(np.log(top_ls.clip(lower=0.05)), BARS_PER_DAY, min_p=48).fillna(0.0)
        smart_retail_diff = np.log(top_ls.clip(lower=0.05) / ret_ls.clip(lower=0.05))
        smart_retail_z = _rolling_z(smart_retail_diff, BARS_PER_DAY, min_p=48).fillna(0.0)

        # Pair DNA: Variance Ratio of 60m return vs 5m return over trailing 24h (288 bars)
        # VR > 1.0 = Trending DNA; VR < 1.0 = Mean-Reverting DNA
        var_5m = r5.rolling(BARS_PER_DAY, min_periods=72).var()
        var_60m = r60.rolling(BARS_PER_DAY, min_periods=72).var()
        vr_60 = (var_60m / (12.0 * var_5m + 1e-12)).clip(0.1, 4.0).fillna(1.0)

        # Rolling BTC Beta & Idiosyncratic Residuals (strictly causal 12h window = 144 bars)
        cov_btc = r5.rolling(144, min_periods=36).cov(btc_r5)
        btc_beta = (cov_btc / (btc_var5_12h + 1e-12)).clip(-1.0, 4.0).fillna(1.0)
        corr_btc = r5.rolling(144, min_periods=36).corr(btc_r5).fillna(0.0)
        btc_r2 = (corr_btc ** 2).clip(0.0, 1.0)

        resid_15m = r15 - btc_beta * btc_r15
        resid_60m = r60 - btc_beta * btc_r60
        resid_z_15m = _rolling_z(resid_15m, 72, min_p=18).fillna(0.0)     # 6h window
        resid_z_60m = _rolling_z(resid_60m, 216, min_p=48).fillna(0.0)    # 18h window

        f = pd.DataFrame(
            {
                "r5": r5,
                "r15": r15,
                "r60": r60,
                "r240": r240,
                "vol_60m": vol_60m,
                "vol_240m": vol_240m,
                "vol_1440m": vol_1440m,
                "vol_ratio": vol_ratio,
                "wick_shock": wick_shock,
                "perp_ofi_5m": perp_ofi_5m,
                "perp_ofi_15m": perp_ofi_15m,
                "perp_ofi_60m": perp_ofi_60m,
                "spot_ofi_5m": spot_ofi_5m,
                "spot_ofi_15m": spot_ofi_15m,
                "spot_ofi_60m": spot_ofi_60m,
                "spot_perp_div_15m": spot_perp_div_15m,
                "spot_perp_div_60m": spot_perp_div_60m,
                "perp_spot_vol_log": perp_spot_vol_log,
                "vol_surge_15m": vol_surge_15m,
                "prem_close_bps": prem * 10000.0,
                "prem_z_240m": prem_z_240m,
                "prem_z_1440m": prem_z_1440m,
                "oi_chg_15m": oi_chg_15m,
                "oi_chg_60m": oi_chg_60m,
                "oi_z_360m": oi_z_360m,
                "real_liq_z": real_liq_z,
                "liq_vol_ratio": liq_vol_ratio,
                "liq_cascade_score": liq_cascade_score,
                "retail_ls_z": retail_ls_z,
                "top_ls_z": top_ls_z,
                "smart_retail_z": smart_retail_z,
                "funding_bps": fund * 10000.0,
                "vr_15": vr_60,  # kept key name vr_15 for clean compatibility across modules
                "btc_beta": btc_beta,
                "btc_r2": btc_r2,
                "resid_15m": resid_15m,
                "resid_60m": resid_60m,
                "resid_z_15m": resid_z_15m,
                "resid_z_60m": resid_z_60m,
                "mkt_vol_z": mkt_frame["mkt_vol_z"],
                "mkt_unanimity": mkt_frame["mkt_unanimity"],
                "mkt_liq_z": mkt_frame["mkt_liq_z"],
                "mkt_shock_score": mkt_frame["mkt_shock_score"],
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
                "mins_to_funding": mins_to_funding,
            },
            index=df.index,
        ).astype("float32")
        feat_map[sym] = f

    return feat_map, mkt_frame

#!/usr/bin/env python3
# backtest_bollinger.py
"""
BOLLINGER bands (fixed params)
- strategy: Bollinger
- n = 8
- k = 8
- rebalance = True
Outputs:
 - OUT_RESULTS: single-row summary CSV (apr, apr_usd, total_fees, hours, hits, hit_rate)
 - OUT_HOURLY_DETAIL: detailed per-observation CSV (fees_generated_by_position, tick bounds, ...)
 - OUT_BANDS_PNG: plot of bands vs price

Requirements:
  - python >= 3.8
  - pandas, numpy, matplotlib
"""
__author__ = "Orwee"

import os
import time
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import multiprocessing

# ---------------- CONFIG ----------------
DATA_CSV = r"C:/Users/Administrator/Downloads/data_backtesting_sample.csv"

OUT_FOLDER = r"C:\Users\Administrator\Desktop\Folder2"
OUT_RESULTS = os.path.join(OUT_FOLDER, "backtesting_results_apr_bollinger_fixed.csv")
OUT_HOURLY_DETAIL = os.path.join(OUT_FOLDER, "best_algorithm_hourly_details_bollinger.csv")
OUT_BANDS_PNG = os.path.join(OUT_FOLDER, "best_algorithm_bands_bollinger.png")

# Fixed algorithm params requested
STRATEGY = "bollinger"
N = 8
K = 8.0
REBALANCE = True

# Sampling / freeze behavior
SAMPLE_PERIOD = 60        # compute bands every SAMPLE_PERIOD observations
REBALANCE_HOLD = 30      # freeze range for this many observations after a forced change
GAS_PENALTY_USD = 0.2    # subtract this fixed USD when range is changed

# Band enter/exit / blending
ENTER_FRAC = 0.95
EXIT_FRAC = 0.7
BLEND_ALPHA = 0.75

# Investment
INVEST_LIQ = 1000.0

# Start date fallback (if blocks aren't unix timestamps)
START_DATE = "2025-09-01T00:00:00Z"

# ---------------- Utilities ----------------
def _format_seconds(s):
    s = int(round(s))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h:d}h {m:02d}m {sec:02d}s"
    if m > 0:
        return f"{m:d}m {sec:d}s"
    return f"{sec:d}s"

def _atomic_write_df(path, df):
    tmp = path + ".tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(tmp, index=False)
        try:
            fd = open(tmp, "rb")
            fd.flush()
            os.fsync(fd.fileno())
            fd.close()
        except Exception:
            pass
        os.replace(tmp, path)
    except Exception:
        try:
            df.to_csv(path, index=False)
        except Exception as e:
            print("Error writing file:", e)

def _choose_tick_bounds_from_prices(ticks_df, lower_price, upper_price):
    if ticks_df is None or len(ticks_df) == 0:
        return None, None
    lowers = pd.to_numeric(ticks_df["tick_lower_transformed"].values, errors="coerce")
    uppers = pd.to_numeric(ticks_df["tick_upper_transformed"].values, errors="coerce")
    mask_lower = np.where(lowers <= lower_price)[0]
    lower_idx = int(mask_lower.max()) if len(mask_lower) > 0 else 0
    mask_upper = np.where(uppers >= upper_price)[0]
    upper_idx = int(mask_upper.min()) if len(mask_upper) > 0 else len(uppers) - 1
    if lower_idx > upper_idx:
        centers = (lowers + uppers) / 2.0
        mid = 0.5 * (lower_price + upper_price)
        idx = int(np.argmin(np.abs(centers - mid)))
        return idx, idx
    return lower_idx, upper_idx

# ---------------- Data prep ----------------
def prepare_from_data_backtesting(data_backtesting):
    df = data_backtesting.copy()
    if "block" not in df.columns:
        if "periodStartUnix" in df.columns:
            df["block"] = df["periodStartUnix"]
        else:
            raise KeyError("data_backtesting debe contener 'block' (o 'periodStartUnix')")
    df["block"] = pd.to_numeric(df["block"], errors="coerce")
    df["tick_lower_transformed"] = pd.to_numeric(df.get("tick_lower_transformed", np.nan), errors="coerce")
    df["tick_upper_transformed"] = pd.to_numeric(df.get("tick_upper_transformed", np.nan), errors="coerce")

    if "amount_usd_total" not in df.columns:
        if ("amount0_usd" in df.columns) and ("amount1_usd" in df.columns):
            df["amount_usd_total"] = pd.to_numeric(df["amount0_usd"].fillna(0.0), errors="coerce") + pd.to_numeric(df["amount1_usd"].fillna(0.0), errors="coerce")
        elif ("pct_tvl_by_tick" in df.columns) and ("pool_totalValueLockedUSD_at_block" in df.columns):
            df["amount_usd_total"] = pd.to_numeric(df["pct_tvl_by_tick"].fillna(0.0), errors="coerce") * pd.to_numeric(df["pool_totalValueLockedUSD_at_block"].fillna(0.0), errors="coerce")
        else:
            df["amount_usd_total"] = np.nan
    else:
        df["amount_usd_total"] = pd.to_numeric(df["amount_usd_total"], errors="coerce")

    df["feesUSD_real"] = pd.to_numeric(df.get("feesUSD_real", 0.0), errors="coerce").fillna(0.0)
    df["token1Price"] = pd.to_numeric(df.get("token1Price", np.nan), errors="coerce")

    ticks_grouped = {}
    for per, grp in df.groupby("block"):
        grp2 = grp.copy()
        grp2["tick_lower_transformed"] = pd.to_numeric(grp2["tick_lower_transformed"], errors="coerce")
        grp2["tick_upper_transformed"] = pd.to_numeric(grp2["tick_upper_transformed"], errors="coerce")
        grp2["amount_usd_total"] = pd.to_numeric(grp2["amount_usd_total"], errors="coerce")
        ticks_grouped[int(per)] = grp2.sort_values("tick_lower_transformed").reset_index(drop=True)

    df_hours = df.groupby("block").agg({
        "token1Price": "first",
        "feesUSD_real": "first"
    }).reset_index()
    df_hours["block"] = pd.to_numeric(df_hours["block"], errors="coerce").astype("Int64")
    df_hours["token1Price"] = pd.to_numeric(df_hours["token1Price"], errors="coerce")
    df_hours["feesUSD_real"] = pd.to_numeric(df_hours["feesUSD_real"], errors="coerce").fillna(0.0)
    df_hours = df_hours.sort_values("block").reset_index(drop=True)

    # try to infer timestamps from block if unix seconds, else fallback to hourly series
    try:
        ts_candidate = pd.to_datetime(df_hours["block"].astype(int), unit="s", utc=True)
        years = ts_candidate.dt.year
        valid_years = years.dropna()
        if len(valid_years) == 0 or (valid_years.min() < 2000) or (valid_years.max() > 2100):
            raise ValueError("block not unix timestamps")
        df_hours["ts"] = ts_candidate
    except Exception:
        start_ts = pd.to_datetime(START_DATE).tz_localize("UTC") if pd.to_datetime(START_DATE).tzinfo is None else pd.to_datetime(START_DATE).tz_convert("UTC")
        df_hours["ts"] = [start_ts + pd.Timedelta(hours=int(i)) for i in range(len(df_hours))]

    return df_hours, ticks_grouped

# ---------------- APR compute (Bollinger only, fixed params) ----------------
def compute_apr_bollinger_from_hours_globals(df_hours, ticks_grouped,
                                             n=N, k=K, rebalance=REBALANCE,
                                             sample_period=SAMPLE_PERIOD, rebalance_hold=REBALANCE_HOLD,
                                             invest_liq=INVEST_LIQ, gas_penalty=GAS_PENALTY_USD):
    prices = df_hours["token1Price"].astype(float).values
    blocks = df_hours["block"].astype("Int64").values
    fees_map = pd.Series(df_hours["feesUSD_real"].values, index=df_hours["block"].astype(int).values).to_dict()

    Nobs = len(prices)
    if Nobs < n + 1:
        return None

    total_fees_collected = 0.0
    hours_evaluated = 0
    hits_price_in_interval = 0

    # state
    freeze_counter = 0
    current_lower_price = None
    current_upper_price = None

    for t in range(n-1, Nobs-1):
        window = prices[t - (n-1): t+1]
        if np.any(np.isnan(window)):
            freeze_counter = 0
            current_lower_price = None
            current_upper_price = None
            continue

        obs_idx = t - (n-1)
        force_change = False

        # compute candidate band only at sample points
        if (obs_idx % sample_period) == 0:
            std = window.std(ddof=0)
            center = float(window.mean())
            lower_cand = center - k * std
            upper_cand = center + k * std

            if current_lower_price is None:
                current_lower_price = float(lower_cand)
                current_upper_price = float(upper_cand)
                force_change = True
            else:
                if freeze_counter == 0:
                    if (abs(lower_cand - current_lower_price) > 1e-12) or (abs(upper_cand - current_upper_price) > 1e-12):
                        current_lower_price = float(lower_cand)
                        current_upper_price = float(upper_cand)
                        force_change = True

        # detect rebalance: allowed only if freeze_counter == 0
        if rebalance and (freeze_counter == 0):
            if current_lower_price is None:
                std = window.std(ddof=0)
                center_tmp = float(window.mean())
                current_lower_price = float(center_tmp - k*std)
                current_upper_price = float(center_tmp + k*std)
                force_change = True

            width = current_upper_price - current_lower_price
            if width <= 0:
                top_enter = current_upper_price
                bottom_enter = current_lower_price
            else:
                top_enter = current_lower_price + ENTER_FRAC * width
                bottom_enter = current_lower_price + (1.0 - ENTER_FRAC) * width

            y_next = prices[t+1]
            if (y_next > top_enter) or (y_next < bottom_enter):
                # trigger rebalance override: set new center blended with price
                center_band = 0.5 * (current_lower_price + current_upper_price)
                override_value = float(BLEND_ALPHA * float(y_next) + (1.0 - BLEND_ALPHA) * center_band)
                std = window.std(ddof=0)
                new_lo = override_value - k * std
                new_hi = override_value + k * std
                current_lower_price = float(new_lo)
                current_upper_price = float(new_hi)
                force_change = True
                freeze_counter = int(max(0, rebalance_hold))

        # if we intentionally changed band, set freeze and mark change
        changed_range = bool(force_change)
        if force_change:
            freeze_counter = int(max(0, rebalance_hold))

        # decrement freeze_counter once per iteration (do not immediately unfreeze if forced this iter)
        if freeze_counter > 0 and not force_change:
            freeze_counter -= 1

        # ensure band exists
        if current_lower_price is None or current_upper_price is None:
            continue

        block_pred = int(blocks[t+1]) if not pd.isna(blocks[t+1]) else None
        if block_pred is None:
            continue
        ticks_df_for_period = ticks_grouped.get(block_pred)
        if ticks_df_for_period is None or len(ticks_df_for_period) == 0:
            continue

        li_idx, ui_idx = _choose_tick_bounds_from_prices(ticks_df_for_period, current_lower_price, current_upper_price)
        if li_idx is None:
            continue
        selected_ticks = ticks_df_for_period.iloc[li_idx:ui_idx+1].copy()
        if selected_ticks.empty:
            continue

        num_ticks = len(selected_ticks)
        invested_per_tick = float(invest_liq) / float(num_ticks)
        fees_total = float(fees_map.get(block_pred, 0.0))

        tick_lower_adj = selected_ticks["tick_lower_transformed"].iloc[0]
        tick_upper_adj = selected_ticks["tick_upper_transformed"].iloc[-1]
        price_at_obs = float(prices[t+1])

        hours_evaluated += 1

        fees_this_hour = 0.0
        if (price_at_obs >= tick_lower_adj) and (price_at_obs <= tick_upper_adj):
            # compute participation-based fees from ticks inside selected range
            full_mask_contains = ticks_df_for_period.apply(
                lambda r: (r.get("tick_lower_transformed") <= price_at_obs) and (price_at_obs <= r.get("tick_upper_transformed")),
                axis=1
            )
            contains_idx_full = np.where(full_mask_contains.values)[0] if hasattr(full_mask_contains, "values") else np.where(full_mask_contains)[0]

            if len(contains_idx_full) >= 1:
                # choose a tick that contains price (prefer one inside selected_ticks)
                chosen_tick_row = None
                for idx_full in contains_idx_full:
                    row_full = ticks_df_for_period.iloc[idx_full]
                    cond_in_selected = ((selected_ticks["tick_lower_transformed"] == row_full["tick_lower_transformed"]) &
                                        (selected_ticks["tick_upper_transformed"] == row_full["tick_upper_transformed"]))
                    if cond_in_selected.any():
                        chosen_tick_row = row_full
                        break
                if chosen_tick_row is None:
                    chosen_tick_row = ticks_df_for_period.iloc[contains_idx_full[0]]

                tick_total = chosen_tick_row.get("amount_usd_total", np.nan)
                if (pd.isna(tick_total) or tick_total <= 0) and ("pct_tvl_by_tick" in chosen_tick_row and "pool_totalValueLockedUSD_at_block" in chosen_tick_row):
                    try:
                        tick_total = float(chosen_tick_row.get("pct_tvl_by_tick", 0.0)) * float(chosen_tick_row.get("pool_totalValueLockedUSD_at_block", 0.0))
                    except Exception:
                        tick_total = np.nan

                belongs_to_selected = ((selected_ticks["tick_lower_transformed"] == chosen_tick_row["tick_lower_transformed"]) &
                                       (selected_ticks["tick_upper_transformed"] == chosen_tick_row["tick_upper_transformed"])).any()

                if belongs_to_selected and (not pd.isna(tick_total)) and tick_total > 0:
                    participation = invested_per_tick / float(tick_total)
                    fees_this_hour = participation * fees_total
                else:
                    for _, tick_row in selected_ticks.iterrows():
                        tick_total2 = tick_row.get("amount_usd_total", np.nan)
                        if (pd.isna(tick_total2) or tick_total2 <= 0) and ("pct_tvl_by_tick" in tick_row and "pool_totalValueLockedUSD_at_block" in tick_row):
                            try:
                                tick_total2 = float(tick_row.get("pct_tvl_by_tick", 0.0)) * float(tick_row.get("pool_totalValueLockedUSD_at_block", 0.0))
                            except Exception:
                                tick_total2 = np.nan
                        if pd.isna(tick_total2) or tick_total2 <= 0:
                            continue
                        participation2 = invested_per_tick / float(tick_total2)
                        fees_this_hour += participation2 * fees_total

            # count hit when price falls in tick interval
            hits_price_in_interval += 1

        # always subtract gas penalty if we changed range this step (even if fees_this_hour == 0)
        if changed_range:
            fees_this_hour = fees_this_hour - float(gas_penalty)

        total_fees_collected += fees_this_hour

    if hours_evaluated == 0:
        return None

    mean_fees_hour = total_fees_collected / float(hours_evaluated)
    apr_pct = (mean_fees_hour / float(invest_liq)) * 24.0 * 365.0 * 100.0
    apr_usd = mean_fees_hour * 24.0 * 365.0
    hit_rate = float(hits_price_in_interval) / float(hours_evaluated) if hours_evaluated > 0 else 0.0

    return {
        "apr_pct": float(apr_pct),
        "apr_usd": float(apr_usd),
        "total_fees": float(total_fees_collected),
        "hours": int(hours_evaluated),
        "hits": int(hits_price_in_interval),
        "hit_rate": float(hit_rate)
    }

# ---------------- Detailed simulation + plot ----------------
def simulate_bollinger_details(df_hours, ticks_grouped, n=N, k=K, rebalance=REBALANCE,
                               sample_period=SAMPLE_PERIOD, rebalance_hold=REBALANCE_HOLD,
                               invest_liq=INVEST_LIQ, gas_penalty=GAS_PENALTY_USD,
                               out_csv=OUT_HOURLY_DETAIL, out_png=OUT_BANDS_PNG):
    prices = df_hours["token1Price"].astype(float).values
    blocks = df_hours["block"].astype("Int64").values
    ts = df_hours["ts"].values
    fees_map = pd.Series(df_hours["feesUSD_real"].values, index=df_hours["block"].astype(int).values).to_dict()
    Nobs = len(prices)

    rows = []
    upper_list = [np.nan] * Nobs
    lower_list = [np.nan] * Nobs

    freeze_counter = 0
    current_lower_price = None
    current_upper_price = None

    for t in range(n-1, Nobs-1):
        window = prices[t - (n-1): t+1]
        if np.any(np.isnan(window)):
            freeze_counter = 0
            current_lower_price = None
            current_upper_price = None
            continue

        obs_idx = t - (n-1)
        force_change = False

        if (obs_idx % sample_period) == 0:
            std = window.std(ddof=0)
            center = float(window.mean())
            lower_cand = center - k * std
            upper_cand = center + k * std
            if current_lower_price is None:
                current_lower_price = float(lower_cand)
                current_upper_price = float(upper_cand)
                force_change = True
            else:
                if freeze_counter == 0:
                    if (abs(lower_cand - current_lower_price) > 1e-12) or (abs(upper_cand - current_upper_price) > 1e-12):
                        current_lower_price = float(lower_cand)
                        current_upper_price = float(upper_cand)
                        force_change = True

        # rebalance logic (same as compute)
        if rebalance and (freeze_counter == 0):
            if current_lower_price is None:
                std = window.std(ddof=0)
                center_tmp = float(window.mean())
                current_lower_price = float(center_tmp - k*std)
                current_upper_price = float(center_tmp + k*std)
                force_change = True

            width = current_upper_price - current_lower_price
            if width <= 0:
                top_enter = current_upper_price
                bottom_enter = current_lower_price
            else:
                top_enter = current_lower_price + ENTER_FRAC * width
                bottom_enter = current_lower_price + (1.0 - ENTER_FRAC) * width

            y_next = prices[t+1]
            if (y_next > top_enter) or (y_next < bottom_enter):
                center_band = 0.5 * (current_lower_price + current_upper_price)
                override_value = float(BLEND_ALPHA * float(y_next) + (1.0 - BLEND_ALPHA) * center_band)
                std = window.std(ddof=0)
                new_lo = override_value - k * std
                new_hi = override_value + k * std
                current_lower_price = float(new_lo)
                current_upper_price = float(new_hi)
                force_change = True
                freeze_counter = int(max(0, rebalance_hold))

        changed_range = bool(force_change)
        if force_change:
            freeze_counter = int(max(0, rebalance_hold))

        if freeze_counter > 0 and not force_change:
            freeze_counter -= 1

        lower_list[t] = float(current_lower_price) if current_lower_price is not None else np.nan
        upper_list[t] = float(current_upper_price) if current_upper_price is not None else np.nan

        block_pred = int(blocks[t+1]) if not pd.isna(blocks[t+1]) else None
        if block_pred is None:
            continue
        ticks_df_for_period = ticks_grouped.get(block_pred)
        if ticks_df_for_period is None or len(ticks_df_for_period) == 0:
            rows.append({
                "block": block_pred,
                "ts": pd.to_datetime(block_pred, unit="s", utc=True) if isinstance(block_pred, (int, np.integer)) else pd.NaT,
                "price": float(prices[t+1]),
                "lower_price": float(current_lower_price) if current_lower_price is not None else np.nan,
                "upper_price": float(current_upper_price) if current_upper_price is not None else np.nan,
                "tick_lower_adj": np.nan,
                "tick_upper_adj": np.nan,
                "num_ticks": 0,
                "invested_per_tick": np.nan,
                "fees_total_period": float(fees_map.get(block_pred, 0.0)),
                "amount_usd_price_tick": np.nan,
                "fees_generated_by_position": 0.0
            })
            continue

        li_idx, ui_idx = _choose_tick_bounds_from_prices(ticks_df_for_period, current_lower_price, current_upper_price)
        if li_idx is None:
            continue
        selected_ticks = ticks_df_for_period.iloc[li_idx:ui_idx+1].copy()
        if selected_ticks.empty:
            continue

        num_ticks = len(selected_ticks)
        invested_per_tick = float(invest_liq) / float(num_ticks)
        fees_total = float(fees_map.get(block_pred, 0.0))

        tick_lower_adj = selected_ticks["tick_lower_transformed"].iloc[0]
        tick_upper_adj = selected_ticks["tick_upper_transformed"].iloc[-1]
        price_at_obs = float(prices[t+1])

        # compute fees as in compute_apr
        fees_this_hour = 0.0
        amount_usd_price_tick = np.nan
        full_mask_contains = ticks_df_for_period.apply(
            lambda r: (r.get("tick_lower_transformed") <= price_at_obs) and (price_at_obs <= r.get("tick_upper_transformed")),
            axis=1
        )
        contains_idx_full = np.where(full_mask_contains.values)[0] if hasattr(full_mask_contains, "values") else np.where(full_mask_contains)[0]
        if len(contains_idx_full) >= 1:
            row_full = ticks_df_for_period.iloc[contains_idx_full[0]]
            tick_total_full = row_full.get("amount_usd_total", np.nan)
            if (pd.isna(tick_total_full) or tick_total_full <= 0) and ("pct_tvl_by_tick" in row_full and "pool_totalValueLockedUSD_at_block" in row_full):
                try:
                    tick_total_full = float(row_full.get("pct_tvl_by_tick", 0.0)) * float(row_full.get("pool_totalValueLockedUSD_at_block", 0.0))
                except Exception:
                    tick_total_full = np.nan
            if (not pd.isna(tick_total_full)):
                amount_usd_price_tick = float(tick_total_full)

        if (price_at_obs >= tick_lower_adj) and (price_at_obs <= tick_upper_adj):
            # compute fees_this_hour using participation logic
            chosen_tick_row = None
            if len(contains_idx_full) >= 1:
                for idx_full in contains_idx_full:
                    row_full = ticks_df_for_period.iloc[idx_full]
                    cond_in_selected = ((selected_ticks["tick_lower_transformed"] == row_full["tick_lower_transformed"]) &
                                        (selected_ticks["tick_upper_transformed"] == row_full["tick_upper_transformed"]))
                    if cond_in_selected.any():
                        chosen_tick_row = row_full
                        break
                if chosen_tick_row is None:
                    chosen_tick_row = ticks_df_for_period.iloc[contains_idx_full[0]]

            if chosen_tick_row is not None:
                tick_total = chosen_tick_row.get("amount_usd_total", np.nan)
                if (pd.isna(tick_total) or tick_total <= 0) and ("pct_tvl_by_tick" in chosen_tick_row and "pool_totalValueLockedUSD_at_block" in chosen_tick_row):
                    try:
                        tick_total = float(chosen_tick_row.get("pct_tvl_by_tick", 0.0)) * float(chosen_tick_row.get("pool_totalValueLockedUSD_at_block", 0.0))
                    except Exception:
                        tick_total = np.nan

                belongs_to_selected = ((selected_ticks["tick_lower_transformed"] == chosen_tick_row["tick_lower_transformed"]) &
                                       (selected_ticks["tick_upper_transformed"] == chosen_tick_row["tick_upper_transformed"])).any()

                if belongs_to_selected and (not pd.isna(tick_total)) and tick_total > 0:
                    participation = invested_per_tick / float(tick_total)
                    fees_this_hour = participation * fees_total
                else:
                    for _, tick_row in selected_ticks.iterrows():
                        tick_total2 = tick_row.get("amount_usd_total", np.nan)
                        if (pd.isna(tick_total2) or tick_total2 <= 0) and ("pct_tvl_by_tick" in tick_row and "pool_totalValueLockedUSD_at_block" in tick_row):
                            try:
                                tick_total2 = float(tick_row.get("pct_tvl_by_tick", 0.0)) * float(tick_row.get("pool_totalValueLockedUSD_at_block", 0.0))
                            except Exception:
                                tick_total2 = np.nan
                        if pd.isna(tick_total2) or tick_total2 <= 0:
                            continue
                        participation2 = invested_per_tick / float(tick_total2)
                        fees_this_hour += participation2 * fees_total

        # apply gas penalty when intentionally changed band
        if changed_range:
            fees_this_hour = fees_this_hour - float(gas_penalty)

        rows.append({
            "block": block_pred,
            "ts": pd.to_datetime(block_pred, unit="s", utc=True) if isinstance(block_pred, (int, np.integer)) else pd.NaT,
            "price": float(prices[t+1]),
            "lower_price": float(current_lower_price),
            "upper_price": float(current_upper_price),
            "tick_lower_adj": float(tick_lower_adj),
            "tick_upper_adj": float(tick_upper_adj),
            "num_ticks": int(num_ticks),
            "invested_per_tick": float(invested_per_tick),
            "fees_total_period": float(fees_total),
            "amount_usd_price_tick": amount_usd_price_tick,
            "fees_generated_by_position": float(fees_this_hour)
        })

    df_hourly = pd.DataFrame(rows)
    if df_hourly.shape[0] > 0 and out_csv is not None:
        _atomic_write_df(out_csv, df_hourly)

    # plot bands
    if out_png is not None:
        try:
            plt.figure(figsize=(14,6))
            ts_plot = df_hours["ts"]
            price_plot = df_hours["token1Price"].astype(float)
            plt.plot(ts_plot, price_plot, label="price", linewidth=1)
            plt.plot(ts_plot, pd.Series(upper_list), label="upper", linewidth=0.9)
            plt.plot(ts_plot, pd.Series(lower_list), label="lower", linewidth=0.9)
            plt.fill_between(ts_plot, pd.Series(lower_list), pd.Series(upper_list), alpha=0.15)
            plt.title(f"Bollinger bands (n={n}, k={k})")
            plt.xlabel("time (UTC)")
            plt.ylabel("price")
            plt.legend()
            plt.tight_layout()
            tmp_png = out_png + ".tmp"
            plt.savefig(tmp_png, dpi=200)
            plt.close()
            try:
                os.replace(tmp_png, out_png)
            except Exception:
                try:
                    if os.path.exists(out_png):
                        os.remove(out_png)
                    os.rename(tmp_png, out_png)
                except Exception:
                    pass
        except Exception as e:
            print("Error saving bands plot:", e)

    return df_hourly

# ---------------- Runner ----------------
def run_backtest_single(data_csv=DATA_CSV):
    t0 = time.time()
    print("Loading data from:", data_csv)
    df = pd.read_csv(data_csv)
    df = df.sort_values(by=["block", "tick_lower_transformed"])
    df_hours, ticks_grouped = prepare_from_data_backtesting(df)
    print(f"Periods: {len(df_hours)}  Periods with ticks: {len(ticks_grouped)}")

    print(f"Computing APR for Bollinger (n={N}, k={K}, rebalance={REBALANCE}) ...")
    res = compute_apr_bollinger_from_hours_globals(df_hours, ticks_grouped,
                                                   n=N, k=K, rebalance=REBALANCE,
                                                   sample_period=SAMPLE_PERIOD, rebalance_hold=REBALANCE_HOLD,
                                                   invest_liq=INVEST_LIQ, gas_penalty=GAS_PENALTY_USD)
    if res is None:
        print("No valid evaluation (no observations). Exiting.")
        return

    # write results (single row)
    df_out = pd.DataFrame([{
        "strategy": STRATEGY,
        "n": int(N),
        "k": float(K),
        "rebalance": bool(REBALANCE),
        "apr": float(res["apr_pct"]),
        "apr_usd": float(res["apr_usd"]),
        "total_fees": float(res["total_fees"]),
        "hours": int(res["hours"]),
        "hits": int(res["hits"]),
        "hit_rate": float(res["hit_rate"])
    }])
    _atomic_write_df(OUT_RESULTS, df_out)
    print("Results saved:", OUT_RESULTS)

    # simulate detailed and save hourly
    df_hourly = simulate_bollinger_details(df_hours, ticks_grouped, n=N, k=K, rebalance=REBALANCE,
                                           sample_period=SAMPLE_PERIOD, rebalance_hold=REBALANCE_HOLD,
                                           invest_liq=INVEST_LIQ, gas_penalty=GAS_PENALTY_USD,
                                           out_csv=OUT_HOURLY_DETAIL, out_png=OUT_BANDS_PNG)

    elapsed = time.time() - t0
    print(f"Done in {_format_seconds(elapsed)}")
    return df_out, df_hourly

if __name__ == "__main__":
    try:
        run_backtest_single(DATA_CSV)
    except Exception as e:
        print("Fatal error:", e)
        raise

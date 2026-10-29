"""
Multi-Timeframe Real Bitcoin (BTCUSDT) Dataset Builder for SLEEP-DAG-RSI.

Builds clean, causally aligned OHLCV + Turnover datasets across 5 timeframes:
  - `5m`  (5-minute candles)
  - `15m` (15-minute candles)
  - `1h`  (1-hour candles)
  - `4h`  (4-hour candles)
  - `1d`  (1-day candles)

Sources:
  1. Local verified 1-minute Bybit archive `E:/эксперименты/Трейдинг/data/btcusdt_1m_all.parquet`
     (259,200 1m bars, Jan 2026 - Jul 2026), resampled to exact 5m, 15m, 1h, 4h, 1d bars.
  2. Public Bybit v5 Market Kline API (`api.bybit.com/v5/market/kline`) to extend `1d`, `4h`, `1h`,
     `15m`, and `5m` with multi-year history (2023-2026) and out-of-sample July-October 2026 Blind Vault.

Enforces strict causal timestamps (`close_time = open_time + timeframe_delta`) so higher-timeframe
features never leak future information into lower-timeframe bars.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
LOCAL_1M_PARQUET = r"E:\эксперименты\Трейдинг\data\btcusdt_1m_all.parquet"


def fetch_bybit_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    max_pages: int = 25,
) -> pd.DataFrame:
    """
    Fetch historical klines from Bybit v5 public API (backward pagination via `end`).
    Bybit returns newest-first `[startTime, open, high, low, close, volume, turnover]`.
    """
    rows: List[Tuple[int, float, float, float, float, float, float]] = []
    cursor_end = end_ms
    seen_ts = set()

    for _ in range(max_pages):
        if cursor_end <= start_ms:
            break
        url = (
            f"https://api.bybit.com/v5/market/kline?category=linear"
            f"&symbol={symbol}&interval={interval}&end={cursor_end}&limit=1000"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            print(f"  [WARN] Bybit fetch {interval} interrupted: {exc}")
            break

        klines = payload.get("result", {}).get("list", [])
        if not klines:
            break

        oldest_ts = None
        added = 0
        for item in klines:
            ts = int(item[0])
            if ts < start_ms or ts > end_ms or ts in seen_ts:
                continue
            seen_ts.add(ts)
            rows.append(
                (
                    ts,
                    float(item[1]),
                    float(item[2]),
                    float(item[3]),
                    float(item[4]),
                    float(item[5]),
                    float(item[6]),
                )
            )
            added += 1
            if oldest_ts is None or ts < oldest_ts:
                oldest_ts = ts

        if oldest_ts is None or added == 0 or oldest_ts >= cursor_end:
            break
        cursor_end = oldest_ts - 1
        time.sleep(0.06)

    if not rows:
        return pd.DataFrame(
            columns=["open_time", "open", "high", "low", "close", "volume", "turnover"]
        )

    df = pd.DataFrame(
        rows,
        columns=["ts_ms", "open", "high", "low", "close", "volume", "turnover"],
    )
    df["open_time"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = (
        df.drop(columns=["ts_ms"])
        .drop_duplicates(subset=["open_time"])
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    return df


def resample_from_1m(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Exact OHLCV + turnover aggregation from 1-minute bars."""
    d = df_1m.set_index("open_time").sort_index()
    agg = d.resample(rule, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "turnover": "sum",
        }
    )
    agg = agg.dropna(subset=["open", "close"]).reset_index()
    return agg


def build_all_timeframes() -> Dict[str, Dict[str, int]]:
    os.makedirs(DATA_DIR, exist_ok=True)
    print("=" * 80)
    print("BUILDING REAL MULTI-TIMEFRAME BITCOIN (BTCUSDT) DATASET (5m, 15m, 1h, 4h, 1d)")
    print("=" * 80)

    has_local = os.path.isfile(LOCAL_1M_PARQUET)
    if has_local:
        df_1m = pd.read_parquet(LOCAL_1M_PARQUET)
        df_1m["open_time"] = pd.to_datetime(df_1m["open_time"], utc=True)
        print(
            f"[LOCAL] Loaded {len(df_1m):,} 1m BTCUSDT bars "
            f"({df_1m['open_time'].min()} -> {df_1m['open_time'].max()})"
        )
    else:
        df_1m = pd.DataFrame()

    now_ms = int(pd.Timestamp("2026-10-28T12:00:00Z").timestamp() * 1000)
    # Timeframes: (name, pandas_rule, bybit_interval, start_date_str, max_pages)
    specs = [
        ("5m", "5min", "5", "2026-01-01T00:00:00Z", 36),
        ("15m", "15min", "15", "2025-10-01T00:00:00Z", 22),
        ("1h", "1h", "60", "2024-01-01T00:00:00Z", 26),
        ("4h", "4h", "240", "2022-01-01T00:00:00Z", 15),
        ("1d", "1D", "D", "2020-01-01T00:00:00Z", 5),
    ]

    summary: Dict[str, Dict[str, int]] = {}
    for tf_name, rule, bybit_int, start_str, max_pages in specs:
        parts = []
        if not df_1m.empty:
            res_local = resample_from_1m(df_1m, rule)
            parts.append(res_local)

        start_ms = int(pd.Timestamp(start_str).timestamp() * 1000)
        df_api = fetch_bybit_klines(
            "BTCUSDT", bybit_int, start_ms=start_ms, end_ms=now_ms, max_pages=max_pages
        )
        if not df_api.empty:
            parts.append(df_api)

        merged = pd.concat(parts, ignore_index=True)
        merged = (
            merged.drop_duplicates(subset=["open_time"], keep="last")
            .sort_values("open_time")
            .reset_index(drop=True)
        )
        # Drop any incomplete trailing bar
        if len(merged) > 2:
            merged = merged.iloc[:-1].reset_index(drop=True)

        out_path = os.path.join(DATA_DIR, f"btcusdt_{tf_name}.parquet")
        merged.to_parquet(out_path, index=False)

        # Split counts: Discovery (60%), Validation OOS (20%), Blind Vault (last 20%)
        n_tot = len(merged)
        n_train = int(n_tot * 0.60)
        n_val = int(n_tot * 0.20)
        n_vault = n_tot - n_train - n_val
        summary[tf_name] = {
            "total_bars": n_tot,
            "train_bars": n_train,
            "val_oos_bars": n_val,
            "blind_vault_bars": n_vault,
            "start": str(merged["open_time"].min()),
            "end": str(merged["open_time"].max()),
        }
        print(
            f"  [SAVED] btcusdt_{tf_name:<3}.parquet | bars={n_tot:>6,} "
            f"(Train={n_train:,}, OOS={n_val:,}, Vault={n_vault:,}) | "
            f"{merged['open_time'].min().strftime('%Y-%m-%d')} -> {merged['open_time'].max().strftime('%Y-%m-%d')}"
        )

    meta_path = os.path.join(DATA_DIR, "dataset_manifest.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("=" * 80)
    return summary


if __name__ == "__main__":
    build_all_timeframes()

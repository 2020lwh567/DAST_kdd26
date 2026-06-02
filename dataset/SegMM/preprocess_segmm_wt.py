"""
Preprocess SegMM for v1 streaming watch-time experiments.

Input raw columns:
- user_id
- time_ms
- photo_id
- duration_ms
- playing_time
- wtd_effective_view

Output is aligned with the v1 streaming trainer conventions used by KuaiLive:
- `enter_ts_ms`: original interaction timestamp in milliseconds
- `enter_ts` / `timestamp`: interaction timestamp in seconds
- `live_id`: alias of `photo_id`
- `duration_sec`
- `play_time_truncate`: playing time in seconds
- `wtd_effective_view`: effective-view flag from the raw data
- session / rank features
- categorical feature mappings
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FIFTEEN_MIN_S = 15 * 60


def _safe_cat_encode(series: pd.Series):
    s = series.astype(object).where(series.notna(), other="<NA>")
    uniques = sorted(pd.unique(s), key=lambda x: str(x))
    mapping = {val: i for i, val in enumerate(uniques, start=1)}
    codes = s.map(mapping).astype(np.int64)
    return codes, mapping


def _ms_to_sec(series: pd.Series) -> pd.Series:
    ms = pd.to_numeric(series, errors="coerce").fillna(0).astype(np.int64)
    return ((ms + 500) // 1000).astype(np.int64)


def preprocess(input_csv: Path, out_csv: Path, mappings_json: Path, get_demo: int = 0) -> None:
    df = pd.read_csv(input_csv)
    required_cols = ["user_id", "time_ms", "photo_id", "duration_ms", "playing_time", "wtd_effective_view"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"SegMM input missing columns: {missing}")

    if get_demo:
        print(f"Before: len(df) = {len(df)}")
        df = df.head(200000).copy()
        print(f"After: len(df) = {len(df)}")

    df["enter_ts_ms"] = pd.to_numeric(df["time_ms"], errors="coerce").fillna(0).astype(np.int64)
    df["enter_ts"] = _ms_to_sec(df["time_ms"])
    df["timestamp"] = df["enter_ts"]
    df["live_id"] = df["photo_id"]
    df["start_timestamp"] = df.groupby("photo_id")["enter_ts"].transform("min").astype(np.int64)
    df["duration_sec"] = np.maximum(1, _ms_to_sec(df["duration_ms"])).astype(np.int64)
    df["play_time_truncate"] = np.maximum(0, _ms_to_sec(df["playing_time"])).astype(np.int64)
    df["PCR"] = df["play_time_truncate"] / df["duration_sec"]
    df["wtd_effective_view"] = pd.to_numeric(df["wtd_effective_view"], errors="coerce").fillna(0).astype(np.int64)
    df["date"] = (df["enter_ts"] // 86400).astype(np.int64)
    df["leave_ts"] = df["enter_ts"] + df["play_time_truncate"]

    df = df.sort_values(["user_id", "enter_ts", "photo_id"], kind="mergesort").reset_index(drop=True)
    prev_leave = df.groupby("user_id")["leave_ts"].shift(1)
    df["gap_from_prev_leave"] = df["enter_ts"] - prev_leave
    new_session_flag = (
        df["gap_from_prev_leave"].isna()
        | (df["gap_from_prev_leave"] >= FIFTEEN_MIN_S)
        | (df["user_id"] != df["user_id"].shift(1))
    )
    df["session_local_id"] = new_session_flag.groupby(df["user_id"]).cumsum().astype(np.int64)

    u64 = pd.to_numeric(df["user_id"], errors="coerce").fillna(0).astype("uint64")
    s64 = pd.to_numeric(df["session_local_id"], errors="coerce").fillna(0).astype("uint64")
    df["session_id"] = (np.left_shift(u64.to_numpy(), np.uint64(32)) + s64.to_numpy()).astype("uint64")

    df["srank_raw"] = df.groupby(["user_id", "session_local_id"])["enter_ts"].rank(method="first").astype(np.int64)
    df["cumulative_wt"] = (
        df.groupby(["user_id", "session_local_id"])["play_time_truncate"].cumsum() - df["play_time_truncate"]
    )
    denom = (df["srank_raw"] - 1).replace(0, np.nan)
    df["ave_pre_wt_s"] = (df["cumulative_wt"] / denom).fillna(0).astype(np.float64)

    discrete_cols = [
        "user_id",
        "live_id",
    ]
    mappings = {}
    for col in discrete_cols:
        codes, mapping = _safe_cat_encode(df[col])
        df[col] = codes
        mappings[col] = {str(k): int(v) for k, v in mapping.items()}

    srank_uniques = sorted(df["srank_raw"].astype(int).unique().tolist())
    srank_map = {int(v): i for i, v in enumerate(srank_uniques)}
    df["srank"] = df["srank_raw"].map(srank_map).astype(np.int64)
    mappings["srank"] = {str(k): int(v) for k, v in srank_map.items()}

    df["ave_pre_wt_sec"] = np.round(df["ave_pre_wt_s"]).astype(np.int64)
    apw_uniques = sorted(df["ave_pre_wt_sec"].unique().tolist())
    apw_map = {int(v): i for i, v in enumerate(apw_uniques)}
    df["ave_pre_wt"] = df["ave_pre_wt_sec"].map(apw_map).astype(np.int64)
    mappings["ave_pre_wt"] = {str(k): int(v) for k, v in apw_map.items()}

    keep_cols = [
        "user_id",
        "live_id",
        "wtd_effective_view",
        "srank",
        "ave_pre_wt",
        "date",
        "duration_sec",
        "play_time_truncate",
        "PCR",
        "session_id",
        "start_timestamp",
        "timestamp",
        "enter_ts",
        "leave_ts",
        "srank_raw",
        "ave_pre_wt_s",
        "enter_ts_ms",
    ]
    out_df = df[keep_cols].copy()
    for col in [
        "wtd_effective_view",
        "date",
        "duration_sec",
        "play_time_truncate",
        "session_id",
        "timestamp",
        "enter_ts",
        "leave_ts",
        "enter_ts_ms",
        "start_timestamp",
    ]:
        out_df[col] = pd.to_numeric(out_df[col], errors="coerce").fillna(0).astype(np.int64)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    meta = {
        "note": "SegMM discrete feature mappings for v1 streaming watch-time pipeline.",
        "effective_view_column": "wtd_effective_view",
        "session_gap_s": int(FIFTEEN_MIN_S),
        "discrete_columns": discrete_cols + ["srank", "ave_pre_wt"],
        "window_seconds": [60, 120, 240],
    }
    mappings_json.parent.mkdir(parents=True, exist_ok=True)
    with open(mappings_json, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "mappings": mappings}, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved preprocessed CSV -> {out_csv}")
    print(f"[OK] Saved mappings JSON   -> {mappings_json}")
    print(out_df.head(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_csv",
        type=str,
        default="./luoxinchen_hezhiyu_SegMM_inter.csv",
        help="Path to raw SegMM interaction CSV",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="./processed_data_stream/preprocessed_data.csv",
        help="Output CSV path",
    )
    ap.add_argument(
        "--mappings_json",
        type=str,
        default="./processed_data_stream/segmm_feature_mappings.json",
        help="Output mappings JSON",
    )
    ap.add_argument("--get_demo", type=int, default=0, help="Whether to get demo data")
    args = ap.parse_args()
    preprocess(Path(args.input_csv), Path(args.out_csv), Path(args.mappings_json), args.get_demo)

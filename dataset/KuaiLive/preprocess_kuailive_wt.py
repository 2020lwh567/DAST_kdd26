"""
Preprocess KuaiLive for watch-time research in v1.

- preserve `enter_ts_ms`
- convert selected time fields from ms to s
- build session-aware features
- encode categorical features and save mappings
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


def _robust_read(path: Path, expect_cols=None, convert_cols=None):
    if not path.exists():
        return pd.DataFrame(columns=expect_cols or [])
    df = pd.read_csv(path)

    if "click" in str(path):
        # Preserve the raw room-enter timestamp in milliseconds for live-age analysis.
        df["enter_ts_ms"] = df["timestamp"].copy()

    if convert_cols:
        for col in convert_cols:
            ms = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int64)
            sec = (ms + 500) // 1000
            df[col] = sec.astype(np.int64)
    return df


def live_id_check(click: pd.DataFrame, room: pd.DataFrame) -> None:
    common_live_ids = set(click["live_id"]).intersection(set(room["live_id"]))
    if len(common_live_ids) == 0:
        return
    temp_merge = click[click["live_id"].isin(common_live_ids)].merge(
        room[room["live_id"].isin(common_live_ids)],
        on="live_id",
        suffixes=("_click", "_room"),
    )
    assert (temp_merge["streamer_id_click"] == temp_merge["streamer_id_room"]).all(), (
        "存在 live_id 相同但 streamer_id 不同的记录"
    )
    print("✅ 断言通过：所有相同 live_id 的 streamer_id 一致")


def preprocess(in_dir: Path, out_csv: Path, mappings_json: Path, long_view_s: int = 30, get_demo: int = 0):
    in_dir = Path(in_dir)

    click = _robust_read(in_dir / "click.csv", convert_cols=["timestamp", "watch_live_time"])
    user_raw = _robust_read(in_dir / "user.csv")
    room_raw = _robust_read(in_dir / "room.csv", convert_cols=["start_timestamp", "end_timestamp"])
    streamer_raw = _robust_read(in_dir / "streamer.csv")
    comment = _robust_read(in_dir / "comment.csv", convert_cols=["timestamp"])
    like = _robust_read(in_dir / "like.csv", convert_cols=["timestamp"])
    gift = _robust_read(
        in_dir / "gift.csv",
        expect_cols=["user_id", "live_id", "streamer_id", "timestamp", "gift_price"],
        convert_cols=["timestamp"],
    )

    room_raw = room_raw.drop_duplicates(subset=["live_id", "start_timestamp", "end_timestamp"])

    user = user_raw.copy()
    user_cols = {}
    for c in user.columns:
        if c != "user_id":
            user_cols[c] = f"user_{c}"
    if user_cols:
        user = user.rename(columns=user_cols)

    streamer = streamer_raw.copy()
    streamer_cols = {}
    for c in streamer.columns:
        if c != "streamer_id":
            streamer_cols[c] = f"streamer_{c}"
    if streamer_cols:
        streamer = streamer.rename(columns=streamer_cols)

    if get_demo:
        print(f"Before: len(click) = {len(click)}")
        cutoff_time = pd.Timestamp("2025-05-08 00:00:00")
        cutoff_timestamp = int(cutoff_time.timestamp())
        start_cutoff_time = pd.Timestamp("2025-05-07 00:00:00")
        start_cutoff_timestamp = int(start_cutoff_time.timestamp())
        click = click[click["timestamp"] <= cutoff_timestamp]
        click = click[click["timestamp"] >= start_cutoff_timestamp]
        print(f"After: len(click) = {len(click)}")

    live_id_check(click, room_raw)
    room = room_raw.drop(columns=["streamer_id"])

    df = click.merge(room, on="live_id", how="left")
    df = df.merge(user, on="user_id", how="left")
    df = df.merge(streamer, on="streamer_id", how="left")

    for col in ["timestamp", "end_timestamp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int64)
    if "watch_live_time" not in df.columns:
        raise ValueError("click.csv must contain 'watch_live_time'")
    df["watch_live_time"] = pd.to_numeric(df["watch_live_time"], errors="coerce").fillna(0).astype(np.int64)

    remaining_s = np.maximum(0, df["end_timestamp"] - df["timestamp"]).astype(np.int64)
    df["duration_sec"] = remaining_s
    df["play_time_truncate"] = np.clip(df["watch_live_time"], 0, remaining_s).astype(np.int64)
    df["PCR"] = df["play_time_truncate"] / df["duration_sec"]
    assert np.all(np.isfinite(df["PCR"].values))

    df["enter_ts"] = df["timestamp"]
    df["leave_ts"] = np.minimum(df["timestamp"] + df["watch_live_time"], df["end_timestamp"])

    df = df.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)
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

    df["long_view2"] = (df["play_time_truncate"] >= long_view_s).astype(np.int64)
    df["date"] = pd.to_numeric(df["p_date"], errors="coerce").fillna(0).astype(np.int64)

    for ev in [comment, like, gift]:
        if "timestamp" in ev.columns:
            ev["timestamp"] = pd.to_numeric(ev["timestamp"], errors="coerce").fillna(0).astype(np.int64)

    discrete_cols = [
        "user_id",
        "live_id",
        "streamer_id",
        "user_gender",
        "user_age",
        "user_country",
        "user_device_brand",
        "user_device_price",
        "user_fans_num",
        "user_follow_num",
        "user_accu_watch_live_cnt",
        "user_accu_watch_live_duration",
        "user_is_live_streamer",
        "user_is_photo_author",
        "streamer_gender",
        "streamer_age",
        "streamer_country",
        "streamer_device_brand",
        "streamer_device_price",
        "streamer_live_operation_tag",
        "streamer_fans_user_num",
        "streamer_fans_group_fans_num",
        "streamer_follow_user_num",
        "streamer_accu_live_cnt",
        "streamer_accu_live_duration",
        "streamer_accu_play_cnt",
        "streamer_accu_play_duration",
        "live_type",
        "live_content_category",
    ]
    mappings = {}
    for col in discrete_cols:
        if col not in df.columns:
            raise ValueError(f"Required discrete column '{col}' not found.")
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

    prefixed_cols = [c for c in df.columns if c.startswith("user_") or c.startswith("room_") or c.startswith("streamer_")]
    fe_cols = [
        "user_id",
        "live_id",
        "streamer_id",
        "live_type",
        "live_content_category",
        "live_name_id",
        "user_device_brand",
        "streamer_device_brand",
        "user_country",
        "srank",
        "ave_pre_wt",
    ]
    aux_cols = [
        "date",
        "duration_sec",
        "play_time_truncate",
        "PCR",
        "long_view2",
        "session_id",
        "start_timestamp",
        "enter_ts",
        "leave_ts",
        "srank_raw",
        "ave_pre_wt_s",
        "enter_ts_ms",
    ]

    keep_cols = list(dict.fromkeys(fe_cols + aux_cols + prefixed_cols))
    out_df = df[keep_cols].copy()

    for c in ["date", "duration_sec", "play_time_truncate", "session_id", "enter_ts", "leave_ts", "enter_ts_ms"]:
        if c in out_df.columns:
            out_df[c] = pd.to_numeric(out_df[c], errors="coerce").fillna(0).astype(np.int64)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    meta = {
        "note": "KuaiLive discrete feature mappings for current FM-style pipeline (plus srank/ave_pre_wt).",
        "long_view_s": int(long_view_s),
        "session_gap_s": int(FIFTEEN_MIN_S),
        "discrete_columns": discrete_cols + ["srank", "ave_pre_wt"],
    }
    mappings_json.parent.mkdir(parents=True, exist_ok=True)
    with open(mappings_json, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "mappings": mappings}, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved preprocessed CSV -> {out_csv}")
    print(f"[OK] Saved mappings JSON   -> {mappings_json}")
    print(out_df.head(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="dataset/KuaiLive/data", help="Path to KuaiLive directory containing CSVs")
    ap.add_argument(
        "--out_csv",
        type=str,
        default="./processed_data_stream/preprocessed_data.csv",
        help="Output CSV path",
    )
    ap.add_argument(
        "--mappings_json",
        type=str,
        default="./processed_data_stream/kuailive_feature_mappings.json",
        help="Output mappings JSON",
    )
    ap.add_argument("--long_view_s", type=int, default=30, help="Threshold (s) for long_view2 label")
    ap.add_argument("--get_demo", type=int, default=0, help="Whether to get demo data")
    args = ap.parse_args()

    if args.get_demo:
        args.out_csv = "../KuaiLive_demo/processed_data_stream/data.csv"
        args.mappings_json = "../KuaiLive_demo/processed_data_stream/kuailive_feature_mappings.json"

    print(f"args.out_csv: {args.out_csv}")
    print(f"args.mappings_json: {args.mappings_json}")
    preprocess(Path(args.in_dir), Path(args.out_csv), Path(args.mappings_json), args.long_view_s, get_demo=args.get_demo)

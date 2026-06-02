"""
创建单窗口发送的数据流。

与 full-space emission 不同，这里每个脚本只生成一个窗口对应的数据流，但仍保留所有样本。
例如 5min 数据流中，对所有样本统一令：
- `send_timestamp = enter_time + 300s`
- `send_window_id = 300`
- `culm_wt = min(真实观看时长, 5400s)`

默认输出：
- 5min: `./processed_data_stream/stream5min_with_gt_label.csv`
- 30min: `./processed_data_stream/stream30min_with_gt_label.csv`
- 90min: `./processed_data_stream/stream90min_with_gt_label.csv`
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


VALID_WINDOWS = {5, 30, 90}


def _choose_time_col(df: pd.DataFrame, time_col) -> str:
    if time_col:
        if time_col not in df.columns:
            raise ValueError(f"指定 time_col='{time_col}' 不存在于输入数据中")
        print(f"use custom time_col: {time_col}")
        return time_col
    if "timestamp" in df.columns:
        print("use default time_col: timestamp")
        return "timestamp"
    if "enter_ts" in df.columns:
        print("use default time_col: enter_ts")
        return "enter_ts"
    raise ValueError("未找到时间列：请提供 --time_col 或确保存在 timestamp/enter_ts")


def _default_output(window_min: int) -> str:
    return f"./processed_data_stream/stream{window_min}min_with_gt_label.csv"


def build_single_window_stream(
    df: pd.DataFrame,
    window_min: int,
    time_col: str,
    play_col: str = "play_time_truncate",
) -> pd.DataFrame:
    if window_min not in VALID_WINDOWS:
        raise ValueError(f"window_min 必须在 {sorted(VALID_WINDOWS)} 中，当前为 {window_min}")
    if play_col not in df.columns:
        raise ValueError(f"输入数据缺少列: {play_col}")

    df = df.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce").fillna(0).astype("int64")
    df[play_col] = pd.to_numeric(df[play_col], errors="coerce").fillna(0).astype("int64")

    window_sec = window_min * 60
    part = df.copy()
    part["send_timestamp"] = part[time_col] + window_sec
    part["send_window_id"] = window_sec
    # Use the same maximum watch-time horizon as the 90min ORM head.
    part["culm_wt"] = np.minimum(part[play_col], 5400).astype("int64")
    part = part.sort_values("send_timestamp", kind="mergesort").reset_index(drop=True)
    return part


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_csv",
        type=str,
        default="./processed_data_stream/preprocessed_data.csv",
        help="输入 CSV（预处理后的 KuaiLive 数据）",
    )
    ap.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="输出 CSV；默认根据 window_min 自动命名",
    )
    ap.add_argument(
        "--window_min",
        type=int,
        default=5,
        help="单窗口大小（分钟），支持 5 / 30 / 90",
    )
    ap.add_argument(
        "--time_col",
        type=str,
        default=None,
        help="时间戳列名，默认优先 timestamp，否则 enter_ts",
    )
    ap.add_argument(
        "--play_col",
        type=str,
        default="play_time_truncate",
        help="观看时长列名（秒）",
    )
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv or _default_output(args.window_min))
    df = pd.read_csv(input_csv)
    time_col = _choose_time_col(df, args.time_col)
    out_df = build_single_window_stream(df, args.window_min, time_col, play_col=args.play_col)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print(f"[OK] input_csv: {input_csv}")
    print(f"[OK] output_csv: {output_csv}")
    print(f"[OK] time_col: {time_col}")
    print(f"[OK] window_min: {args.window_min}")
    print(out_df.head(3))


if __name__ == "__main__":
    main()

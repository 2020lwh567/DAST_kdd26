"""
创建 SegMM 单窗口发送的数据流。

支持窗口：60 / 120 / 240 秒。
每个单窗口数据流保留所有样本：
- `send_timestamp = enter_time + window_size`
- `send_window_id = window_size`
- `culm_wt = min(real_watch_time, 240)`
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


VALID_WINDOWS = {60, 120, 240}
MAX_WATCH_SEC = 240


def _choose_time_col(df: pd.DataFrame, time_col) -> str:
    if time_col:
        if time_col not in df.columns:
            raise ValueError(f"指定 time_col='{time_col}' 不存在于输入数据中")
        return time_col
    if "timestamp" in df.columns:
        return "timestamp"
    if "enter_ts" in df.columns:
        return "enter_ts"
    raise ValueError("未找到时间列：请提供 --time_col 或确保存在 timestamp/enter_ts")


def _default_output(window_sec: int) -> str:
    return f"./processed_data_stream/stream{window_sec}s_with_gt_label.csv"


def build_single_window_stream(
    df: pd.DataFrame,
    window_sec: int,
    time_col: str,
    play_col: str = "play_time_truncate",
) -> pd.DataFrame:
    if window_sec not in VALID_WINDOWS:
        raise ValueError(f"window_sec 必须在 {sorted(VALID_WINDOWS)} 中，当前为 {window_sec}")
    if play_col not in df.columns:
        raise ValueError(f"输入数据缺少列: {play_col}")

    df = df.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce").fillna(0).astype(np.int64)
    df[play_col] = pd.to_numeric(df[play_col], errors="coerce").fillna(0).astype(np.int64)

    part = df.copy()
    part["send_timestamp"] = part[time_col] + window_sec
    part["send_window_id"] = window_sec
    part["culm_wt"] = np.minimum(part[play_col], MAX_WATCH_SEC).astype(np.int64)
    part = part.sort_values("send_timestamp", kind="mergesort").reset_index(drop=True)
    return part


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, default="./processed_data_stream/preprocessed_data.csv")
    ap.add_argument("--output_csv", type=str, default=None)
    ap.add_argument("--window_sec", type=int, default=60, help="单窗口大小（秒），支持 60 / 120 / 240")
    ap.add_argument("--time_col", type=str, default=None)
    ap.add_argument("--play_col", type=str, default="play_time_truncate")
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv or _default_output(args.window_sec))
    df = pd.read_csv(input_csv)
    time_col = _choose_time_col(df, args.time_col)
    out_df = build_single_window_stream(df, args.window_sec, time_col, play_col=args.play_col)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print(f"[OK] input_csv: {input_csv}")
    print(f"[OK] output_csv: {output_csv}")
    print(f"[OK] time_col: {time_col}")
    print(f"[OK] window_sec: {args.window_sec}")
    print(out_df.head(3))


if __name__ == "__main__":
    main()

"""创建 SegMM 30s-180s 两窗口 full-space emission 数据流。"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


WINDOWS_SEC = [30, 180]


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


def create_two_window_stream(df: pd.DataFrame, time_col: str, play_col: str = "play_time_truncate") -> pd.DataFrame:
    if play_col not in df.columns:
        raise ValueError(f"输入数据缺少列: {play_col}")

    df = df.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce").fillna(0).astype(np.int64)
    df[play_col] = pd.to_numeric(df[play_col], errors="coerce").fillna(0).astype(np.int64)

    parts = []
    for window_sec in WINDOWS_SEC:
        part = df.copy()
        part["send_timestamp"] = part[time_col] + window_sec
        part["send_window_id"] = window_sec
        if window_sec == 30:
            part["culm_wt"] = np.minimum(part[play_col], 30).astype(np.int64)
        else:
            part["culm_wt"] = np.minimum(part[play_col], 180).astype(np.int64)
        parts.append(part)

    out_df = pd.concat(parts, ignore_index=True)
    out_df = out_df.sort_values("send_timestamp", kind="mergesort").reset_index(drop=True)
    return out_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, default="./processed_data_stream/preprocessed_data.csv")
    ap.add_argument("--output_csv", type=str, default="./processed_data_stream/window2_30s_180s_full_space_emit.csv")
    ap.add_argument("--time_col", type=str, default=None)
    ap.add_argument("--play_col", type=str, default="play_time_truncate")
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    df = pd.read_csv(input_csv)
    time_col = _choose_time_col(df, args.time_col)
    out_df = create_two_window_stream(df, time_col, play_col=args.play_col)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print(f"[OK] input_csv: {input_csv}")
    print(f"[OK] output_csv: {output_csv}")
    print(f"[OK] windows_sec: {WINDOWS_SEC}")
    print(out_df.head(3))


if __name__ == "__main__":
    main()

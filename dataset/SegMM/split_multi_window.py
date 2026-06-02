"""
对 SegMM 预处理数据做多窗口拆分（全空间发送版本）。

默认窗口为 60 / 120 / 240 秒。
每条原始样本会在每个窗口发送一次：
- `send_timestamp = enter_time + window_sec`
- `send_window_id = window_sec`
- `culm_wt = min(real_watch_time, window_sec)`
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _parse_windows_sec(windows_sec: str) -> List[int]:
    if not windows_sec:
        raise ValueError("windows_sec 不能为空，例如 '60,120,240'")
    parts = [p.strip() for p in windows_sec.split(",") if p.strip()]
    out = []
    for p in parts:
        try:
            v = int(p)
        except ValueError as exc:
            raise ValueError(f"窗口参数必须为整数秒，非法值: {p}") from exc
        if v <= 0:
            raise ValueError(f"窗口必须为正整数秒，非法值: {v}")
        out.append(v)
    out = sorted(set(out))
    if len(out) == 0:
        raise ValueError("解析后的窗口为空")
    return out


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


def split_multi_window(
    df: pd.DataFrame,
    windows_sec: List[int],
    time_col: str,
    play_col: str = "play_time_truncate",
) -> pd.DataFrame:
    if play_col not in df.columns:
        raise ValueError(f"输入数据缺少列: {play_col}")

    df = df.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce").fillna(0).astype(np.int64)
    df[play_col] = pd.to_numeric(df[play_col], errors="coerce").fillna(0).astype(np.int64)

    windows_sec = sorted(set(int(w) for w in windows_sec))
    parts = []
    for window_sec in windows_sec:
        part = df.copy()
        part["send_timestamp"] = part[time_col] + window_sec
        part["send_window_id"] = window_sec
        part["culm_wt"] = np.minimum(part[play_col], window_sec).astype(np.int64)
        parts.append(part)

    if not parts:
        return df.head(0)

    out_df = pd.concat(parts, ignore_index=True)
    out_df = out_df.sort_values("send_timestamp", kind="mergesort").reset_index(drop=True)
    return out_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, default="./processed_data_stream/preprocessed_data.csv")
    ap.add_argument("--output_csv", type=str, default="./processed_data_stream/window3_60s_120s_240s_full_space_emit.csv")
    ap.add_argument("--windows_sec", type=str, default="60,120,240")
    ap.add_argument("--time_col", type=str, default=None)
    ap.add_argument("--play_col", type=str, default="play_time_truncate")
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    windows_sec = _parse_windows_sec(args.windows_sec)
    print(f"windows_sec: {windows_sec}")

    df = pd.read_csv(input_csv)
    time_col = _choose_time_col(df, args.time_col)
    print(f"time_col: {time_col}")

    out_df = split_multi_window(df, windows_sec, time_col, play_col=args.play_col)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print(f"[OK] input_csv: {input_csv}")
    print(f"[OK] output_csv: {output_csv}")
    print(f"[OK] windows_sec: {windows_sec}")
    print(out_df.head(3))


if __name__ == "__main__":
    main()

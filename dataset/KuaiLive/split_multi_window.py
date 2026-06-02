"""
对 KuaiLive 预处理数据做多窗口拆分（全空间发送版本）。

输入：预处理后的 CSV（每行代表一次观看行为）
输出：追加 `send_timestamp` / `send_window_id` / `culm_wt` 三列，并按 `send_timestamp` 排序。
其中 `send_window_id` 统一使用“秒”作为单位，避免分钟制编码给后续窗口扩展带来歧义。

三窗口命令：
python split_multi_window.py \
  --input_csv "./processed_data_stream/preprocessed_data.csv" \
  --output_csv "./processed_data_stream/window3_full_space_emit.csv" \
  --windows_min "5,30,90"
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _parse_windows_min(windows_min: str) -> List[int]:
    if not windows_min:
        raise ValueError("windows_min 不能为空，例如 '5,30,90'")
    parts = [p.strip() for p in windows_min.split(",") if p.strip()]
    out = []
    for p in parts:
        try:
            v = int(p)
        except ValueError as exc:
            raise ValueError(f"窗口参数必须为整数分钟，非法值: {p}") from exc
        if v <= 0:
            raise ValueError(f"窗口必须为正整数分钟，非法值: {v}")
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
    windows_min: List[int],
    time_col: str,
    play_col: str = "play_time_truncate",
) -> pd.DataFrame:
    if play_col not in df.columns:
        raise ValueError(f"输入数据缺少列: {play_col}")

    df = df.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce").fillna(0).astype(np.int64)
    df[play_col] = pd.to_numeric(df[play_col], errors="coerce").fillna(0).astype(np.int64)

    windows_sec = [w * 60 for w in windows_min]
    if sorted(windows_sec) != windows_sec:
        raise ValueError("windows_min 必须为递增顺序")

    parts = []
    for w_min, w_sec in zip(windows_min, windows_sec):
        # Full-space emission: every original record emits once at every window.
        part = df.copy()
        part["send_timestamp"] = part[time_col] + w_sec
        part["send_window_id"] = w_sec
        part["culm_wt"] = np.minimum(part[play_col], w_sec).astype(np.int64)
        parts.append(part)

    if not parts:
        return df.head(0)

    out_df = pd.concat(parts, ignore_index=True)
    out_df = out_df.sort_values("send_timestamp", kind="mergesort").reset_index(drop=True)
    return out_df


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
        default="./processed_data_stream/window3_full_space_emit.csv",
        help="输出 CSV",
    )
    ap.add_argument(
        "--windows_min",
        type=str,
        default="5,30,90",
        help="窗口列表（分钟），逗号分隔，如 5,30,90",
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
    output_csv = Path(args.output_csv)
    windows_min = _parse_windows_min(args.windows_min)
    print(f"windows_min: {windows_min}")

    df = pd.read_csv(input_csv)
    time_col = _choose_time_col(df, args.time_col)
    print(f"time_col: {time_col}")

    out_df = split_multi_window(df, windows_min, time_col, play_col=args.play_col)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print(f"[OK] input_csv: {input_csv}")
    print(f"[OK] output_csv: {output_csv}")
    print(f"[OK] time_col: {time_col}")
    print(f"[OK] windows_min: {windows_min}")
    print(out_df.head(3))


if __name__ == "__main__":
    main()

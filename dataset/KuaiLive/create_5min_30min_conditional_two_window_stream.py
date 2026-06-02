"""
创建 5min-30min 两窗口 conditional emission 数据流。

设计约定：
- 保留所有原始样本的 5min 发送
- 只有 `staytime >= 5min` 的样本才会进入二阶段 30min 窗口发送
- 每条 5min 样本：
  - `send_timestamp = enter_time + 300`
  - `send_window_id = 300`
  - `culm_wt = min(real_watch_time, 300)`
- 每条 30min 样本：
  - `send_timestamp = enter_time + 1800`
  - `send_window_id = 1800`
  - `culm_wt = real_watch_time`
- `send_window_id` 统一使用“秒”作为单位

默认输出：
- `./processed_data_stream/window2_5min_30min_conditional_emit.csv`
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FIRST_WINDOW_SEC = 300
SECOND_WINDOW_SEC = 1800
WINDOWS_SEC = [FIRST_WINDOW_SEC, SECOND_WINDOW_SEC]


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


def _choose_staytime_col(df: pd.DataFrame, staytime_col: str, play_col: str) -> str:
    if staytime_col:
        if staytime_col not in df.columns:
            raise ValueError(f"指定 staytime_col='{staytime_col}' 不存在于输入数据中")
        return staytime_col
    return play_col


def create_conditional_two_window_stream(
    df: pd.DataFrame,
    time_col: str,
    play_col: str = "play_time_truncate",
    staytime_col: str = None,
) -> pd.DataFrame:
    if play_col not in df.columns:
        raise ValueError(f"输入数据缺少列: {play_col}")
    stay_col = _choose_staytime_col(df, staytime_col, play_col)

    df = df.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce").fillna(0).astype(np.int64)
    df[play_col] = pd.to_numeric(df[play_col], errors="coerce").fillna(0).astype(np.int64)
    df[stay_col] = pd.to_numeric(df[stay_col], errors="coerce").fillna(0).astype(np.int64)

    part_5min = df.copy()
    part_5min["send_timestamp"] = part_5min[time_col] + FIRST_WINDOW_SEC
    part_5min["send_window_id"] = FIRST_WINDOW_SEC
    part_5min["culm_wt"] = np.minimum(part_5min[play_col], FIRST_WINDOW_SEC).astype(np.int64)

    part_30min = df[df[stay_col] >= FIRST_WINDOW_SEC].copy()
    part_30min["send_timestamp"] = part_30min[time_col] + SECOND_WINDOW_SEC
    part_30min["send_window_id"] = SECOND_WINDOW_SEC
    part_30min["culm_wt"] = part_30min[play_col].astype(np.int64)

    out_df = pd.concat([part_5min, part_30min], ignore_index=True)
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
        default="./processed_data_stream/window2_5min_30min_conditional_emit.csv",
        help="输出 CSV",
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
        help="真实观看时长列名（秒）",
    )
    ap.add_argument(
        "--staytime_col",
        type=str,
        default=None,
        help="二阶段条件判断使用的 staytime 列；默认使用 play_col",
    )
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    df = pd.read_csv(input_csv)
    time_col = _choose_time_col(df, args.time_col)
    out_df = create_conditional_two_window_stream(
        df,
        time_col,
        play_col=args.play_col,
        staytime_col=args.staytime_col,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print(f"[OK] input_csv: {input_csv}")
    print(f"[OK] output_csv: {output_csv}")
    print(f"[OK] time_col: {time_col}")
    print(f"[OK] windows_sec: {WINDOWS_SEC}")
    print(f"[OK] output_rows: {len(out_df)}")
    print(out_df.head(3))


if __name__ == "__main__":
    main()

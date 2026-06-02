"""创建 SegMM 60s-120s 两窗口 conditional emission 数据流。"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FIRST_WINDOW_SEC = 60
SECOND_WINDOW_SEC = 120
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

    part_60s = df.copy()
    part_60s["send_timestamp"] = part_60s[time_col] + FIRST_WINDOW_SEC
    part_60s["send_window_id"] = FIRST_WINDOW_SEC
    part_60s["culm_wt"] = np.minimum(part_60s[play_col], FIRST_WINDOW_SEC).astype(np.int64)

    part_120s = df[df[stay_col] >= FIRST_WINDOW_SEC].copy()
    part_120s["send_timestamp"] = part_120s[time_col] + SECOND_WINDOW_SEC
    part_120s["send_window_id"] = SECOND_WINDOW_SEC
    part_120s["culm_wt"] = part_120s[play_col].astype(np.int64)

    out_df = pd.concat([part_60s, part_120s], ignore_index=True)
    out_df = out_df.sort_values("send_timestamp", kind="mergesort").reset_index(drop=True)
    return out_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, default="./processed_data_stream/preprocessed_data.csv")
    ap.add_argument("--output_csv", type=str, default="./processed_data_stream/window2_60s_120s_conditional_emit.csv")
    ap.add_argument("--time_col", type=str, default=None)
    ap.add_argument("--play_col", type=str, default="play_time_truncate")
    ap.add_argument("--staytime_col", type=str, default=None)
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
    print(f"[OK] windows_sec: {WINDOWS_SEC}")
    print(f"[OK] output_rows: {len(out_df)}")
    print(out_df.head(3))


if __name__ == "__main__":
    main()

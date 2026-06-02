from typing import List

import numpy as np
import pandas as pd


def parse_list(s: str) -> List[str]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if not parts:
        raise ValueError("fe_cols 不能为空")
    return parts


def parse_duration_to_seconds(s: str) -> int:
    s = str(s).strip().lower()
    if s.isdigit():
        return int(s)
    if s.endswith("h"):
        return int(float(s[:-1]) * 3600)
    if s.endswith("m"):
        return int(float(s[:-1]) * 60)
    if s.endswith("s"):
        return int(float(s[:-1]))
    return int(float(s))


def parse_t1_to_timestamp(t1: str) -> int:
    s = str(t1).strip()
    if s.isdigit() and len(s) == 8:
        ts = pd.Timestamp(f"{s[:4]}-{s[4:6]}-{s[6:]} 00:00:00").timestamp()
        return int(ts)
    if s.isdigit() and len(s) >= 10:
        return int(s)
    return int(pd.Timestamp(s).timestamp())


def normalize_window_ids_to_seconds(window_ids: pd.Series, model_name: str, valid_window_ids=None) -> pd.Series:
    """
    将数据中的窗口编码统一映射为“秒”。

    兼容两类输入：
    1. 旧数据流：分钟制编码，例如 5/30/90
    2. 新数据流：秒制编码，例如 300/1800/5400

    注意 30 在旧三窗口流里表示 30min，在两窗口 30s-60min 流里表示 30s，
    因此这里需要结合 model_name 判定。
    """
    if valid_window_ids is not None:
        valid_window_ids = [int(v) for v in valid_window_ids]
        window_map = {v: v for v in valid_window_ids}
    elif model_name in {"ORM2W_30S_3600S", "ORM2W_30S_3600S_DELAYED3"}:
        window_map = {
            30: 30,
            3600: 3600,
        }
    else:
        window_map = {
            1: 60,
            5: 300,
            30: 1800,
            90: 5400,
            60: 60,
            300: 300,
            1800: 1800,
            5400: 5400,
        }
    return window_ids.map(window_map).fillna(0).astype(np.int64)

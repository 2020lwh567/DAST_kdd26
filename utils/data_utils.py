from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


@dataclass
class TensorFrame:
    x: torch.Tensor
    y: torch.Tensor
    ts: Optional[np.ndarray] = None
    window_ids: Optional[torch.Tensor] = None
    metric_window_ids: Optional[np.ndarray] = None
    keys: Optional[torch.Tensor] = None
    user_ids: Optional[np.ndarray] = None
    live_ids: Optional[np.ndarray] = None
    enter_ts_ms: Optional[np.ndarray] = None
    start_ts: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(self.x.size(0))

    def slice(self, start_idx: int, end_idx: int) -> "TensorFrame":
        return TensorFrame(
            x=self.x[start_idx:end_idx],
            y=self.y[start_idx:end_idx],
            ts=None if self.ts is None else self.ts[start_idx:end_idx],
            window_ids=None if self.window_ids is None else self.window_ids[start_idx:end_idx],
            metric_window_ids=(
                None if self.metric_window_ids is None else self.metric_window_ids[start_idx:end_idx]
            ),
            keys=None if self.keys is None else self.keys[start_idx:end_idx],
            user_ids=None if self.user_ids is None else self.user_ids[start_idx:end_idx],
            live_ids=None if self.live_ids is None else self.live_ids[start_idx:end_idx],
            enter_ts_ms=None if self.enter_ts_ms is None else self.enter_ts_ms[start_idx:end_idx],
            start_ts=None if self.start_ts is None else self.start_ts[start_idx:end_idx],
        )


def build_tensor_frame(
    df: pd.DataFrame,
    fe_cols: List[str],
    label_col: str,
    send_ts_col: Optional[str] = None,
    window_ids_col: Optional[str] = None,
    metric_window_ids_col: Optional[str] = None,
    include_keys: bool = False,
    user_col: str = "user_id",
    live_col: str = "live_id",
    enter_ts_ms_col: Optional[str] = None,
    start_ts_col: Optional[str] = None,
) -> TensorFrame:
    packed_keys = None
    if include_keys:
        user_ids_arr = df[user_col].to_numpy(dtype=np.int64)
        live_ids_arr = df[live_col].to_numpy(dtype=np.int64)
        packed_keys_np = (user_ids_arr.astype(np.int64) << 32) | (live_ids_arr.astype(np.int64) & ((1 << 32) - 1))
        packed_keys = torch.as_tensor(packed_keys_np, dtype=torch.long)

    return TensorFrame(
        x=torch.as_tensor(df[fe_cols].to_numpy(dtype=np.int64), dtype=torch.long),
        y=torch.as_tensor(df[label_col].to_numpy(dtype=np.float32), dtype=torch.float32),
        ts=None if send_ts_col is None else df[send_ts_col].to_numpy(dtype=np.int64),
        window_ids=(
            None
            if window_ids_col is None
            else torch.as_tensor(df[window_ids_col].to_numpy(dtype=np.int64), dtype=torch.long)
        ),
        metric_window_ids=(
            None if metric_window_ids_col is None else df[metric_window_ids_col].to_numpy(dtype=np.int64)
        ),
        keys=packed_keys,
        user_ids=None if user_col not in df.columns else df[user_col].to_numpy(dtype=np.int64),
        live_ids=None if live_col not in df.columns else df[live_col].to_numpy(dtype=np.int64),
        enter_ts_ms=(
            None if enter_ts_ms_col is None else df[enter_ts_ms_col].to_numpy(dtype=np.int64)
        ),
        start_ts=None if start_ts_col is None else df[start_ts_col].to_numpy(dtype=np.int64),
    )


def get_window_bounds(ts_all: np.ndarray, start_ts: int, end_ts: int) -> Tuple[int, int]:
    start_idx = int(np.searchsorted(ts_all, start_ts, side="left"))
    end_idx = int(np.searchsorted(ts_all, end_ts, side="left"))
    return start_idx, end_idx


def iter_tensor_batches(
    frame: TensorFrame,
    batch_size: int,
    shuffle_within_batch: bool = False,
) -> Iterator[dict]:
    total = len(frame)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        x = frame.x[start:end]
        y = frame.y[start:end]
        window_ids = None if frame.window_ids is None else frame.window_ids[start:end]
        keys = None if frame.keys is None else frame.keys[start:end]

        if shuffle_within_batch and x.size(0) > 1:
            perm = torch.randperm(x.size(0))
            x = x[perm]
            y = y[perm]
            if window_ids is not None:
                window_ids = window_ids[perm]
            if keys is not None:
                keys = keys[perm]

        batch = {"x": x, "y": y}
        if window_ids is not None:
            batch["window_ids"] = window_ids
        if keys is not None:
            batch["keys"] = keys
        yield batch

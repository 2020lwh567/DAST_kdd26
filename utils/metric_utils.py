from typing import Dict, List, Tuple
import pandas as pd

import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score

from utils.xauc import xauc_score


def safe_auc(labels: np.ndarray, preds: np.ndarray) -> float:
    if labels.size < 2:
        return np.nan
    if np.all(labels == labels[0]):
        return np.nan
    return float(roc_auc_score(labels, preds))


def safe_regauc(labels: np.ndarray, preds: np.ndarray) -> float:
    if labels.size < 2:
        return np.nan
    if np.all(labels == labels[0]):
        return np.nan
    return float(xauc_score(labels.reshape(-1), preds.reshape(-1)))


def calc_uauc(labels: np.ndarray, preds: np.ndarray, users: np.ndarray) -> float:
    uauc_vals = []
    for uid in np.unique(users):
        idx = users == uid
        if idx.sum() < 2:
            continue
        lbl = labels[idx]
        if np.all(lbl == lbl[0]):
            continue
        uauc_vals.append(xauc_score(lbl.reshape(-1), preds[idx].reshape(-1)))
    return float(np.mean(uauc_vals)) if uauc_vals else np.nan


def calc_basic_metrics(labels: np.ndarray, preds: np.ndarray, users: np.ndarray) -> dict:
    if labels.size == 0:
        return {"regauc": np.nan, "uauc": np.nan, "rmse": np.nan, "cr": np.nan, "count": 0}
    return {
        "regauc": float(xauc_score(labels.reshape(-1), preds.reshape(-1))),
        "uauc": calc_uauc(labels, preds, users),
        "rmse": float(np.sqrt(mean_squared_error(labels, preds))),
        "cr": float(np.sum(preds) / np.sum(labels)) if np.sum(labels) > 0 else np.nan,
        "count": int(labels.size),
    }


def calc_window_regauc(labels: np.ndarray, preds: np.ndarray, window_ids: np.ndarray, win_id: int) -> float:
    if labels.size == 0:
        return np.nan
    idx = window_ids == win_id
    if not np.any(idx):
        return np.nan
    return safe_regauc(labels[idx], preds[idx])


CVR_THRESHOLDS = [1, 5, 10, 30, 60, 120, 180, 240, 300, 600, 900, 1200, 1500, 1800, 2400, 5400]


def _infer_logit_dim(logit) -> int:
    if logit is None:
        return 0
    if getattr(logit, "ndim", 0) == 0:
        return 0
    return int(logit.shape[-1])


def _build_cvr_map_from_threshold_bins(
    win_id: int,
    logit_dim: int,
    cvr_thresholds: List[int],
    threshold_bins_by_window: Dict[int, List[int]],
) -> Dict[int, int]:
    bins = threshold_bins_by_window.get(int(win_id), [])
    out = {}
    for idx, thr in enumerate(bins[:logit_dim]):
        if int(thr) in cvr_thresholds:
            out[int(thr)] = int(idx)
    return out


def _build_cvr_map_for_window(win_id: int, logit_dim: int, cvr_thresholds: List[int] = None) -> Dict[int, int]:
    cvr_thresholds = CVR_THRESHOLDS if cvr_thresholds is None else cvr_thresholds
    if win_id == 60:
        return {t: t - 1 for t in cvr_thresholds if 1 <= t <= 60 and t <= logit_dim}
    if win_id == 300:
        if logit_dim == 300:
            return {t: t - 1 for t in cvr_thresholds if 1 <= t <= 300}
        if logit_dim == 240:
            return {t: t - 61 for t in cvr_thresholds if 61 <= t <= 300}
        return {}
    if win_id == 1800:
        return {t: (t - 305) // 5 for t in cvr_thresholds if 305 <= t <= 1800 and (t - 305) % 5 == 0}
    if win_id == 5400:
        return {t: (t - 1830) // 30 for t in cvr_thresholds if 1830 <= t <= 5400 and (t - 1830) % 30 == 0}
    return {}


def build_cvr_map(
    logits_by_window: Dict[int, np.ndarray] = None,
    cvr_thresholds: List[int] = None,
    threshold_bins_by_window: Dict[int, List[int]] = None,
):
    cvr_thresholds = CVR_THRESHOLDS if cvr_thresholds is None else cvr_thresholds
    if logits_by_window is None:
        return {
            300: _build_cvr_map_for_window(300, 300, cvr_thresholds),
            1800: _build_cvr_map_for_window(1800, 300, cvr_thresholds),
            5400: _build_cvr_map_for_window(5400, 120, cvr_thresholds),
        }
    out = {}
    for win_id, logit in logits_by_window.items():
        if threshold_bins_by_window is not None:
            mapping = _build_cvr_map_from_threshold_bins(
                int(win_id), _infer_logit_dim(logit), cvr_thresholds, threshold_bins_by_window
            )
        else:
            mapping = _build_cvr_map_for_window(int(win_id), _infer_logit_dim(logit), cvr_thresholds)
        if mapping:
            out[int(win_id)] = mapping
    return out


def calc_cvr_preds_for_sample(logits_by_window=None, logit300=None, logit1800=None, logit5400=None, cvr_map=None, logit60=None):
    if logits_by_window is None:
        logits_by_window = {}
    else:
        logits_by_window = {int(k): v for k, v in logits_by_window.items() if v is not None}
    # Backward-compatible keyword path.
    if logit60 is not None:
        logits_by_window[60] = logit60
    if logit300 is not None:
        logits_by_window[300] = logit300
    if logit1800 is not None:
        logits_by_window[1800] = logit1800
    if logit5400 is not None:
        logits_by_window[5400] = logit5400
    if cvr_map is None:
        cvr_map = build_cvr_map(logits_by_window)
    preds = {}
    for win_id, logit in logits_by_window.items():
        for thr, idx in cvr_map.get(int(win_id), {}).items():
            preds[thr] = float(1 / (1 + np.exp(-logit[idx])))
    return preds


def append_cvr_records(
    store: dict,
    send_ts: np.ndarray,
    labels: np.ndarray,
    window_ids: np.ndarray,
    logits_by_window: dict,
    cvr_thresholds: List[int] = None,
    threshold_bins_by_window: Dict[int, List[int]] = None,
) -> None:
    for win_id, logit in logits_by_window.items():
        cvr_map = build_cvr_map(
            {win_id: logit},
            cvr_thresholds=cvr_thresholds,
            threshold_bins_by_window=threshold_bins_by_window,
        )
        if win_id not in cvr_map:
            continue
        idx = window_ids == win_id
        if not np.any(idx):
            continue
        logit_w = logit[idx]
        ts_w = send_ts[idx]
        y_w = labels[idx]
        for thr, col_idx in cvr_map[win_id].items():
            key = f"cvr_{thr}"
            if key not in store:
                store[key] = {"ts": [], "preds": [], "labels": []}
            preds = 1 / (1 + np.exp(-logit_w[:, col_idx]))
            lbl = (y_w >= thr).astype(np.int64)
            store[key]["ts"].extend(ts_w.tolist())
            store[key]["preds"].extend(preds.tolist())
            store[key]["labels"].extend(lbl.tolist())


def compute_t3_metrics(
    record_ts: List[int],
    record_label: List[float],
    record_pred: List[float],
    record_user: List[int],
    record_window: List[int],
    cvr_store: dict,
    win_start: int,
    win_end: int,
    main_windows: List[int] = None,
    window_metric_names: Dict[int, str] = None,
    cvr_output_threshold_by_actual: Dict[int, int] = None,
) -> dict:
    ts_arr = np.asarray(record_ts, dtype=np.int64)
    mask = (ts_arr >= win_start) & (ts_arr < win_end) if ts_arr.size else np.array([], dtype=bool)
    metrics = calc_basic_metrics(
        np.asarray(record_label, dtype=np.float32)[mask] if ts_arr.size else np.array([], dtype=np.float32),
        np.asarray(record_pred, dtype=np.float32)[mask] if ts_arr.size else np.array([], dtype=np.float32),
        np.asarray(record_user, dtype=np.int64)[mask] if ts_arr.size else np.array([], dtype=np.int64),
    )
    row = {
        "window_start": int(win_start),
        "window_end": int(win_end),
        "count": metrics["count"],
        "regauc": metrics["regauc"],
        "uauc": metrics["uauc"],
        "rmse": metrics["rmse"],
        "cr": metrics["cr"],
    }
    if record_window and ts_arr.size:
        main_windows = [300, 1800, 5400] if main_windows is None else [int(w) for w in main_windows]
        window_metric_names = window_metric_names or {
            main_windows[0]: "5min",
            main_windows[1]: "30min",
            main_windows[2]: "90min",
        }
        win_arr = np.asarray(record_window, dtype=np.int64)
        lbl_arr = np.asarray(record_label, dtype=np.float32)
        pred_arr = np.asarray(record_pred, dtype=np.float32)
        for win_id in main_windows:
            row[f"regauc_{window_metric_names.get(win_id, str(win_id))}"] = calc_window_regauc(
                lbl_arr[mask], pred_arr[mask], win_arr[mask], win_id
            )
    for key, rec in cvr_store.items():
        ts = np.asarray(rec["ts"], dtype=np.int64)
        cvr_mask = (ts >= win_start) & (ts < win_end)
        out_key = key
        if cvr_output_threshold_by_actual:
            try:
                actual_thr = int(str(key).replace("cvr_", ""))
                out_key = f"cvr_{int(cvr_output_threshold_by_actual.get(actual_thr, actual_thr))}"
            except ValueError:
                out_key = key
        row[f"{out_key}_auc"] = safe_auc(
            np.asarray(rec["labels"], dtype=np.int64)[cvr_mask],
            np.asarray(rec["preds"], dtype=np.float32)[cvr_mask],
        )
    return row


def compute_overall_unbias_metrics(first_pred_map: dict, last_label_map: dict) -> dict:
    common_keys = [k for k in first_pred_map.keys() if k in last_label_map]
    if not common_keys:
        return {
            "unbias_count": 0,
            "regauc": np.nan,
            "uauc": np.nan,
            "rmse": np.nan,
            "cr": np.nan,
            "regauc_raw": np.nan,
            "uauc_raw": np.nan,
            "rmse_raw": np.nan,
            "cr_raw": np.nan,
        }
    unbias_preds = np.asarray([first_pred_map[k] for k in common_keys], dtype=np.float32)
    unbias_labels = np.asarray([last_label_map[k][1] for k in common_keys], dtype=np.float32)
    print(f"unbias_labels: max {unbias_labels.max()}, min {unbias_labels.min()}, mean {unbias_labels.mean()}, std {unbias_labels.std()}")
    unbias_users = np.asarray([k[0] for k in common_keys], dtype=np.int64)
    raw_preds = np.asarray([last_label_map[k][2] if len(last_label_map[k]) > 2 else last_label_map[k][1] for k in common_keys], dtype=np.float32)
    return {
        "unbias_count": len(common_keys),
        "regauc": safe_regauc(unbias_labels, unbias_preds),
        "uauc": calc_uauc(unbias_labels, unbias_preds, unbias_users),
        "rmse": float(np.sqrt(mean_squared_error(unbias_labels, unbias_preds))),
        "cr": float(np.sum(unbias_preds) / np.sum(unbias_labels)) if np.sum(unbias_labels) > 0 else np.nan,
        "regauc_raw": safe_regauc(unbias_labels, raw_preds),
        "uauc_raw": calc_uauc(unbias_labels, raw_preds, unbias_users),
        "rmse_raw": float(np.sqrt(mean_squared_error(unbias_labels, raw_preds))),
        "cr_raw": float(np.sum(raw_preds) / np.sum(unbias_labels)) if np.sum(unbias_labels) > 0 else np.nan,
    }


def compute_window_unbias_metrics(
    record_user: List[int],
    record_live: List[int],
    record_label: List[float],
    record_pred: List[float],
    record_window: List[int],
    first_pred_map: dict,
    second_window_id: int = 1800,
    third_window_id: int = 5400,
    second_window_name: str = "30min",
    third_window_name: str = "90min",
) -> dict:
    raw_second_map = {}
    raw_third_map = {}
    win_arr = np.asarray(record_window, dtype=np.int64)
    lbl_arr = np.asarray(record_label, dtype=np.float32)
    pred_arr = np.asarray(record_pred, dtype=np.float32)
    for i in range(len(win_arr)):
        key = (int(record_user[i]), int(record_live[i]))
        if win_arr[i] == second_window_id:
            raw_second_map[key] = (float(lbl_arr[i]), float(pred_arr[i]))
        elif win_arr[i] == third_window_id:
            raw_third_map[key] = (float(lbl_arr[i]), float(pred_arr[i]))

    out = {}
    common_second = list(set(raw_second_map.keys()).intersection(set(first_pred_map.keys())))
    if common_second:
        labels = np.asarray([raw_second_map[k][0] for k in common_second], dtype=np.float32)
        raw_preds = np.asarray([raw_second_map[k][1] for k in common_second], dtype=np.float32)
        unbias_preds = np.asarray([first_pred_map[k] for k in common_second], dtype=np.float32)
        out[f"regauc_{second_window_name}"] = safe_regauc(labels, raw_preds)
        out[f"regauc_{second_window_name}_unbias"] = safe_regauc(labels, unbias_preds)
        out[f"regauc_{second_window_name}_n"] = len(common_second)

    common_third = list(set(raw_third_map.keys()).intersection(set(first_pred_map.keys())))
    if common_third:
        labels = np.asarray([raw_third_map[k][0] for k in common_third], dtype=np.float32)
        raw_preds = np.asarray([raw_third_map[k][1] for k in common_third], dtype=np.float32)
        unbias_preds = np.asarray([first_pred_map[k] for k in common_third], dtype=np.float32)
        out[f"regauc_{third_window_name}"] = safe_regauc(labels, raw_preds)
        out[f"regauc_{third_window_name}_unbias"] = safe_regauc(labels, unbias_preds)
        out[f"regauc_{third_window_name}_n"] = len(common_third)
    return out


def compute_cvr_unbias_metrics(raw_cvr_map: dict, first_cvr_map: dict, output_threshold_by_actual: Dict[int, int] = None) -> dict:
    out = {}
    output_threshold_by_actual = output_threshold_by_actual or {}
    for thr, raw_map in raw_cvr_map.items():
        common_keys = [
            key
            for key in set(raw_map.keys()).intersection(set(first_cvr_map.keys()))
            if thr in first_cvr_map[key]
        ]
        if not common_keys:
            continue
        raw_labels = np.asarray([raw_map[k][0] for k in common_keys], dtype=np.int64)
        raw_preds = np.asarray([raw_map[k][1] for k in common_keys], dtype=np.float32)
        unbias_preds = np.asarray([first_cvr_map[k][thr] for k in common_keys], dtype=np.float32)
        out_thr = int(output_threshold_by_actual.get(thr, thr))
        out[f"cvr_{out_thr}_auc"] = safe_auc(raw_labels, raw_preds)
        out[f"cvr_{out_thr}_unbias"] = safe_auc(raw_labels, unbias_preds)
        out[f"cvr_{out_thr}_n"] = len(common_keys)
    return out


def compute_cvr_first_window_metrics(
    first_cvr_map: dict,
    last_label_map: dict,
    threshold: int = 300,
    output_threshold: int = None,
) -> dict:
    output_threshold = int(threshold if output_threshold is None else output_threshold)
    keys = [
        key
        for key, cvr_preds in first_cvr_map.items()
        if key in last_label_map and threshold in cvr_preds
    ]
    if not keys:
        return {
            f"cvr_{output_threshold}_unbias_auc": np.nan,
            f"cvr_{output_threshold}_cr": np.nan,
            f"cvr_{output_threshold}_n": 0,
        }

    labels = np.asarray([1 if last_label_map[key][1] >= threshold else 0 for key in keys], dtype=np.int64)
    preds = np.asarray([first_cvr_map[key][threshold] for key in keys], dtype=np.float32)
    return {
        f"cvr_{output_threshold}_unbias_auc": safe_auc(labels, preds),
        f"cvr_{output_threshold}_cr": float(np.sum(preds) / np.sum(labels)) if np.sum(labels) > 0 else np.nan,
        f"cvr_{output_threshold}_n": len(keys),
    }


def compute_cvr300_first_window_metrics(first_cvr_map: dict, last_label_map: dict) -> dict:
    return compute_cvr_first_window_metrics(first_cvr_map, last_label_map, threshold=300, output_threshold=300)


def _resolve_live_start_ms_for_early_metrics(
    first_pred_map: dict,
    last_label_map: dict,
    first_seen_ts_ms_map: dict,
    start_ts_map: dict,
) -> Dict[Tuple[int, int], int]:
    valid_keys = [
        key
        for key in first_pred_map.keys()
        if key in last_label_map and key in first_seen_ts_ms_map
    ]
    has_materialized_start = any(int(start_ts_map.get(key, 0)) > 0 for key in valid_keys)
    start_ms_map = {}
    missing_start_keys = []
    for key in valid_keys:
        start_ts = int(start_ts_map.get(key, 0))
        if start_ts > 0 or (has_materialized_start and key in start_ts_map):
            start_ms_map[key] = start_ts * 1000
        else:
            missing_start_keys.append(key)

    if missing_start_keys:
        live_min_seen_ms = {}
        for key in missing_start_keys:
            live_id = key[1]
            seen_ms = int(first_seen_ts_ms_map[key])
            prev = live_min_seen_ms.get(live_id)
            if prev is None or seen_ms < prev:
                live_min_seen_ms[live_id] = seen_ms
        for key in missing_start_keys:
            # Fallback for legacy SegMM streams generated before start_timestamp
            # was materialized during preprocessing.
            start_ms_map[key] = live_min_seen_ms.get(key[1], int(first_seen_ts_ms_map[key]))
    return start_ms_map


def compute_early_live_metrics(
    first_pred_map: dict,
    first_cvr_map: dict,
    last_label_map: dict,
    first_seen_ts_ms_map: dict,
    start_ts_map: dict,
    k_minutes: List[int],
    cvr_threshold: int = 300,
    cvr_output_threshold: int = None,
) -> dict:
    out = {}
    cvr_output_threshold = int(cvr_threshold if cvr_output_threshold is None else cvr_output_threshold)
    live_start_ms_map = _resolve_live_start_ms_for_early_metrics(
        first_pred_map,
        last_label_map,
        first_seen_ts_ms_map,
        start_ts_map,
    )
    live_id_evaluated = None
    for k in k_minutes:
        k_ms = k * 60 * 1000
        keys_k = [
            kk
            for kk in first_pred_map.keys()
            if kk in last_label_map
            and kk in first_seen_ts_ms_map
            and kk in live_start_ms_map
            and (first_seen_ts_ms_map[kk] - live_start_ms_map[kk] <= k_ms)
        ]
        if live_id_evaluated is None:
            live_id_evaluated = {kk[1] for kk in keys_k}
        keys_k = [kk for kk in keys_k if kk[1] in live_id_evaluated]
        if keys_k:
            labels = np.asarray([last_label_map[kk][1] for kk in keys_k], dtype=np.float32)
            preds = np.asarray([first_pred_map[kk] for kk in keys_k], dtype=np.float32)
            out[f"regauc_unbias_k{k}m"] = safe_regauc(labels, preds)
            out[f"cr_unbias_k{k}m"] = float(np.sum(preds) / np.sum(labels)) if np.sum(labels) > 0 else np.nan
            out[f"regauc_unbias_k{k}m_n"] = len(keys_k)
            out[f"cr_unbias_k{k}m_n"] = len(keys_k)
        else:
            out[f"regauc_unbias_k{k}m"] = np.nan
            out[f"cr_unbias_k{k}m"] = np.nan
            out[f"regauc_unbias_k{k}m_n"] = 0
            out[f"cr_unbias_k{k}m_n"] = 0

        cvr_keys = [kk for kk in keys_k if kk in first_cvr_map and cvr_threshold in first_cvr_map[kk]]
        if cvr_keys:
            labels = np.asarray([1 if last_label_map[kk][1] >= cvr_threshold else 0 for kk in cvr_keys], dtype=np.int64)
            preds = np.asarray([first_cvr_map[kk][cvr_threshold] for kk in cvr_keys], dtype=np.float32)
            out[f"cvr_{cvr_output_threshold}_unbias_k{k}m"] = safe_auc(labels, preds)
            out[f"cvr_{cvr_output_threshold}_cr_unbias_k{k}m"] = float(np.sum(preds) / np.sum(labels)) if np.sum(labels) > 0 else np.nan
            out[f"cvr_{cvr_output_threshold}_unbias_k{k}m_n"] = len(cvr_keys)
            out[f"cvr_{cvr_output_threshold}_cr_unbias_k{k}m_n"] = len(cvr_keys)
        else:
            out[f"cvr_{cvr_output_threshold}_unbias_k{k}m"] = np.nan
            out[f"cvr_{cvr_output_threshold}_cr_unbias_k{k}m"] = np.nan
            out[f"cvr_{cvr_output_threshold}_unbias_k{k}m_n"] = 0
            out[f"cvr_{cvr_output_threshold}_cr_unbias_k{k}m_n"] = 0
    return out


def build_user_stratification_payload(
    df: pd.DataFrame,
    user_col: str,
    feature_cols: List[str],
    feature_mappings: Dict[str, Dict[str, int]] = None,
    num_bins: int = 5,
) -> dict:
    user_df = df[[user_col] + feature_cols].groupby(user_col, sort=False).tail(1).copy()
    payload = {
        "user_col": user_col,
        "source": "last_observation_per_user",
        "num_users_total": int(len(user_df)),
        "num_bins_requested": int(num_bins),
        "feature_group_maps": {},
        "feature_group_meta": {},
    }

    for feature_col in feature_cols:
        values = pd.to_numeric(user_df[feature_col], errors="coerce")
        group_labels = pd.Series(["missing"] * len(user_df), index=user_df.index, dtype="object")
        group_meta = []
        mapping_for_feature = None if feature_mappings is None else feature_mappings.get(feature_col)

        valid_mask = values.notna()
        valid_values = values[valid_mask]
        if mapping_for_feature:
            payload["source"] = "last_observation_per_user_with_preprocessed_buckets"
            code_to_bucket = {int(code): str(raw_bucket) for raw_bucket, code in mapping_for_feature.items()}
            valid_codes = valid_values.astype(np.int64)
            observed_codes = sorted(set(valid_codes.tolist()))
            for code in observed_codes:
                group_name = f"bin{int(code)}"
                code_mask = valid_codes == code
                group_labels.loc[valid_codes.index[code_mask]] = group_name
                bucket_label = code_to_bucket.get(int(code), str(int(code)))
                group_meta.append(
                    {
                        "group": group_name,
                        "bucket_code": int(code),
                        "bucket_label": bucket_label,
                        "interval": bucket_label,
                        "left": None,
                        "right": None,
                    }
                )
        else:
            unique_count = int(valid_values.nunique())
            if valid_values.empty:
                pass
            elif unique_count <= 1:
                group_labels.loc[valid_mask] = "bin1"
                scalar = float(valid_values.iloc[0])
                interval_text = f"[{scalar}, {scalar}]"
                group_meta.append(
                    {
                        "group": "bin1",
                        "bucket_code": 1,
                        "bucket_label": interval_text,
                        "interval": interval_text,
                        "left": scalar,
                        "right": scalar,
                    }
                )
            else:
                q = max(1, min(int(num_bins), unique_count))
                bucket = pd.qcut(valid_values, q=q, duplicates="drop")
                bucket_codes = bucket.cat.codes
                categories = bucket.cat.categories
                actual_bins = len(categories)
                for idx in range(actual_bins):
                    group_name = f"bin{idx + 1}"
                    code_mask = bucket_codes == idx
                    group_labels.loc[valid_values.index[code_mask]] = group_name
                    interval = categories[idx]
                    interval_text = f"[{float(interval.left)}, {float(interval.right)}]"
                    group_meta.append(
                        {
                            "group": group_name,
                            "bucket_code": idx + 1,
                            "bucket_label": interval_text,
                            "interval": interval_text,
                            "left": float(interval.left),
                            "right": float(interval.right),
                        }
                    )

        payload["feature_group_maps"][feature_col] = {
            int(uid): str(group)
            for uid, group in zip(user_df[user_col].tolist(), group_labels.tolist())
        }
        payload["feature_group_meta"][feature_col] = group_meta

    return payload


def compute_user_group_stratified_unbias_metrics(
    first_pred_map: dict,
    last_label_map: dict,
    first_seen_ts_ms_map: dict,
    start_ts_map: dict,
    group_map: Dict[int, str],
    group_meta: List[dict],
) -> List[dict]:
    common_keys = [k for k in first_pred_map.keys() if k in last_label_map]
    one_min_keys = [
        k
        for k in common_keys
        if k in first_seen_ts_ms_map
        and k in start_ts_map
        and (first_seen_ts_ms_map[k] - start_ts_map[k] * 1000 <= 60 * 1000)
    ]
    grouped = {}
    grouped_1min = {}
    for key in common_keys:
        uid = int(key[0])
        group_name = group_map.get(uid, "missing")
        grouped.setdefault(group_name, {"labels": [], "preds": [], "users": [], "keys": 0})
        grouped[group_name]["labels"].append(float(last_label_map[key][1]))
        grouped[group_name]["preds"].append(float(first_pred_map[key]))
        grouped[group_name]["users"].append(uid)
        grouped[group_name]["keys"] += 1
    for key in one_min_keys:
        uid = int(key[0])
        group_name = group_map.get(uid, "missing")
        grouped_1min.setdefault(group_name, {"labels": [], "preds": []})
        grouped_1min[group_name]["labels"].append(float(last_label_map[key][1]))
        grouped_1min[group_name]["preds"].append(float(first_pred_map[key]))

    ordered_groups = [meta["group"] for meta in group_meta]
    if "missing" in grouped and "missing" not in ordered_groups:
        ordered_groups.append("missing")

    results = []
    meta_map = {meta["group"]: meta for meta in group_meta}
    for group_name in ordered_groups:
        rec = grouped.get(group_name)
        meta = meta_map.get(
            group_name,
            {
                "group": group_name,
                "bucket_code": None,
                "bucket_label": None,
                "interval": None,
                "left": None,
                "right": None,
            },
        )
        if rec is None:
            results.append(
                {
                    "group": group_name,
                    "bucket_code": meta.get("bucket_code"),
                    "bucket_label": meta.get("bucket_label"),
                    "interval": meta["interval"],
                    "left": meta["left"],
                    "right": meta["right"],
                    "sample_count": 0,
                    "user_count": 0,
                    "sample_count_1min": 0,
                    "unbias_regauc": np.nan,
                    "unbias_uauc": np.nan,
                    "unbias_rmse": np.nan,
                    "unbias_cr": np.nan,
                    "unbias_regauc_1min": np.nan,
                }
            )
            continue

        labels = np.asarray(rec["labels"], dtype=np.float32)
        preds = np.asarray(rec["preds"], dtype=np.float32)
        users = np.asarray(rec["users"], dtype=np.int64)
        rec_1min = grouped_1min.get(group_name)
        if rec_1min is not None:
            labels_1min = np.asarray(rec_1min["labels"], dtype=np.float32)
            preds_1min = np.asarray(rec_1min["preds"], dtype=np.float32)
            regauc_1min = safe_regauc(labels_1min, preds_1min)
            sample_count_1min = int(len(labels_1min))
        else:
            regauc_1min = np.nan
            sample_count_1min = 0
        results.append(
            {
                "group": group_name,
                "bucket_code": meta.get("bucket_code"),
                "bucket_label": meta.get("bucket_label"),
                "interval": meta["interval"],
                "left": meta["left"],
                "right": meta["right"],
                "sample_count": int(rec["keys"]),
                "user_count": int(np.unique(users).size),
                "sample_count_1min": sample_count_1min,
                "unbias_regauc": safe_regauc(labels, preds),
                "unbias_uauc": calc_uauc(labels, preds, users),
                "unbias_rmse": float(np.sqrt(mean_squared_error(labels, preds))),
                "unbias_cr": float(np.sum(preds) / np.sum(labels)) if np.sum(labels) > 0 else np.nan,
                "unbias_regauc_1min": regauc_1min,
            }
        )
    return results


def count_shared_and_private_params(model) -> Tuple[int, float]:
    shared_params = 0
    if hasattr(model, "embedding"):
        shared_params += int(sum(p.numel() for p in model.embedding.parameters()))
    if hasattr(model, "bottom"):
        shared_params += int(sum(p.numel() for p in model.bottom.parameters()))
    if hasattr(model, "orm_heads"):
        head_params = int(sum(p.numel() for p in model.orm_heads.parameters()))
    elif hasattr(model, "head"):
        head_params = int(sum(p.numel() for p in model.head.parameters()))
    elif hasattr(model, "heads"):
        head_params = int(sum(p.numel() for p in model.heads.parameters()))
    else:
        head_params = np.nan
    return shared_params, head_params

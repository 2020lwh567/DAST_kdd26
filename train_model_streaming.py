import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.nn import MSELoss
from torch.optim import Adam

from model.model_factory import build_model
from utils.save_res import save_metrics_to_excel
from utils.set_seed import setup_seed
from utils.streaming_utils import (
    normalize_window_ids_to_seconds,
    parse_duration_to_seconds,
    parse_list,
    parse_t1_to_timestamp,
)
from utils.dataset_config import get_dataset_config
from utils.memory_bank import MemoryBank
from utils.profile_utils import count_total_params, estimate_forward_flops
from utils.metric_utils import (
    append_cvr_records,
    build_user_stratification_payload,
    build_cvr_map,
    calc_basic_metrics,
    calc_cvr_preds_for_sample,
    calc_window_regauc,
    compute_user_group_stratified_unbias_metrics,
    compute_cvr_first_window_metrics,
    compute_cvr_unbias_metrics,
    compute_early_live_metrics,
    compute_overall_unbias_metrics,
    compute_t3_metrics,
    compute_window_unbias_metrics,
    count_shared_and_private_params,
)
from utils.data_utils import TensorFrame, build_tensor_frame, get_window_bounds, iter_tensor_batches
from model.window_utils import build_three_window_thresholds


def calc_field_dims(df: pd.DataFrame, fe_cols: List[str]) -> List[int]:
    offset = 0 if 0 in df[fe_cols[0]].unique() else 1
    return [int(df[c].nunique()) + offset for c in fe_cols]


def encode_window_id(df: pd.DataFrame, col: str) -> str:
    new_col = f"{col}_enc"
    df[new_col] = pd.Categorical(df[col]).codes.astype(np.int64) + 1
    return new_col


def _window_metric_names(main_windows: Tuple[int, int, int]) -> dict:
    return {
        int(main_windows[0]): "5min",
        int(main_windows[1]): "30min",
        int(main_windows[2]): "90min",
    }


def _build_cvr_threshold_bins(main_windows: Tuple[int, int, int]) -> dict:
    return build_three_window_thresholds(main_windows)


def _build_raw_cvr_thresholds_by_window(main_windows: Tuple[int, int, int], cvr_thresholds: List[int]) -> dict:
    first, second, third = [int(w) for w in main_windows]
    return {
        second: [thr for thr in cvr_thresholds if first < thr <= second],
        third: [thr for thr in cvr_thresholds if second < thr <= third],
    }


def train_one_pass(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    frame: TensorFrame,
    batch_size: int,
    device: torch.device,
    shuffle_within_batch: bool,
    is_pretrain: bool = False,
) -> float:
    if len(frame) == 0:
        return 0.0
    loss_fn = MSELoss()
    model.train()
    losses_list = []
    step = 0
    for batch in iter_tensor_batches(frame, batch_size=batch_size, shuffle_within_batch=shuffle_within_batch):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        optimizer.zero_grad()
        if hasattr(model, "compute_loss"):
            if frame.window_ids is None:
                raise ValueError("ORM3W 训练需要 window_ids_col")
            window_ids = batch["window_ids"].to(device, non_blocking=True)
            if getattr(model, "requires_keys", False):
                keys = batch["keys"].to(device, non_blocking=True)
                out = model.compute_loss(x, y, window_ids, keys)
            else:
                out = model.compute_loss(x, y, window_ids)
            loss = out[0] if isinstance(out, tuple) else out
        else:
            pred = model(x).view(-1)
            loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        losses_list.append(float(loss.item()))
        if is_pretrain:
            step += 1
            if step % 200 == 1:
                print(f"[PRETRAIN] step {step}, batch loss: {loss.item()}")
    return float(np.mean(losses_list)) if losses_list else 0.0


def predict(
    model: torch.nn.Module,
    frame: TensorFrame,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if len(frame) == 0:
        return np.array([], dtype=np.float32)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in iter_tensor_batches(frame, batch_size=batch_size, shuffle_within_batch=False):
            x = batch["x"].to(device, non_blocking=True)
            pred = (
                model.predict(x).view(-1).detach().cpu().numpy()
                if hasattr(model, "predict")
                else model(x).view(-1).detach().cpu().numpy()
            )
            preds.append(pred)
    return np.concatenate(preds, axis=0)


def predict_with_logits(
    model: torch.nn.Module,
    frame: TensorFrame,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, dict]:
    if len(frame) == 0:
        return np.array([], dtype=np.float32), {}
    model.eval()
    preds = []
    logits_by_window = {}
    with torch.no_grad():
        for batch in iter_tensor_batches(frame, batch_size=batch_size, shuffle_within_batch=False):
            x = batch["x"].to(device, non_blocking=True)
            pred, logits = model.predict_with_logits(x)
            preds.append(pred.view(-1).detach().cpu().numpy())
            for key, value in logits.items():
                win_id = int(key)
                logits_by_window.setdefault(win_id, []).append(value.detach().cpu().numpy())
    logits_np = {win_id: np.concatenate(parts, axis=0) for win_id, parts in logits_by_window.items()}
    return np.concatenate(preds, axis=0), logits_np


def update_memory_bank(
    bank: MemoryBank,
    frame: TensorFrame,
    preds: np.ndarray,
    logits_np: Union[dict, None],
    cvr_map: dict,
    first_window_id: int = 300,
    raw_cvr_thresholds_by_window: dict = None,
) -> None:
    if frame.ts is None or frame.user_ids is None or frame.live_ids is None:
        raise ValueError("更新 MemoryBank 需要时间戳、user_id 和 live_id")
    label_arr = frame.y.numpy()
    bank.record_ts.extend(frame.ts)
    bank.record_user.extend(frame.user_ids)
    bank.record_live.extend(frame.live_ids)
    bank.record_pred.extend(preds.astype(np.float32))
    bank.record_label.extend(label_arr.astype(np.float32))

    if frame.metric_window_ids is None:
        return

    win_arr = frame.metric_window_ids
    bank.record_window.extend(win_arr)
    user_arr = frame.user_ids
    live_arr = frame.live_ids
    enter_arr = frame.enter_ts_ms
    start_arr = frame.start_ts
    if enter_arr is None or start_arr is None:
        raise ValueError("更新无偏指标缓存需要 enter_ts_ms 和 start_timestamp")

    for i in range(len(win_arr)):
        key = (int(user_arr[i]), int(live_arr[i]))
        win_id = int(win_arr[i])
        if win_id == first_window_id and key not in bank.first_pred_map:
            bank.first_pred_map[key] = float(preds[i])
            bank.first_seen_ts_ms_map[key] = int(enter_arr[i])
            bank.start_ts_map[key] = int(start_arr[i])
            if logits_np is not None:
                logits_for_sample = {int(win): logit[i] for win, logit in logits_np.items()}
                bank.first_cvr_map[key] = calc_cvr_preds_for_sample(
                    logits_by_window=logits_for_sample,
                    cvr_map=cvr_map,
                )
        prev = bank.last_label_map.get(key)
        if prev is None or win_id > prev[0]:
            bank.last_label_map[key] = (win_id, float(label_arr[i]), float(preds[i]))
        if logits_np is not None:
            logits_for_sample = {int(win): logit[i] for win, logit in logits_np.items()}
            cvr_pred_map = calc_cvr_preds_for_sample(
                logits_by_window=logits_for_sample,
                cvr_map=cvr_map,
            )
            for thr in (raw_cvr_thresholds_by_window or {}).get(win_id, []):
                if thr in cvr_pred_map:
                    bank.raw_cvr_map.setdefault(thr, {})[key] = (
                        1 if label_arr[i] >= thr else 0,
                        cvr_pred_map[thr],
                    )


def resolve_first_window_metric_id(model: torch.nn.Module, model_name: str, metric_window_ids_col: Union[str, None], df: pd.DataFrame) -> int:
    first_window_id = int(getattr(model, "metric_first_window_id", getattr(model, "first_window_id", 300)))
    if model_name in {"ORM5_STREAM", "ORM30_STREAM", "ORM90_STREAM"}:
        if metric_window_ids_col is not None:
            unique_window_ids = sorted(pd.unique(df[metric_window_ids_col].dropna()).astype(np.int64).tolist())
            if len(unique_window_ids) == 1:
                first_window_id = int(unique_window_ids[0])
            elif first_window_id not in unique_window_ids:
                raise ValueError(
                    f"{model_name} 期望 first_window_id={first_window_id}, "
                    f"但数据流窗口为 {unique_window_ids}"
                )
    return first_window_id


def split_pretrain_online_by_t1(df: pd.DataFrame, t1: str, send_ts_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    t1_text = str(t1).strip()
    if t1_text.endswith("%"):
        pct_text = t1_text[:-1].strip()
        if not pct_text:
            raise ValueError("百分比形式的 t1 不能为空，例如 30%")
        pct = float(pct_text)
        if pct < 0 or pct > 100:
            raise ValueError(f"百分比形式的 t1 必须在 [0, 100] 内，当前为 {t1}")
        split_idx = int(len(df) * pct / 100.0)
        pre_df = df.iloc[:split_idx].copy()
        online_df = df.iloc[split_idx:].copy()
        if len(online_df) > 0:
            t1_ts = int(online_df[send_ts_col].iloc[0])
        elif len(df) > 0:
            t1_ts = int(df[send_ts_col].iloc[-1]) + 1
        else:
            t1_ts = 0
        print(
            f"[OK] t1 percent split: t1={t1_text}, pretrain_rows={len(pre_df)}, "
            f"online_rows={len(online_df)}, boundary_send_ts={t1_ts}"
        )
        return pre_df, online_df, t1_ts

    t1_ts = parse_t1_to_timestamp(t1_text)
    pre_df = df[df[send_ts_col] < t1_ts]
    online_df = df[df[send_ts_col] >= t1_ts]
    return pre_df, online_df, t1_ts


def run_streaming(args: argparse.Namespace) -> None:
    setup_seed(args.randseed)
    dataset_config = get_dataset_config(args.dat_name)
    window_spec = dataset_config.window_spec
    args.main_windows = window_spec.main_windows
    args.short_long_windows = window_spec.short_long_windows
    main_windows = tuple(int(w) for w in args.main_windows)
    cvr_thresholds = window_spec.cvr_thresholds
    cvr_output_threshold_by_actual = {int(v): int(k) for k, v in window_spec.cvr_threshold_map.items()}
    cvr_threshold_bins = _build_cvr_threshold_bins(main_windows)
    raw_cvr_thresholds_by_window = _build_raw_cvr_thresholds_by_window(main_windows, cvr_thresholds)
    window_metric_names = _window_metric_names(main_windows)
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    job_name = f"{args.dat_name}_{args.model_name}_t1{args.t1}_t2_{args.t2}_t3_{args.t3}_bsz{args.batch_size}_lr{args.lr}_seed{args.randseed}_fullFeat{args.use_full_features}_hid{args.hidden_dim}"
    if args.model_name in {"ORM3W_DELAYED3", "ORM2W_30S_3600S_DELAYED3"}:
        job_name += f"_align_weight{args.align_weight}"
    if args.model_name in {"DEFER-5-30", "DEFER-5-90", "ES-DFM-5-30", "ES-DFM-5-90"}:
        job_name += f"_fakeNeg{args.fake_negative_weight}"
    if args.backbone != "MLP":
        job_name += f"_backbone{args.backbone}"
    print(f"[OK] job_name: {job_name}")

    df = pd.read_csv(args.input_csv)
    print(f"[OK] read csv: {args.input_csv} successfully")
    required = [args.label_col, args.send_ts_col, args.user_col, "live_id", args.start_ts_col, args.enter_ts_ms_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"输入数据缺少列: {col}")
    df[args.send_ts_col] = pd.to_numeric(df[args.send_ts_col], errors="coerce").fillna(0).astype(np.int64)
    df[args.label_col] = pd.to_numeric(df[args.label_col], errors="coerce").fillna(0).astype(np.float32)

    fe_cols = parse_list(args.fe_cols)
    if args.use_window_id:
        if args.window_id_col not in df.columns:
            raise ValueError(f"输入数据缺少窗口列: {args.window_id_col}")
        fe_cols = fe_cols + [encode_window_id(df, args.window_id_col)]
    if args.use_full_features == 1 and args.dat_name == "KuaiLive":
        fe_cols = fe_cols + [
            "live_type", "live_content_category", "user_device_brand", "streamer_device_brand", "user_country",
            "user_age", "user_gender", "user_device_price", "user_fans_num", "user_follow_num",
            "user_accu_watch_live_cnt", "user_accu_watch_live_duration", "user_is_live_streamer", "user_is_photo_author",
            "streamer_gender", "streamer_age", "streamer_country", "streamer_device_price",
            "streamer_live_operation_tag", "streamer_fans_user_num", "streamer_fans_group_fans_num",
            "streamer_follow_user_num", "streamer_accu_live_cnt", "streamer_accu_live_duration",
            "streamer_accu_play_cnt", "streamer_accu_play_duration",
        ]
    elif args.use_full_features == 1:
        print(f"[WARN] {args.dat_name} does not define KuaiLive full feature columns; use base fe_cols only.")
    for c in fe_cols:
        if c not in df.columns:
            raise ValueError(f"输入数据缺少特征列: {c}")
    print(f"fe_cols: {fe_cols}")

    df = df.sort_values(args.send_ts_col, kind="mergesort").reset_index(drop=True)
    t2_s = parse_duration_to_seconds(args.t2)
    t3_s = parse_duration_to_seconds(args.t3)
    user_stratification_payload = None
    if args.enable_user_stratified_eval:
        feature_mappings = None
        mappings_json_path = Path(args.feature_mappings_json)
        if mappings_json_path.exists():
            with open(mappings_json_path, "r", encoding="utf-8") as f:
                feature_mappings = json.load(f).get("mappings", {})
        strat_feature_cols = [args.user_watch_cnt_col, args.user_watch_dur_col]
        for col in strat_feature_cols:
            if col not in df.columns:
                raise ValueError(f"开启用户分层评测时，输入数据缺少列: {col}")
        user_stratification_payload = build_user_stratification_payload(
            df=df,
            user_col=args.user_col,
            feature_cols=strat_feature_cols,
            feature_mappings=feature_mappings,
            # num_bins=args.user_stratify_bins,
        )

    field_dims = calc_field_dims(df, fe_cols)
    model = build_model(args, field_dims).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    window_ids_col = None
    window_metric_ids_col = None
    if args.window_id_col in df.columns:
        window_metric_ids_col = "__window_id_sec_metric__"
        valid_window_ids = (
            args.short_long_windows
            if args.model_name in {"ORM2W_30S_3600S", "ORM2W_30S_3600S_DELAYED3"}
            else args.main_windows
        )
        df[window_metric_ids_col] = normalize_window_ids_to_seconds(
            df[args.window_id_col],
            args.model_name,
            valid_window_ids=valid_window_ids,
        )
        if (df[window_metric_ids_col] == 0).any():
            bad = df[df[window_metric_ids_col] == 0][args.window_id_col].unique()
            raise ValueError(f"未知的窗口取值: {bad}")
    if getattr(model, "requires_window_id", False):
        window_ids_col = window_metric_ids_col
    first_window_metric_id = resolve_first_window_metric_id(model, args.model_name, window_metric_ids_col, df)
    print(f"[OK] first_window_metric_id: {first_window_metric_id}")

    pre_df, online_df, t1_ts = split_pretrain_online_by_t1(df, args.t1, args.send_ts_col)
    include_keys = bool(getattr(model, "requires_keys", False))
    pre_frame = build_tensor_frame(
        pre_df,
        fe_cols,
        args.label_col,
        window_ids_col=window_ids_col,
        include_keys=include_keys,
        user_col=args.user_col,
    )
    online_frame = build_tensor_frame(
        online_df,
        fe_cols,
        args.label_col,
        send_ts_col=args.send_ts_col,
        window_ids_col=window_ids_col,
        metric_window_ids_col=window_metric_ids_col,
        include_keys=include_keys,
        user_col=args.user_col,
        enter_ts_ms_col=args.enter_ts_ms_col,
        start_ts_col=args.start_ts_col,
    )
    pretrain_sample_count = len(pre_df)
    online_sample_count = len(online_df)
    print(f"[OK] pretrain_samples: {pretrain_sample_count}")
    print(f"[OK] online_samples: {online_sample_count}")
    del pre_df
    del online_df
    online_step = 0
    t3_eval_every_steps = max(1, int(args.t3_eval_every_steps))
    online_eval_time_total = 0.0
    online_train_time_total = 0.0
    online_t3_time_total = 0.0
    online_active_windows = 0

    total_params = count_total_params(model)
    shared_params, head_params = count_shared_and_private_params(model)
    profile_source = pre_frame if len(pre_frame) > 0 else online_frame
    profile_batch_size = min(args.batch_size, len(profile_source), 256) if len(profile_source) > 0 else 0
    profile_x = profile_source.x[:profile_batch_size] if profile_batch_size > 0 else torch.empty((0, len(fe_cols)), dtype=torch.long)
    flops_info = estimate_forward_flops(model, profile_x, device)
    print(
        f"[PROFILE] total_params={total_params} shared_params={shared_params} "
        f"head_params={head_params} forward_flops={flops_info['forward_flops']} "
        f"forward_flops_per_sample={flops_info['forward_flops_per_sample']} "
        f"profile_batch_size={flops_info['profile_batch_size']}"
    )

    for ep in range(args.pretrain_epochs):
        t0 = time.time()
        loss = train_one_pass(
            model,
            optimizer,
            pre_frame,
            args.batch_size,
            device,
            True,
            is_pretrain=True,
        )
        print(f"[PRETRAIN] epoch={ep} loss={loss:.6f} time={time.time()-t0:.2f}s")

    bank = MemoryBank()
    cvr_map = None
    cur_t = t1_ts
    end_t = t1_ts + t2_s
    win_start = t1_ts
    win_end = t1_ts + t3_s
    max_ts = int(online_frame.ts[-1]) if online_frame.ts is not None and len(online_frame) > 0 else t1_ts

    while cur_t <= max_ts:
        start_idx, end_idx_ = get_window_bounds(online_frame.ts, cur_t, end_t)
        if end_idx_ > start_idx:
            test_frame = online_frame.slice(start_idx, end_idx_)
            online_step += 1
            online_active_windows += 1
            logits_np = None
            eval_t0 = time.time()
            if hasattr(model, "predict_with_logits"):
                preds, logits_np = predict_with_logits(
                    model,
                    test_frame,
                    args.batch_size,
                    device,
                )
                if test_frame.ts is not None and test_frame.window_ids is not None:
                    append_cvr_records(
                        bank.cvr_store,
                        test_frame.ts,
                        test_frame.y.numpy(),
                        test_frame.window_ids.numpy(),
                        logits_np,
                        cvr_thresholds=cvr_thresholds,
                        threshold_bins_by_window=cvr_threshold_bins,
                    )
            else:
                preds = predict(
                    model,
                    test_frame,
                    args.batch_size,
                    device,
                )

            cvr_map = build_cvr_map(
                logits_np,
                cvr_thresholds=cvr_thresholds,
                threshold_bins_by_window=cvr_threshold_bins,
            ) if logits_np is not None else None
            update_memory_bank(
                bank,
                test_frame,
                preds,
                logits_np,
                cvr_map,
                first_window_id=first_window_metric_id,
                raw_cvr_thresholds_by_window=raw_cvr_thresholds_by_window,
            )
            eval_time = time.time() - eval_t0
            online_eval_time_total += eval_time

            train_t0 = time.time()
            loss = train_one_pass(
                model,
                optimizer,
                test_frame,
                args.batch_size,
                device,
                True,
            )
            train_time = time.time() - train_t0
            online_train_time_total += train_time
            print(
                f"[ONLINE] window=({cur_t},{end_t}) size={len(test_frame)} "
                f"eval_time={eval_time:.2f}s train_time={train_time:.2f}s train_loss={loss:.6f}"
            )

        while end_t >= win_end:
            should_eval_t3 = (online_step % t3_eval_every_steps == 0) and (online_step > 0)
            if should_eval_t3:
                t3_t0 = time.time()
                t3_row = compute_t3_metrics(
                    bank.record_ts,
                    bank.record_label,
                    bank.record_pred,
                    bank.record_user,
                    bank.record_window,
                    bank.cvr_store,
                    win_start,
                    win_end,
                    main_windows=list(main_windows),
                    window_metric_names=window_metric_names,
                    cvr_output_threshold_by_actual=cvr_output_threshold_by_actual,
                )
                t3_eval_time = time.time() - t3_t0
                online_t3_time_total += t3_eval_time
                t3_row["job_name"] = job_name
                t3_row["online_step"] = online_step
                t3_row["eval_time_sec"] = t3_eval_time
                bank.t3_records.append(t3_row)
                print(
                    f"[EVAL-T3] window=({win_start},{win_end}) count={t3_row['count']} "
                    f"regauc={t3_row['regauc']:.6f} uauc={t3_row['uauc']:.6f} "
                    f"rmse={t3_row['rmse']:.6f} cr={t3_row['cr']:.6f} time={t3_eval_time:.2f}s"
                )
            win_start = win_end
            win_end += t3_s
        cur_t = end_t
        end_t += t2_s

    if bank.record_ts:
        basic_metrics_t0 = time.time()
        overall = calc_basic_metrics(
            np.asarray(bank.record_label, dtype=np.float32),
            np.asarray(bank.record_pred, dtype=np.float32),
            np.asarray(bank.record_user, dtype=np.int64),
        )
        basic_metrics_time = time.time() - basic_metrics_t0
        print(f"[EVAL-BASIC] time={basic_metrics_time:.2f}s")
        excel_row = {
            "job_name": job_name,
            "model_name": args.model_name,
            "label": args.label_col,
            "t1": args.t1,
            "t2": args.t2,
            "t3": args.t3,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "regauc": overall["regauc"],
            "uauc": overall["uauc"],
            "rmse": overall["rmse"],
            "cr": overall["cr"],
            "count": overall["count"],
        }

        if args.report_param_size:
            excel_row["embed_dim"] = args.embed_dim
            excel_row["hidden_dim"] = args.hidden_dim

        final_unbias_eval_time = np.nan
        user_stratified_eval_payload = None
        if args.unbias_final_eval:
            final_unbias_eval_t0 = time.time()
            excel_row.update(compute_overall_unbias_metrics(bank.first_pred_map, bank.last_label_map))
            if bank.record_window:
                excel_row["regauc_5min"] = calc_window_regauc(
                    np.asarray(bank.record_label, dtype=np.float32),
                    np.asarray(bank.record_pred, dtype=np.float32),
                    np.asarray(bank.record_window, dtype=np.int64),
                    main_windows[0],
                )
            excel_row.update(
                compute_window_unbias_metrics(
                    bank.record_user,
                    bank.record_live,
                    bank.record_label,
                    bank.record_pred,
                    bank.record_window,
                    bank.first_pred_map,
                    second_window_id=main_windows[1],
                    third_window_id=main_windows[2],
                    second_window_name=window_metric_names[main_windows[1]],
                    third_window_name=window_metric_names[main_windows[2]],
                )
            )
            excel_row.update(
                compute_cvr_unbias_metrics(
                    bank.raw_cvr_map,
                    bank.first_cvr_map,
                    output_threshold_by_actual=cvr_output_threshold_by_actual,
                )
            )
            first_cvr_threshold = main_windows[0]
            excel_row.update(
                compute_cvr_first_window_metrics(
                    bank.first_cvr_map,
                    bank.last_label_map,
                    threshold=first_cvr_threshold,
                    output_threshold=cvr_output_threshold_by_actual.get(first_cvr_threshold, first_cvr_threshold),
                )
            )
            excel_row.update(
                compute_early_live_metrics(
                    bank.first_pred_map,
                    bank.first_cvr_map,
                    bank.last_label_map,
                    bank.first_seen_ts_ms_map,
                    bank.start_ts_map,
                    [1, 3, 5, 10, 30, 45, 60, 90],
                    cvr_threshold=first_cvr_threshold,
                    cvr_output_threshold=cvr_output_threshold_by_actual.get(first_cvr_threshold, first_cvr_threshold),
                )
            )
            if args.enable_user_stratified_eval and user_stratification_payload is not None:
                user_stratified_eval_payload = {
                    "job_name": job_name,
                    "model_name": args.model_name,
                    "stratify_user_col": args.user_col,
                    "num_users_total": user_stratification_payload["num_users_total"],
                    "num_bins_requested": user_stratification_payload["num_bins_requested"],
                    "source": user_stratification_payload["source"],
                    "metrics": {},
                }
                for feature_col in [args.user_watch_cnt_col, args.user_watch_dur_col]:
                    user_stratified_eval_payload["metrics"][feature_col] = {
                        "feature": feature_col,
                        "groups": compute_user_group_stratified_unbias_metrics(
                            bank.first_pred_map,
                            bank.last_label_map,
                            bank.first_seen_ts_ms_map,
                            bank.start_ts_map,
                            user_stratification_payload["feature_group_maps"][feature_col],
                            user_stratification_payload["feature_group_meta"][feature_col],
                        ),
                    }
            final_unbias_eval_time = time.time() - final_unbias_eval_t0
            print(f"[EVAL-FINAL-UNBIAS] time={final_unbias_eval_time:.2f}s")

        excel_row.update(
            {
                "total_param_size": total_params,
                "shared_param_size": shared_params,
                "head_param_size": head_params,
                "forward_flops": flops_info["forward_flops"],
                "forward_flops_per_sample": flops_info["forward_flops_per_sample"],
                "profile_batch_size": flops_info["profile_batch_size"],
                "online_eval_time_sec": online_eval_time_total,
                "online_train_time_sec": online_train_time_total,
                "online_t3_eval_time_sec": online_t3_time_total,
                "online_total_time_sec": online_eval_time_total + online_train_time_total + online_t3_time_total,
                "online_avg_eval_time_sec": (
                    online_eval_time_total / online_active_windows if online_active_windows > 0 else np.nan
                ),
                "online_avg_train_time_sec": (
                    online_train_time_total / online_active_windows if online_active_windows > 0 else np.nan
                ),
                "basic_metrics_time_sec": basic_metrics_time,
                "final_unbias_eval_time_sec": final_unbias_eval_time,
            }
        )

        out_dir = Path(f"results/{args.dat_name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        save_metrics_to_excel(excel_row, out_dir / "results_streaming_v1.xlsx")
        print(f"[OK] saved metrics -> {out_dir / 'results_streaming_v1.xlsx'}")
        if user_stratified_eval_payload is not None:
            strat_dir = out_dir / "user_stratified"
            strat_dir.mkdir(parents=True, exist_ok=True)
            strat_path = strat_dir / f"{job_name}_user_stratified.json"
            with open(strat_path, "w", encoding="utf-8") as f:
                json.dump(user_stratified_eval_payload, f, ensure_ascii=False, indent=2)
            print(f"[OK] saved user stratified metrics -> {strat_path}")
    else:
        print("[EVAL-ALL] no test samples after T1")

    t3_dir = Path(f"results/{args.dat_name}/streaming")
    t3_dir.mkdir(parents=True, exist_ok=True)
    t3_path = t3_dir / f"{job_name}_v1_hourly.json"
    with open(t3_path, "w", encoding="utf-8") as f:
        json.dump(bank.t3_records, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved t3 metrics -> {t3_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dat_name", type=str, default="KuaiLive")
    parser.add_argument("--model_name", type=str, default="ORM3W", choices=["DEFER-5-30", "DEFER-5-90", "ES-DFM-5-30", "ES-DFM-5-90", "ORM3W", 
        "ORM2W_30S_3600S", "ORM3W_DELAYED3", "ORM5_STREAM", "ORM30_STREAM", "ORM90_STREAM"])
    parser.add_argument("--label_col", type=str, default="culm_wt")
    parser.add_argument("--embed_dim", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--backbone", type=str, choices=["MLP", "DCN"], default="MLP")
    parser.add_argument("--send_ts_col", type=str, default="send_timestamp")
    parser.add_argument("--start_ts_col", type=str, default="start_timestamp")
    parser.add_argument("--enter_ts_ms_col", type=str, default="enter_ts_ms")
    parser.add_argument("--user_col", type=str, default="user_id")
    parser.add_argument("--fe_cols", type=str, default=None)
    parser.add_argument("--use_full_features", type=int, default=0)
    parser.add_argument("--use_window_id", type=int, default=0)
    parser.add_argument("--window_id_col", type=str, default="send_window_id")
    parser.add_argument("--unbias_final_eval", type=int, default=1)
    parser.add_argument("--enable_user_stratified_eval", type=int, default=0)
    # parser.add_argument("--user_stratify_bins", type=int, default=5)
    parser.add_argument("--user_watch_cnt_col", type=str, default="user_accu_watch_live_cnt")
    parser.add_argument("--user_watch_dur_col", type=str, default="user_accu_watch_live_duration")
    parser.add_argument(
        "--feature_mappings_json",
        type=str,
        default=None,
    )
    parser.add_argument("--report_param_size", type=int, default=0)
    parser.add_argument("--align_weight", type=float, default=1.0)
    parser.add_argument("--fake_negative_weight", type=float, default=0.1)
    parser.add_argument("--t1", type=str, default="20250515")
    parser.add_argument("--t2", type=str, default="1h")
    parser.add_argument("--t3", type=str, default="4h")
    parser.add_argument("--t3_eval_every_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--pretrain_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--randseed", type=int, default=42)
    parser.add_argument("--use_cuda", type=int, default=1)
    args = parser.parse_args()
    dataset_config = get_dataset_config(args.dat_name)
    if args.fe_cols is None:
        args.fe_cols = dataset_config.default_fe_cols
    if args.feature_mappings_json is None:
        args.feature_mappings_json = dataset_config.feature_mappings_json

    # 针对数据集和模型选择输入文件。
    model_input_file = dataset_config.model_input_files.get(args.model_name)
    if model_input_file is None:
        raise ValueError(f"No input file configured for model_name={args.model_name} on dataset={args.dat_name}")
    args.input_csv = f"./dataset/{args.dat_name}/processed_data_stream/{model_input_file}"
    print(f"[OK] input_csv: {args.input_csv}")
    run_streaming(args)


if __name__ == "__main__":
    main()

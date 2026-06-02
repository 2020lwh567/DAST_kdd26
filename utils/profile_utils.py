from typing import Dict

import numpy as np
import torch


def count_total_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _run_model_for_profile(model: torch.nn.Module, sample_x: torch.Tensor):
    if hasattr(model, "predict"):
        return model.predict(sample_x)
    return model(sample_x)


def estimate_forward_flops(model: torch.nn.Module, sample_x: torch.Tensor, device: torch.device) -> Dict[str, float]:
    if sample_x.numel() == 0:
        return {
            "profile_batch_size": 0,
            "forward_flops": np.nan,
            "forward_flops_per_sample": np.nan,
        }

    sample_x = sample_x.to(device)
    batch_size = int(sample_x.size(0))
    was_training = model.training
    model.eval()

    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        with torch.no_grad():
            try:
                from torch.profiler import ProfilerActivity, profile

                activities = [ProfilerActivity.CPU]
                if device.type == "cuda":
                    activities.append(ProfilerActivity.CUDA)
                with profile(activities=activities, with_flops=True) as prof:
                    _run_model_for_profile(model, sample_x)
                total_flops = float(
                    sum(evt.flops for evt in prof.key_averages() if getattr(evt, "flops", None) is not None)
                )
            except Exception:
                with torch.autograd.profiler.profile(
                    use_cuda=(device.type == "cuda"),
                    with_flops=True,
                ) as prof:
                    _run_model_for_profile(model, sample_x)
                total_flops = float(
                    sum(evt.flops for evt in prof.function_events if getattr(evt, "flops", None) is not None)
                )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        if total_flops <= 0:
            total_flops = np.nan
        return {
            "profile_batch_size": batch_size,
            "forward_flops": total_flops,
            "forward_flops_per_sample": (
                float(total_flops / batch_size) if batch_size > 0 and not np.isnan(total_flops) else np.nan
            ),
        }
    except Exception:
        return {
            "profile_batch_size": batch_size,
            "forward_flops": np.nan,
            "forward_flops_per_sample": np.nan,
        }
    finally:
        if was_training:
            model.train()

import torch
from torch import nn

from model.layer_utils import FeaturesEmbedding, build_bottom_backbone
from model.window_utils import build_three_window_thresholds, build_three_window_weights


class ThreeWindowORMDelayed3Model(nn.Module):
    """
    双 embedding 版本：
    - 5min 使用 fast embedding
    - 30/90min 使用 slow embedding
    - 辅助对齐损失：lambda * || proj(h2) - stop_grad(h1) ||^2
      h1: 第一次 5min 窗口的 fast embed_x
      h2: 后续窗口的 slow embed_x
    """

    requires_window_id = True
    requires_keys = True
    name = "ORM3W_DELAYED3"

    def __init__(
        self,
        field_dims,
        embed_dim=16,
        bottom_dim=64,
        align_weight: float = 1e-2,
        backbone: str = "MLP",
        main_windows=(300, 1800, 5400),
    ):
        super().__init__()
        self.main_windows = tuple(int(w) for w in main_windows)
        self.first_window_id, self.second_window_id, self.third_window_id = self.main_windows
        self.window_keys = [str(w) for w in self.main_windows]
        self.fast_embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.slow_embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.align_weight = align_weight
        self.backbone = backbone

        self.bottom_fast = build_bottom_backbone(self.embed_output_dim, bottom_dim, backbone=backbone)
        self.bottom_slow = build_bottom_backbone(self.embed_output_dim, bottom_dim, backbone=backbone)

        thresholds_by_window = build_three_window_thresholds(self.main_windows)
        weights_by_window = build_three_window_weights(self.main_windows)
        self.orm_heads = nn.ModuleDict(
            {
                self.window_keys[0]: nn.Linear(bottom_dim, len(thresholds_by_window[self.first_window_id])),
                self.window_keys[1]: nn.Linear(bottom_dim, len(thresholds_by_window[self.second_window_id])),
                self.window_keys[2]: nn.Linear(bottom_dim, len(thresholds_by_window[self.third_window_id])),
            }
        )

        self.threshold_buffer_names = {}
        self.weight_buffer_names = {}
        for win_id in self.main_windows:
            threshold_name = f"orm_thresholds_{win_id}"
            weight_name = f"orm_weights_{win_id}"
            self.register_buffer(threshold_name, torch.tensor(thresholds_by_window[win_id], dtype=torch.float32))
            self.register_buffer(weight_name, torch.tensor(weights_by_window[win_id], dtype=torch.float32))
            self.threshold_buffer_names[win_id] = threshold_name
            self.weight_buffer_names[win_id] = weight_name

        self.idx_first = len(thresholds_by_window[self.first_window_id]) - 1
        self.idx_second = len(thresholds_by_window[self.second_window_id]) - 1
        self.orm_offset = 0.5

        # self.align_proj = nn.Linear(self.embed_output_dim, self.embed_output_dim)
        self.register_buffer("fast_cache_keys", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("fast_cache_values", torch.empty(0, bottom_dim, dtype=torch.float32), persistent=False)

    def _encode_branch(self, embedding, bottom, x):
        h = embedding(x).view(-1, self.embed_output_dim)
        return bottom(h)

    def _encode_window_states(self, x):
        feat_first = self._encode_branch(self.fast_embedding, self.bottom_fast, x)
        feat_second = self._encode_branch(self.slow_embedding, self.bottom_slow, x)
        feat_third = feat_second
        return feat_first, feat_second, feat_third

    def _forward_logits(self, feat_first, feat_second, feat_third):
        logit_first = self.orm_heads[self.window_keys[0]](feat_first)
        logit_second = self.orm_heads[self.window_keys[1]](feat_second)
        logit_third = self.orm_heads[self.window_keys[2]](feat_third)
        return logit_first, logit_second, logit_third

    def _get_thresholds(self, win_id: int):
        return getattr(self, self.threshold_buffer_names[int(win_id)])

    def _get_weights(self, win_id: int):
        return getattr(self, self.weight_buffer_names[int(win_id)])

    def _combine_pred(self, logit_first, logit_second, logit_third):
        sig_first = torch.sigmoid(logit_first)
        sig_second = torch.sigmoid(logit_second)
        sig_third = torch.sigmoid(logit_third)

        w_first = self._get_weights(self.first_window_id).view(1, -1)
        w_second = self._get_weights(self.second_window_id).view(1, -1)
        w_third = self._get_weights(self.third_window_id).view(1, -1)
        pred_st_first = torch.sum(sig_first * w_first, dim=1)
        pred_st_second = torch.sum(sig_second * w_second, dim=1)
        pred_st_third = torch.sum(sig_third * w_third, dim=1)
        # pred = pred_st_300 + pred_cvr_300s * pred_st_1800 + pred_cvr_300s * pred_cvr_1800s * pred_st_5400 + self.orm_offset
        pred = pred_st_first + pred_st_second + pred_st_third + self.orm_offset
        return pred

    def _cache_fast(self, keys, fast_x):
        if keys.numel() == 0:
            return

        sorted_keys, sort_idx = torch.sort(keys.long())
        sorted_fast_x = fast_x.detach()[sort_idx]
        keep = torch.ones_like(sorted_keys, dtype=torch.bool)
        keep[1:] = sorted_keys[1:] != sorted_keys[:-1]
        new_keys = sorted_keys[keep]
        new_values = sorted_fast_x[keep]

        if self.fast_cache_keys.numel() > 0:
            insert_pos = torch.searchsorted(self.fast_cache_keys, new_keys)
            probe_pos = torch.clamp(insert_pos, max=self.fast_cache_keys.numel() - 1)
            hit = (insert_pos < self.fast_cache_keys.numel()) & (self.fast_cache_keys[probe_pos] == new_keys)
            new_keys = new_keys[~hit]
            new_values = new_values[~hit]

        if new_keys.numel() == 0:
            return

        merged_keys = torch.cat([self.fast_cache_keys, new_keys], dim=0)
        merged_values = torch.cat([self.fast_cache_values, new_values], dim=0)
        order = torch.argsort(merged_keys)
        self.fast_cache_keys = merged_keys[order]
        self.fast_cache_values = merged_values[order]

    def _compute_align_loss_for_branch(self, keys, branch_x):
        if self.fast_cache_keys.numel() == 0 or keys.numel() == 0:
            return torch.tensor(0.0, device=branch_x.device), 0

        lookup_keys = keys.long()
        insert_pos = torch.searchsorted(self.fast_cache_keys, lookup_keys)
        probe_pos = torch.clamp(insert_pos, max=self.fast_cache_keys.numel() - 1)
        hit = (insert_pos < self.fast_cache_keys.numel()) & (self.fast_cache_keys[probe_pos] == lookup_keys)
        if not torch.any(hit):
            return torch.tensor(0.0, device=branch_x.device), 0

        cached_fast_x = self.fast_cache_values[insert_pos[hit]]
        branch_hit = branch_x[hit]
        diff = branch_hit - cached_fast_x
        align_loss = (diff * diff).sum()
        return align_loss, int(hit.sum().item())

    def _compute_align_loss(self, keys, window_ids, feat_second, feat_third):
        total_align_loss = torch.tensor(0.0, device=feat_second.device)
        total_align_count = 0
        idx_second = window_ids == self.second_window_id
        if torch.any(idx_second):
            loss_second, count_second = self._compute_align_loss_for_branch(keys[idx_second], feat_second[idx_second])
            total_align_loss = total_align_loss + loss_second
            total_align_count += count_second
        idx_third = window_ids == self.third_window_id
        if torch.any(idx_third):
            loss_third, count_third = self._compute_align_loss_for_branch(keys[idx_third], feat_third[idx_third])
            total_align_loss = total_align_loss + loss_third
            total_align_count += count_third
        return total_align_loss, total_align_count

    def forward(self, x, window_ids=None, return_logits=False):
        feat_first, feat_second, feat_third = self._encode_window_states(x)
        logit_first, logit_second, logit_third = self._forward_logits(feat_first, feat_second, feat_third)
        if return_logits:
            return logit_first, logit_second, logit_third
        else:
            return self._combine_pred(logit_first, logit_second, logit_third)

    def predict(self, x):
        return self.forward(x)

    def predict_with_logits(self, x):
        feat_first, feat_second, feat_third = self._encode_window_states(x)
        logit_first, logit_second, logit_third = self._forward_logits(feat_first, feat_second, feat_third)
        pred = self._combine_pred(logit_first, logit_second, logit_third)
        return pred, {
            str(self.first_window_id): logit_first,
            str(self.second_window_id): logit_second,
            str(self.third_window_id): logit_third,
        }

    def compute_loss(self, x, y, window_ids, keys):
        feat_first, feat_second, feat_third = self._encode_window_states(x)
        debug = {
            f"loss_{self.first_window_id}_sum": 0.0,
            f"loss_{self.second_window_id}_sum": 0.0,
            f"loss_{self.third_window_id}_sum": 0.0,
            f"count_{self.first_window_id}": 0,
            f"count_{self.second_window_id}": 0,
            f"count_{self.third_window_id}": 0,
        }

        if torch.any(window_ids == self.first_window_id):
            idx_first = window_ids == self.first_window_id
            self._cache_fast(keys[idx_first], feat_first[idx_first])

        logit_first, logit_second, logit_third = self._forward_logits(feat_first, feat_second, feat_third)
        pred = self._combine_pred(logit_first, logit_second, logit_third)

        bce = nn.BCEWithLogitsLoss(reduction="none")
        loss_total = torch.tensor(0.0, device=x.device)
        count_total = 0
        for win_id, logits in zip(self.main_windows, [logit_first, logit_second, logit_third]):
            idx = window_ids == win_id
            if not torch.any(idx):
                continue
            logit_w = logits[idx]
            y_w = y[idx].view(-1, 1)
            thresholds = self._get_thresholds(win_id).view(1, -1)
            weights = self._get_weights(win_id).view(1, -1)
            label = (y_w >= thresholds).float()
            loss = bce(logit_w, label) * weights
            loss = loss.sum(dim=1)
            loss_total = loss_total + loss.sum()
            count_total += loss.numel()

            debug[f"loss_{win_id}_sum"] += float(loss.sum().detach().item())
            debug[f"count_{win_id}"] += int(loss.numel())

        base_loss = loss_total / count_total if count_total > 0 else loss_total

        # align loss
        align_loss, align_count = self._compute_align_loss(keys, window_ids, feat_second, feat_third)

        loss = base_loss + self.align_weight * align_loss
        debug.update({
            "align_loss": float(align_loss.detach().item()) if isinstance(align_loss, torch.Tensor) else float(align_loss),
            "align_count": int(align_count),
            "base_loss": float(base_loss.detach().item()) if isinstance(base_loss, torch.Tensor) else float(base_loss),
            "count_total": int(count_total),
        })
        return loss, pred, debug

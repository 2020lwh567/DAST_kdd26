import torch
from torch import nn

from model.layer_utils import FeaturesEmbedding, build_bottom_backbone
from model.window_utils import build_two_window_thresholds, build_two_window_weights


class TwoWindowORM30s3600sModel(nn.Module):
    """
    两窗口 ORM 模型：
    - 30s 窗口：只建模 1~30s 的 30 个 1s 桶
    - 3600s 窗口：建模 31~300s 的 1s 桶、305~1800s 的 5s 桶、1830~3600s 的 30s 桶
    - 30s 样本仅训练 1~30s 桶
    - 3600s 样本仅训练 31~3600s 桶
    """

    requires_window_id = True
    first_window_id = 30

    def __init__(
        self,
        field_dims,
        embed_dim=8,
        bottom_dim=48,
        backbone: str = "MLP",
        short_long_windows=(30, 3600),
    ):
        super().__init__()
        self.short_long_windows = tuple(int(w) for w in short_long_windows)
        self.short_window_id, self.long_window_id = self.short_long_windows
        self.first_window_id = self.short_window_id
        self.window_keys = [str(w) for w in self.short_long_windows]
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim

        self.bottom = build_bottom_backbone(self.embed_output_dim, bottom_dim, backbone=backbone)

        self.backbone = backbone

        thresholds_by_window = build_two_window_thresholds(self.short_long_windows)
        weights_by_window = build_two_window_weights(self.short_long_windows)

        self.orm_heads = nn.ModuleDict(
            {
                self.window_keys[0]: nn.Linear(bottom_dim, len(thresholds_by_window[self.short_window_id])),
                self.window_keys[1]: nn.Linear(bottom_dim, len(thresholds_by_window[self.long_window_id])),
            }
        )

        self.threshold_buffer_names = {}
        self.weight_buffer_names = {}
        for win_id in self.short_long_windows:
            threshold_name = f"orm_thresholds_{win_id}"
            weight_name = f"orm_weights_{win_id}"
            self.register_buffer(threshold_name, torch.tensor(thresholds_by_window[win_id], dtype=torch.float32))
            self.register_buffer(weight_name, torch.tensor(weights_by_window[win_id], dtype=torch.float32))
            self.threshold_buffer_names[win_id] = threshold_name
            self.weight_buffer_names[win_id] = weight_name
        self.orm_offset = 0.5

    def _get_thresholds(self, win_id: int):
        return getattr(self, self.threshold_buffer_names[int(win_id)])

    def _get_weights(self, win_id: int):
        return getattr(self, self.weight_buffer_names[int(win_id)])

    def _forward_logits(self, x):
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        h = self.bottom(embed_x)

        logit_short = self.orm_heads[self.window_keys[0]](h)
        logit_long = self.orm_heads[self.window_keys[1]](h)
        return logit_short, logit_long

    def _combine_pred(self, logit_short, logit_long):
        sig_short = torch.sigmoid(logit_short)
        sig_long = torch.sigmoid(logit_long)
        pred_st_short = torch.sum(sig_short * self._get_weights(self.short_window_id).view(1, -1), dim=1)
        pred_st_long = torch.sum(sig_long * self._get_weights(self.long_window_id).view(1, -1), dim=1)
        return pred_st_short + pred_st_long + self.orm_offset

    def forward(self, x):
        logit_short, logit_long = self._forward_logits(x)
        return self._combine_pred(logit_short, logit_long)

    def predict(self, x):
        return self.forward(x)

    def predict_with_logits(self, x):
        logit_short, logit_long = self._forward_logits(x)
        pred = self._combine_pred(logit_short, logit_long)
        return pred, {str(self.short_window_id): logit_short, str(self.long_window_id): logit_long}

    def compute_loss(self, x, y, window_ids):
        logit_short, logit_long = self._forward_logits(x)
        pred = self._combine_pred(logit_short, logit_long)

        device = x.device
        loss_total = torch.tensor(0.0, device=device)
        count_total = 0
        bce = nn.BCEWithLogitsLoss(reduction="none")

        idx_short = window_ids == self.short_window_id
        if torch.any(idx_short):
            y_short = y[idx_short].view(-1, 1)
            thresholds_short = self._get_thresholds(self.short_window_id).view(1, -1).to(device)
            weights_short = self._get_weights(self.short_window_id).view(1, -1).to(device)
            label_short = (y_short >= thresholds_short).float()
            loss_short = bce(logit_short[idx_short], label_short) * weights_short
            loss_short = loss_short.sum(dim=1)
            loss_total = loss_total + loss_short.sum()
            count_total += loss_short.numel()

        idx_long = window_ids == self.long_window_id
        if torch.any(idx_long):
            y_long = y[idx_long].view(-1, 1)
            thresholds_long = self._get_thresholds(self.long_window_id).view(1, -1).to(device)
            weights_long = self._get_weights(self.long_window_id).view(1, -1).to(device)
            label_long = (y_long >= thresholds_long).float()
            loss_long = bce(logit_long[idx_long], label_long) * weights_long
            loss_long = loss_long.sum(dim=1)
            loss_total = loss_total + loss_long.sum()
            count_total += loss_long.numel()

        if count_total == 0:
            return loss_total, pred
        return loss_total / count_total, pred

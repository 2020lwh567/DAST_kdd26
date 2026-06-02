import torch
from torch import nn
from model.layer_utils import FeaturesEmbedding, build_bottom_backbone
from model.window_utils import build_three_window_thresholds, build_three_window_weights


class ThreeWindowORMModel(nn.Module):
    """
    三窗口 ORM 基线模型（5min/30min/90min）。
    - 底层 embedding -> H
    - 每个窗口有专属 tower 生成输入
    - 每个窗口对应一个 ORM head（300/300/120 桶）
    """

    requires_window_id = True

    def __init__(
        self,
        field_dims,
        embed_dim=8,
        bottom_dim=48,
        backbone: str = "MLP",
        main_windows=(300, 1800, 5400),
    ):
        super().__init__()
        self.main_windows = tuple(int(w) for w in main_windows)
        self.first_window_id, self.second_window_id, self.third_window_id = self.main_windows
        self.window_keys = [str(w) for w in self.main_windows]
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim

        self.bottom = build_bottom_backbone(self.embed_output_dim, bottom_dim, backbone=backbone)

        self.backbone = backbone

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
        self.orm_offset = 0.5
        self.idx_first = len(thresholds_by_window[self.first_window_id]) - 1
        self.idx_second = len(thresholds_by_window[self.second_window_id]) - 1

    def _forward_logits(self, x):
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        h = self.bottom(embed_x)
        # h = embed_x  # Todo2: 去掉bottom

        logit_first = self.orm_heads[self.window_keys[0]](h)
        logit_second = self.orm_heads[self.window_keys[1]](h)
        logit_third = self.orm_heads[self.window_keys[2]](h)
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

        # pred = (
        #     pred_st_300
        #     + pred_cvr_300s * pred_st_1800
        #     + pred_cvr_300s * pred_cvr_1800s * pred_st_5400
        #     + self.orm_offset
        # )
        # todo full window: 改成全窗口发送
        pred = pred_st_first + pred_st_second + pred_st_third + self.orm_offset
        return pred

    def forward(self, x):
        logit_first, logit_second, logit_third = self._forward_logits(x)
        return self._combine_pred(logit_first, logit_second, logit_third)

    def predict(self, x):
        return self.forward(x)

    def predict_with_logits(self, x):
        logit_first, logit_second, logit_third = self._forward_logits(x)
        pred = self._combine_pred(logit_first, logit_second, logit_third)
        return pred, {
            str(self.first_window_id): logit_first,
            str(self.second_window_id): logit_second,
            str(self.third_window_id): logit_third,
        }

    def compute_loss(self, x, y, window_ids):
        logit_first, logit_second, logit_third = self._forward_logits(x)
        pred = self._combine_pred(logit_first, logit_second, logit_third)

        device = x.device
        loss_total = torch.tensor(0.0, device=device)
        count_total = 0
        bce = nn.BCEWithLogitsLoss(reduction="none")

        for win_id, logits in zip(self.main_windows, [logit_first, logit_second, logit_third]):
            idx = window_ids == win_id
            if not torch.any(idx):
                continue
            logit_w = logits[idx]
            y_w = y[idx].view(-1, 1)

            thresholds = self._get_thresholds(win_id).view(1, -1).to(device)
            weights = self._get_weights(win_id).view(1, -1).to(device)

            label = (y_w >= thresholds).float()
            loss = bce(logit_w, label)
            loss = loss * weights
            loss = loss.sum(dim=1)
            loss_total = loss_total + loss.sum()
            count_total += loss.numel()

        if count_total == 0:
            return loss_total, pred
        return loss_total / count_total, pred

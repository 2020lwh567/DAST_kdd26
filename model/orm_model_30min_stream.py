import torch
from torch import nn

from model.orm_3window import ThreeWindowORMModel


class ORMModelFor30minStream(ThreeWindowORMModel):
    """
    专门用于 30min 数据流的 ORM 模型。

    - 保留 5min 和 30min 两个 tower
    - `self.orm_heads = nn.ModuleDict({"300": ..., "1800": ...})`
    - 每个训练样本的 `send_window_id` 视为 30min（映射后为 1800）
    - 每次训练时同时更新 300 和 1800 两个 orm tower
    - 最大可建模时长为 1800s
    """

    def __init__(
        self,
        field_dims,
        embed_dim=8,
        bottom_dim=48,
        backbone: str = "MLP",
        main_windows=(300, 1800, 5400),
    ):
        super().__init__(
            field_dims=field_dims,
            embed_dim=embed_dim,
            bottom_dim=bottom_dim,
            backbone=backbone,
            main_windows=main_windows,
        )
        self.metric_first_window_id = self.second_window_id
        self.orm_heads = nn.ModuleDict(
            {
                self.window_keys[0]: self.orm_heads[self.window_keys[0]],
                self.window_keys[1]: self.orm_heads[self.window_keys[1]],
            }
        )
        self.win_proj_third = None

    def _forward_logits(self, x):
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        h = self.bottom(embed_x)

        logit_first = self.orm_heads[self.window_keys[0]](h)
        logit_second = self.orm_heads[self.window_keys[1]](h)
        return logit_first, logit_second

    def _combine_pred(self, logit_first, logit_second):
        sig_first = torch.sigmoid(logit_first)
        sig_second = torch.sigmoid(logit_second)
        w_first = self._get_weights(self.main_windows[0]).view(1, -1)
        w_second = self._get_weights(self.main_windows[1]).view(1, -1)
        pred_st_first = torch.sum(sig_first * w_first, dim=1)
        pred_st_second = torch.sum(sig_second * w_second, dim=1)
        return pred_st_first + pred_st_second + self.orm_offset

    def forward(self, x):
        logit_first, logit_second = self._forward_logits(x)
        return self._combine_pred(logit_first, logit_second)

    def predict(self, x):
        return self.forward(x)

    def predict_with_logits(self, x):
        logit_first, logit_second = self._forward_logits(x)
        pred = self._combine_pred(logit_first, logit_second)
        return pred, {str(self.main_windows[0]): logit_first, str(self.main_windows[1]): logit_second}

    def compute_loss(self, x, y, window_ids):
        logit_first, logit_second = self._forward_logits(x)
        pred = self._combine_pred(logit_first, logit_second)
        device = x.device
        y_w = y.view(-1, 1)
        bce = nn.BCEWithLogitsLoss(reduction="none")

        thresholds_first = self._get_thresholds(self.main_windows[0]).view(1, -1).to(device)
        weights_first = self._get_weights(self.main_windows[0]).view(1, -1).to(device)
        label_first = (y_w >= thresholds_first).float()
        loss_first = bce(logit_first, label_first) * weights_first
        loss_first = loss_first.sum(dim=1)

        thresholds_second = self._get_thresholds(self.main_windows[1]).view(1, -1).to(device)
        weights_second = self._get_weights(self.main_windows[1]).view(1, -1).to(device)
        label_second = (y_w >= thresholds_second).float()
        loss_second = bce(logit_second, label_second) * weights_second
        loss_second = loss_second.sum(dim=1)

        loss = loss_first + loss_second
        return loss.mean(), pred

import torch
from torch import nn

from model.orm_3window import ThreeWindowORMModel


class ORMModelFor5minStream(ThreeWindowORMModel):
    """
    仅保留 5min tower 的 ORM 模型。

    - 只建模 0~300s
    - 对于 `culm_wt > 300s` 的样本，loss 仍只在 300s 以下阈值上建模
    - `forward()` / `predict()` / `predict_with_logits()` 只返回 5min tower 的结果
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
        # keep only the 5min head to make the model explicitly single-window
        self.orm_heads = nn.ModuleDict({self.window_keys[0]: self.orm_heads[self.window_keys[0]]})
        self.win_proj_second = None
        self.win_proj_third = None

    def _forward_logits(self, x):
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        h = self.bottom(embed_x)
        logit_first = self.orm_heads[self.window_keys[0]](h)
        return logit_first

    def _combine_pred(self, logit_first):
        sig_first = torch.sigmoid(logit_first)
        w_first = self._get_weights(self.first_window_id).view(1, -1)
        pred_st_first = torch.sum(sig_first * w_first, dim=1)
        return pred_st_first + self.orm_offset

    def forward(self, x):
        logit_first = self._forward_logits(x)
        return self._combine_pred(logit_first)

    def predict(self, x):
        return self.forward(x)

    def predict_with_logits(self, x):
        logit_first = self._forward_logits(x)
        pred = self._combine_pred(logit_first)
        return pred, {str(self.first_window_id): logit_first}

    def compute_loss(self, x, y, window_ids):
        logit_first = self._forward_logits(x)
        pred = self._combine_pred(logit_first)
        device = x.device
        logit_w = logit_first
        y_w = y.view(-1, 1)
        thresholds = self._get_thresholds(self.first_window_id).view(1, -1).to(device)
        weights = self._get_weights(self.first_window_id).view(1, -1).to(device)
        label = (y_w >= thresholds).float()
        bce = nn.BCEWithLogitsLoss(reduction="none")
        loss = bce(logit_w, label) * weights
        loss = loss.sum(dim=1)
        return loss.mean(), pred

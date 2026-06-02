import torch
from torch import nn

from model.orm_3window import ThreeWindowORMModel


class ORMModelFor90minStream(ThreeWindowORMModel):
    """
    专门用于 90min 数据流的 ORM 模型。

    - 保留完整的 5/30/90 三个 tower
    - 每个训练样本在一次 `compute_loss()` 中同时更新三个 orm tower
    - 大部分函数直接继承 `ThreeWindowORMModel`
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
        self.metric_first_window_id = self.third_window_id

    def compute_loss(self, x, y, window_ids):
        logit_first, logit_second, logit_third = self._forward_logits(x)
        pred = self._combine_pred(logit_first, logit_second, logit_third)

        device = x.device
        loss_total = torch.tensor(0.0, device=device)
        count_total = 0
        bce = nn.BCEWithLogitsLoss(reduction="none")

        for win_id, logits in zip(self.main_windows, [logit_first, logit_second, logit_third]):
            thresholds = self._get_thresholds(win_id).view(1, -1).to(device)
            weights = self._get_weights(win_id).view(1, -1).to(device)
            y_w = y.view(-1, 1)
            label = (y_w >= thresholds).float()
            loss = bce(logits, label) * weights
            loss = loss.sum(dim=1)
            loss_total = loss_total + loss.sum()
            count_total += loss.numel()

        if count_total == 0:
            return loss_total, pred
        return loss_total / count_total, pred

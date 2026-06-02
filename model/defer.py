import torch
from torch import nn

from model.layer_utils import FeaturesEmbedding, build_bottom_backbone
from model.window_utils import segment_bounds_from_windows


class DEFERBaseModel(torch.nn.Module):
    """
    DEFER 分段 tower 基类。

    tower 语义：
    - 300 tower: 预测 0~300s 段的归一化观看比例
    - 1800 tower: 预测 300~1800s 段的归一化观看比例
    - 5400 tower: 预测 1800~5400s 段的归一化观看比例

    在 observation window 到达时，未来 attribution tower 使用 fake negative
    目标 0，并乘以较小权重；在 attribution window 到达时，未来 tower
    使用真实分段标签正常更新。
    """

    requires_window_id = True
    first_window_id = 300

    def __init__(
        self,
        field_dims,
        embed_dim=8,
        bottom_dim=48,
        tower_ids=None,
        attribution_window_id=1800,
        fake_negative_weight=0.1,
        backbone: str = "MLP",
        main_windows=(300, 1800, 5400),
    ):
        super().__init__()
        self.main_windows = tuple(int(w) for w in main_windows)
        tower_ids = tower_ids or [300, 1800]
        self.tower_ids = tuple(int(win_id) for win_id in tower_ids)
        self.attribution_window_id = int(attribution_window_id)
        self.first_window_id = self.main_windows[0]
        self.fake_negative_weight = float(fake_negative_weight)
        self.backbone = backbone

        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.bottom = build_bottom_backbone(
            self.embed_output_dim,
            bottom_dim,
            backbone=backbone,
        )
        tower_input_dim = bottom_dim
        self.towers = nn.ModuleDict({str(win_id): nn.Linear(tower_input_dim, 1) for win_id in self.tower_ids})

        self.segment_bounds = segment_bounds_from_windows(self.main_windows, self.tower_ids)
        self.segment_widths = {
            win_id: self.segment_bounds[win_id][1] - self.segment_bounds[win_id][0] for win_id in self.tower_ids
        }

    def _shared_representation(self, x):
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        return self.bottom(embed_x)

    def _forward_logits(self, x):
        h = self._shared_representation(x)
        return {win_id: self.towers[str(win_id)](h).squeeze(1) for win_id in self.tower_ids}

    def _segment_target(self, y, win_id):
        left, right = self.segment_bounds[win_id]
        width = right - left
        target = torch.clamp(y - left, min=0.0, max=width) / width
        return target

    def _combine_pred(self, logits):
        pred = torch.zeros_like(next(iter(logits.values())))
        for win_id in self.tower_ids:
            pred = pred + torch.sigmoid(logits[win_id]) * self.segment_widths[win_id]
        return pred

    def forward(self, x):
        logits = self._forward_logits(x)
        return self._combine_pred(logits)

    def predict(self, x):
        return self.forward(x)

    def _loss_for_tower(self, bce, logits, y, win_id, sample_weight):
        target = self._segment_target(y, win_id)
        loss = bce(logits[win_id], target)
        return loss * sample_weight


class DEFER5to30Model(DEFERBaseModel):
    """
    DEFER-5-30:
    - 5min 为 observation window
    - 30min 为 attribution window
    - 模型包含 5min / 30min 两个 tower
    """

    name = "DEFER-5-30"

    def __init__(
        self,
        field_dims,
        embed_dim=8,
        bottom_dim=48,
        fake_negative_weight=0.1,
        backbone: str = "MLP",
        main_windows=(300, 1800, 5400),
    ):
        main_windows = tuple(int(w) for w in main_windows)
        super().__init__(
            field_dims=field_dims,
            embed_dim=embed_dim,
            bottom_dim=bottom_dim,
            tower_ids=[main_windows[0], main_windows[1]],
            attribution_window_id=main_windows[1],
            fake_negative_weight=fake_negative_weight,
            backbone=backbone,
            main_windows=main_windows,
        )

    def compute_loss(self, x, y, window_ids):
        logits = self._forward_logits(x)
        pred = self._combine_pred(logits)
        bce = nn.BCEWithLogitsLoss(reduction="none")
        loss = torch.zeros_like(y)

        first_id, second_id = self.tower_ids
        idx_first = window_ids == first_id
        if torch.any(idx_first):
            loss[idx_first] = loss[idx_first] + self._loss_for_tower(bce, logits, y, first_id, 1.0)[idx_first]
            loss[idx_first] = loss[idx_first] + self._loss_for_tower(
                bce, logits, y.new_zeros(y.size()), second_id, self.fake_negative_weight
            )[idx_first]

        idx_second = window_ids == second_id
        if torch.any(idx_second):
            loss[idx_second] = loss[idx_second] + self._loss_for_tower(bce, logits, y, second_id, 1.0)[idx_second]

        return loss.mean(), pred


class DEFER5to90Model(DEFERBaseModel):
    """
    DEFER-5-90:
    - 5min 为 observation window
    - 90min 为 attribution window
    - 模型包含 5min / 30min / 90min 三个 tower
    """

    name = "DEFER-5-90"

    def __init__(
        self,
        field_dims,
        embed_dim=8,
        bottom_dim=48,
        fake_negative_weight=0.1,
        backbone: str = "MLP",
        main_windows=(300, 1800, 5400),
    ):
        main_windows = tuple(int(w) for w in main_windows)
        super().__init__(
            field_dims=field_dims,
            embed_dim=embed_dim,
            bottom_dim=bottom_dim,
            tower_ids=[main_windows[0], main_windows[1], main_windows[2]],
            attribution_window_id=main_windows[2],
            fake_negative_weight=fake_negative_weight,
            backbone=backbone,
            main_windows=main_windows,
        )

    def compute_loss(self, x, y, window_ids):
        logits = self._forward_logits(x)
        pred = self._combine_pred(logits)
        bce = nn.BCEWithLogitsLoss(reduction="none")
        loss = torch.zeros_like(y)

        first_id, second_id, third_id = self.tower_ids
        idx_first = window_ids == first_id
        if torch.any(idx_first):
            loss[idx_first] = loss[idx_first] + self._loss_for_tower(bce, logits, y, first_id, 1.0)[idx_first]
            fake_y = y.new_zeros(y.size())
            loss[idx_first] = loss[idx_first] + self._loss_for_tower(
                bce, logits, fake_y, second_id, self.fake_negative_weight
            )[idx_first]
            loss[idx_first] = loss[idx_first] + self._loss_for_tower(
                bce, logits, fake_y, third_id, self.fake_negative_weight
            )[idx_first]

        idx_third = window_ids == third_id
        if torch.any(idx_third):
            loss[idx_third] = loss[idx_third] + self._loss_for_tower(bce, logits, y, second_id, 1.0)[idx_third]
            loss[idx_third] = loss[idx_third] + self._loss_for_tower(bce, logits, y, third_id, 1.0)[idx_third]

        return loss.mean(), pred

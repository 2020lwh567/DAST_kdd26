from typing import List

import torch

from model.defer import DEFER5to30Model, DEFER5to90Model
from model.es_dfm import ESDFM5to30Model, ESDFM5to90Model
from model.orm_3window import ThreeWindowORMModel
from model.orm_2window_30s_3600s import TwoWindowORM30s3600sModel
from model.orm_3window_delayed3 import ThreeWindowORMDelayed3Model
from model.orm_model_5min_stream import ORMModelFor5minStream
from model.orm_model_30min_stream import ORMModelFor30minStream
from model.orm_model_90min_stream import ORMModelFor90minStream


def build_model(args, field_dims: List[int]) -> torch.nn.Module:
    main_windows = tuple(getattr(args, "main_windows", (300, 1800, 5400)))
    short_long_windows = tuple(getattr(args, "short_long_windows", (30, 3600)))
    if args.model_name == "DEFER-5-30":
        return DEFER5to30Model(
            field_dims=field_dims,
            embed_dim=args.embed_dim,
            bottom_dim=args.hidden_dim,
            fake_negative_weight=args.fake_negative_weight,
            backbone=args.backbone,
            main_windows=main_windows,
        )
    if args.model_name == "DEFER-5-90":
        return DEFER5to90Model(
            field_dims=field_dims,
            embed_dim=args.embed_dim,
            bottom_dim=args.hidden_dim,
            fake_negative_weight=args.fake_negative_weight,
            backbone=args.backbone,
            main_windows=main_windows,
        )
    if args.model_name == "ES-DFM-5-30":
        return ESDFM5to30Model(
            field_dims=field_dims,
            embed_dim=args.embed_dim,
            bottom_dim=args.hidden_dim,
            fake_negative_weight=args.fake_negative_weight,
            backbone=args.backbone,
            main_windows=main_windows,
        )
    if args.model_name == "ES-DFM-5-90":
        return ESDFM5to90Model(
            field_dims=field_dims,
            embed_dim=args.embed_dim,
            bottom_dim=args.hidden_dim,
            fake_negative_weight=args.fake_negative_weight,
            backbone=args.backbone,
            main_windows=main_windows,
        )
    if args.model_name == "ORM3W":
        return ThreeWindowORMModel(
            field_dims=field_dims,
            embed_dim=args.embed_dim,
            bottom_dim=args.hidden_dim,
            backbone=args.backbone,
            main_windows=main_windows,
        )
    if args.model_name == "ORM2W_30S_3600S":
        return TwoWindowORM30s3600sModel(
            field_dims=field_dims,
            embed_dim=args.embed_dim,
            bottom_dim=args.hidden_dim,
            backbone=args.backbone,
            short_long_windows=short_long_windows,
        )
    if args.model_name == "ORM3W_DELAYED3":
        return ThreeWindowORMDelayed3Model(
            field_dims=field_dims,
            embed_dim=args.embed_dim,
            bottom_dim=args.hidden_dim,
            align_weight=args.align_weight,
            backbone=args.backbone,
            main_windows=main_windows,
        )
    if args.model_name == "ORM5_STREAM":
        return ORMModelFor5minStream(
            field_dims=field_dims,
            embed_dim=args.embed_dim,
            bottom_dim=args.hidden_dim,
            backbone=args.backbone,
            main_windows=main_windows,
        )
    if args.model_name == "ORM30_STREAM":
        return ORMModelFor30minStream(
            field_dims=field_dims,
            embed_dim=args.embed_dim,
            bottom_dim=args.hidden_dim,
            backbone=args.backbone,
            main_windows=main_windows,
        )
    if args.model_name == "ORM90_STREAM":
        return ORMModelFor90minStream(
            field_dims=field_dims,
            embed_dim=args.embed_dim,
            bottom_dim=args.hidden_dim,
            backbone=args.backbone,
            main_windows=main_windows,
        )
    raise ValueError(f"Unsupported model_name: {args.model_name}")

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class WindowSpec:
    main_windows: Tuple[int, int, int]
    short_long_windows: Tuple[int, int]
    cvr_threshold_map: Dict[int, int]

    @property
    def first_window(self) -> int:
        return self.main_windows[0]

    @property
    def second_window(self) -> int:
        return self.main_windows[1]

    @property
    def third_window(self) -> int:
        return self.main_windows[2]

    @property
    def cvr_thresholds(self) -> List[int]:
        return sorted(set(self.cvr_threshold_map.values()))


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    window_spec: WindowSpec
    default_fe_cols: str
    feature_mappings_json: str
    model_input_files: Dict[str, str]


KUAILIVE_WINDOW_SPEC = WindowSpec(
    main_windows=(300, 1800, 5400),
    short_long_windows=(30, 3600),
    cvr_threshold_map={
        1: 1,
        5: 5,
        10: 10,
        30: 30,
        60: 60,
        120: 120,
        180: 180,
        240: 240,
        300: 300,
        600: 600,
        900: 900,
        1200: 1200,
        1500: 1500,
        1800: 1800,
        2400: 2400,
        5400: 5400,
    },
)


SEGMM_WINDOW_SPEC = WindowSpec(
    main_windows=(60, 120, 240),
    short_long_windows=(30, 180),
    cvr_threshold_map={
        1: 1,
        5: 1,
        10: 2,
        30: 6,
        60: 12,
        120: 24,
        180: 36,
        240: 48,
        300: 60,
        600: 72,
        900: 84,
        1200: 96,
        1500: 108,
        1800: 120,
        2400: 140,
        5400: 240,
    },
)


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "KuaiLive": DatasetConfig(
        name="KuaiLive",
        window_spec=KUAILIVE_WINDOW_SPEC,
        default_fe_cols="user_id,live_id,streamer_id",
        feature_mappings_json="./dataset/KuaiLive/processed_data_stream/kuailive_feature_mappings.json",
        model_input_files={
            "ORM5_STREAM": "stream5min_with_gt_label.csv",
            "ORM30_STREAM": "stream30min_with_gt_label.csv",
            "ORM90_STREAM": "stream90min_with_gt_label.csv",
            "DEFER-5-30": "window2_5min_30min_full_space_emit.csv",
            "DEFER-5-90": "window2_5min_90min_full_space_emit.csv",
            "ES-DFM-5-30": "window2_5min_30min_conditional_emit.csv",
            "ES-DFM-5-90": "window2_5min_90min_conditional_emit.csv",
            "ORM2W_30S_3600S": "window2_30s_60min_full_space_emit.csv",
            "ORM2W_30S_3600S_DELAYED3": "window2_30s_60min_full_space_emit.csv",
            "ORM3W": "window3_full_space_emit.csv",
            "ORM3W_DELAYED3": "window3_full_space_emit.csv",
        },
    ),
    "SegMM": DatasetConfig(
        name="SegMM",
        window_spec=SEGMM_WINDOW_SPEC,
        default_fe_cols="user_id,live_id",
        feature_mappings_json="./dataset/SegMM/processed_data_stream/segmm_feature_mappings.json",
        model_input_files={
            "ORM5_STREAM": "stream60s_with_gt_label.csv",
            "ORM30_STREAM": "stream120s_with_gt_label.csv",
            "ORM90_STREAM": "stream240s_with_gt_label.csv",
            "DEFER-5-30": "window2_60s_120s_full_space_emit.csv",
            "DEFER-5-90": "window2_60s_240s_full_space_emit.csv",
            "ES-DFM-5-30": "window2_60s_120s_conditional_emit.csv",
            "ES-DFM-5-90": "window2_60s_240s_conditional_emit.csv",
            "ORM2W_30S_3600S": "window2_30s_180s_full_space_emit.csv",
            "ORM2W_30S_3600S_DELAYED3": "window2_30s_180s_full_space_emit.csv",
            "ORM3W": "window3_60s_120s_240s_full_space_emit.csv",
            "ORM3W_DELAYED3": "window3_60s_120s_240s_full_space_emit.csv",
        },
    ),
}


def get_dataset_config(dat_name: str) -> DatasetConfig:
    if dat_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dat_name: {dat_name}. Available: {sorted(DATASET_CONFIGS)}")
    return DATASET_CONFIGS[dat_name]

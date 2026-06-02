from dataclasses import dataclass, field
from typing import Dict, List, Tuple


Key = Tuple[int, int]


@dataclass
class MemoryBank:
    # online prediction records
    record_ts: List[int] = field(default_factory=list)
    record_user: List[int] = field(default_factory=list)
    record_live: List[int] = field(default_factory=list)
    record_label: List[float] = field(default_factory=list)
    record_pred: List[float] = field(default_factory=list)
    record_window: List[int] = field(default_factory=list)

    # per-threshold CVR store for T3 evaluation
    cvr_store: Dict[str, dict] = field(default_factory=dict)
    t3_records: List[dict] = field(default_factory=list)

    # unbias memory
    first_pred_map: Dict[Key, float] = field(default_factory=dict)
    first_cvr_map: Dict[Key, Dict[int, float]] = field(default_factory=dict)
    last_label_map: Dict[Key, Tuple[int, float, float]] = field(default_factory=dict)
    first_seen_ts_ms_map: Dict[Key, int] = field(default_factory=dict)
    start_ts_map: Dict[Key, int] = field(default_factory=dict)

    # raw cvr by latest window
    raw_cvr_map: Dict[int, Dict[Key, Tuple[int, float]]] = field(default_factory=dict)

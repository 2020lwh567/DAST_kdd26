from typing import Dict, Iterable, List, Sequence, Tuple


def as_int_tuple(values: Iterable[int]) -> Tuple[int, ...]:
    return tuple(int(v) for v in values)


def build_three_window_thresholds(main_windows: Sequence[int]) -> Dict[int, List[int]]:
    w1, w2, w3 = as_int_tuple(main_windows)
    if (w1, w2, w3) == (300, 1800, 5400):
        return {
            300: list(range(1, 301)),
            1800: list(range(305, 1801, 5)),
            5400: list(range(1830, 5401, 30)),
        }
    if (w1, w2, w3) == (60, 120, 240):
        return {
            60: list(range(1, 61)),
            120: list(range(63, 121, 3)),
            240: list(range(125, 241, 5)),
        }
    return {
        w1: list(range(1, w1 + 1)),
        w2: list(range(w1 + 1, w2 + 1)),
        w3: list(range(w2 + 1, w3 + 1)),
    }


def build_three_window_weights(main_windows: Sequence[int]) -> Dict[int, List[float]]:
    w1, w2, w3 = as_int_tuple(main_windows)
    if (w1, w2, w3) == (300, 1800, 5400):
        return {
            300: [1.0] * 299 + [3.0],
            1800: [5.0] * 299 + [17.5],
            5400: [30.0] * 119 + [75.0],
        }
    if (w1, w2, w3) == (60, 120, 240):
        return {
            60: [1.0] * 60,
            120: [3.0] * 20,
            240: [5.0] * 24,
        }
    thresholds = build_three_window_thresholds((w1, w2, w3))
    return {win_id: [1.0] * len(thr) for win_id, thr in thresholds.items()}


def build_two_window_thresholds(short_long_windows: Sequence[int]) -> Dict[int, List[int]]:
    w1, w2 = as_int_tuple(short_long_windows)
    if (w1, w2) == (30, 3600):
        return {
            30: list(range(1, 31)),
            3600: list(range(31, 301)) + list(range(305, 1801, 5)) + list(range(1830, 3601, 30)),
        }
    return {
        w1: list(range(1, w1 + 1)),
        w2: list(range(w1 + 1, w2 + 1)),
    }


def build_two_window_weights(short_long_windows: Sequence[int]) -> Dict[int, List[float]]:
    w1, w2 = as_int_tuple(short_long_windows)
    if (w1, w2) == (30, 3600):
        return {
            30: [1.0] * 30,
            3600: [1.0] * 269 + [3.0] + [5.0] * 299 + [17.5] + [30.0] * 59 + [75.0],
        }
    thresholds = build_two_window_thresholds((w1, w2))
    return {win_id: [1.0] * len(thr) for win_id, thr in thresholds.items()}


def segment_bounds_from_windows(main_windows: Sequence[int], tower_ids: Sequence[int]) -> Dict[int, Tuple[float, float]]:
    w1, w2, w3 = as_int_tuple(main_windows)
    all_bounds = {
        w1: (0.0, float(w1)),
        w2: (float(w1), float(w2)),
        w3: (float(w2), float(w3)),
    }
    return {int(win_id): all_bounds[int(win_id)] for win_id in tower_ids}

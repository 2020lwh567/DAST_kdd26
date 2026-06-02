import math
import numpy as np
import itertools


class Fenwick:
    def __init__(self, n):
        self.n = n
        self.bit = [0]*(n+1)
    def add(self, i, v):
        while i <= self.n:
            self.bit[i] += v
            i += i & -i
    def sum(self, i):
        s = 0
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s

def xauc_score(labels, preds):
    data = list(zip(labels, preds))
    data.sort(key=lambda x: x[1])  # pred 升序

    # label 坐标压缩（连续值也 OK：按数值排序即可）
    uniq_labels = sorted(set(labels))
    rank = {y:i+1 for i, y in enumerate(uniq_labels)}  # 1..K
    K = len(uniq_labels)

    fw = Fenwick(K)          # 存“之前组”每个label出现次数
    prev_count = [0]*(K+1)   # 精确拿 equal_pairs
    total_seen = 0

    numerator = 0.0
    denom = 0.0

    for pred, group in itertools.groupby(data, key=lambda x: x[1]):
        cur = []
        cur_total = 0
        for y, _ in group:
            cur.append(rank[y])
            cur_total += 1

        # 统计当前组内 label 频次
        cur_freq = {}
        for r in cur:
            cur_freq[r] = cur_freq.get(r, 0) + 1

        # ① 当前组 vs 之前组（pred 更大）
        if total_seen > 0:
            less_pairs = 0
            equal_pairs = 0
            for r, c in cur_freq.items():
                prev_less = fw.sum(r-1)
                prev_eq = prev_count[r]
                less_pairs += c * prev_less
                equal_pairs += c * prev_eq

            comparable_pairs = total_seen * cur_total - equal_pairs
            denom += comparable_pairs
            numerator += less_pairs

        # 当前组内部（pred ties：组内 0.5）
        if cur_total >= 2:
            total_pairs_in_group = cur_total * (cur_total - 1) / 2.0
            same_label_pairs_in_group = sum(v * (v - 1) / 2.0 for v in cur_freq.values())
            comparable_in_group = total_pairs_in_group - same_label_pairs_in_group
            denom += comparable_in_group
            numerator += 0.5 * comparable_in_group

        # 更新“之前组”统计
        for r, c in cur_freq.items():
            fw.add(r, c)
            prev_count[r] += c
        total_seen += cur_total

    return 1.0 if denom == 0 else numerator / denom

if __name__ == '__main__':
    labels = np.array([1,0,0,0,1,0,1,0])
    preds = np.array([0.9, 0.8, 0.3, 0.1,0.4,0.9,0.66,0.7])
    print(xauc_score(labels, preds))
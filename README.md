# DAST

![Data leakage illustration](illustration.png)

## Overview

We reveal that multi-window streaming in live-streaming recommendation inherently introduces data leakage issue (i.e., short-window training updates 
embeddings that long-window serving-time prediction cannot access). This causes a training–serving mismatch and significantly degrades long-horizon 
staytime prediction. To better analyze this problem, we construct the first benchmark for data leakage detection in live-streaming recommendation. We 
reveal that this issue mainly stems from shared bottom embeddings rather than task-specific heads, and identify a cutoff time beyond which the 
timeliness gains are outweighed by leakage-induced losses. In light of this analysis, we propose a simple but effective approach, called **DAST**, 
which decouples embeddings across different temporal windows to blocks leakage paths while preserving the advantages of multi-window supervision. 
Experiments demonstrate consistent improvements on staytime prediction, and our method can be seamlessly integrated into existing frameworks, making 
it a general and model-agnostic solution for leakage-aware live-streaming recommendation.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Dataset Preparation](#dataset-preparation)
- [Training and Reproduction](#training-and-reproduction)
- [Implemented Baselines](#implemented-baselines)
- [Troubleshooting](#troubleshooting)

## Repository Structure

```text
DAST/
├── dataset/
│   ├── KuaiLive/
│   └── SegMM/
├── model/
│   ├── orm_single.py
│   ├── orm_3window.py
│   ├── orm_3window_delayed3.py
│   ├── orm_3window_esdfm.py
│   └── duration_defer.py
├── utils/
├── train_single.py
├── train_model_streaming.py
├── train_model_streaming_esdfm.py
├── train_30s.py
├── run_kuailive.sh
└── run_segmm.sh
```


## Dataset Preparation

### 1) KuaiLive

1. Download KuaiLive from: <https://imgkkk574.github.io/KuaiLive/>
2. Put raw files under `dataset/KuaiLive/`
3. Run preprocessing:

```bash
python dataset/KuaiLive/preprocess_kuailive_wt.py
python dataset/KuaiLive/split_multi_window_kuailive.py
python dataset/KuaiLive/split_30s_interval.py
```

### 2) SegMM

1. Download SegMM from: <https://github.com/hezy18/SegMMInterest/blob/main/SegMM.md>
2. Put `SegMM_inter.csv` under `dataset/SegMM/`
3. Run preprocessing:

```bash
python dataset/SegMM/preprocess_segmm_wt.py
python dataset/SegMM/split_multi_window_segmm.py
python dataset/SegMM/split_30s_interval.py
```

Expected processed outputs include:
- `dataset/KuaiLive/processed_wt/preprocessed_data_full_window.csv`
- `dataset/KuaiLive/processed_30s_interval/kuailive_30s_interval.csv`
- `dataset/SegMM/processed_wt/segmm_full_window.csv`
- `dataset/SegMM/processed_30s_interval/segmm_30s_interval.csv`

## Training and Reproduction

Use the provided scripts to reproduce experiments:

```bash
bash run_kuailive.sh
bash run_segmm.sh
```

You can also run each model directly (examples below).

### Example: DAST on KuaiLive

```bash
python train_model_streaming.py \
  --model_name ORM3W_DELAYED3 \
  --randseed 61 \
  --dat_name KuaiLive \
  --lr 1e-3 \
  --use_full_features 1 \
  --batch_size 2048 \
  --align_weight 1 \
  --hidden_dim 32 \
  --embed_dim 12 \
  --input_csv dataset/KuaiLive/processed_wt/preprocessed_data_full_window.csv \
  --backbone linear
```

### Example: DAST on SegMM

```bash
python train_model_streaming.py \
  --model_name ORM3W_DELAYED3 \
  --randseed 61 \
  --dat_name SegMM \
  --lr 1e-3 \
  --use_full_features 0 \
  --use_5min_embedding_as_feature 0 \
  --batch_size 2048 \
  --align_weight 1 \
  --hidden_dim 32 \
  --embed_dim 12 \
  --t2 1h \
  --t3 4h \
  --fe_cols user_id,photo_id \
  --w1 1 --w2 2 --w3 4 \
  --input_csv dataset/SegMM/processed_wt/segmm_full_window.csv \
  --backbone linear
```

## Implemented Baselines

The run scripts include:
- Direct Training (`train_single.py`)
- ES-DFM (`train_model_streaming_esdfm.py`, `--use_defer 0`)
- DEFER (`train_model_streaming_esdfm.py`, `--use_defer 1`)
- MS3M (`train_model_streaming.py --model_name ORM3W`)
- Sliver (`train_30s.py`)
- DAST (`train_model_streaming.py --model_name ORM3W_DELAYED3`)

## Troubleshooting

- Ensure dataset paths match the `--input_csv` argument.
- Use the same random seed and preprocessing steps for reproducible comparisons.


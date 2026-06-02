#!/bin/bash
set -e
set -x

GPU_ID=0

# SegMM uses t1 percentage splitting: 30% means the first 30% rows after
# send_timestamp sorting are used for pre-training, and the rest for online
# training/evaluation.
for backbone in "MLP" "DCN"
do
    for modelname in "ORM5_STREAM" "ORM30_STREAM" "ORM90_STREAM" "ES-DFM-5-30" "ES-DFM-5-90" "DEFER-5-30" "DEFER-5-90" "ORM2W_30S_3600S" "ORM3W" "ORM3W_DELAYED3"
    do
        for lr in 1e-3
        do
            for align_weight in 1
            do
                CUDA_VISIBLE_DEVICES=${GPU_ID} python train_model_streaming.py \
                    --model_name ${modelname} \
                    --randseed 61 \
                    --dat_name SegMM \
                    --t1 "30%" \
                    --lr ${lr} \
                    --use_full_features 0 \
                    --batch_size 1024 \
                    --align_weight ${align_weight} \
                    --hidden_dim 32 \
                    --embed_dim 12 \
                    --t3_eval_every_steps 50 \
                    --enable_user_stratified_eval 0 \
                    --backbone ${backbone}
            done
        done
    done
done

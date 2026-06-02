#!/bin/bash
set -e
set -x

 
for backbone in "MLP" "DCN"
do
    for modelname in "ORM5_STREAM" "ORM30_STREAM" "ORM90_STREAM" "ES-DFM-5-30" "ES-DFM-5-90" "DEFER-5-30" "DEFER-5-90" "ORM2W_30S_3600S" "ORM3W" "ORM3W_DELAYED3"
    do
        for use_full_features in 1
        do
            for lr in 1e-3 
            do
                for align_weight in 1
                do
                    CUDA_VISIBLE_DEVICES=0 python train_model_streaming.py --model_name ${modelname} --randseed 61 --dat_name KuaiLive --lr ${lr} --use_full_features ${use_full_features} --batch_size 2048 \
                    --align_weight ${align_weight} --hidden_dim 32 --embed_dim 12 --t3_eval_every_steps 50 --enable_user_stratified_eval 0 --backbone ${backbone}
                done
            done
        done
    done
done
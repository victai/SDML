#!/bin/bash

source /nfs/Sif/victai/pytorch_v4/bin/activate

time python3 run_spotlight.py   --mode="mine" \
                                --loss="adaptive_hinge" \
                                --representation="mylstm" \
                                --epoch 1 \
                                --seq_len 100 \
                                --min_seq_len 0 \
                                --embedding_dim 64 \
                                --neg_mode="no" \
                                --num_negative_samples 20 \
                                --model_path="model2.h5" \
                                --save_model \
                                --calc_map
                                #--step_size 1 \
                                #--create_data

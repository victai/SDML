#!/bin/bash

source /nfs/Sif/victai/pytorch_v4/bin/activate

test_data=$1
output_file=$2
test_data="data/hw3_1/length/test.csv"
output_file="out_length.txt"

time python3 main_length.py --test_data $test_data \
                            --output_file $output_file \
                            --encoder_path="encoder_attn_length.pt" \
                            --decoder_path="decoder_attn_length.pt" \
                            --attn
                            #--train \

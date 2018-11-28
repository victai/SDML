#!/bin/bash

source /nfs/Sif/victai/pytorch_v4/bin/activate

test_data=$1
output_file=$2
test_data="data/hw3_1/rhyme/test.csv"
output_file="rhyme_out.txt"

time python3 rhyme_main.py  --test_data $test_data \
                            --output_file $output_file \
                            --encoder_path="rhyme_encoder.pt" \
                            --decoder_path="rhyme_decoder.pt" \
                            #--train \
                            #--attn

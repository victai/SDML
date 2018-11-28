#!/bin/bash

source /nfs/Sif/victai/pytorch_v4/bin/activate

time python3 main_attn.py --test_data $1 \
                          --output_file $2 \
                          --encoder_path="encoder.pt" \
                          --decoder_path="decoder.pt" \
                          #--train

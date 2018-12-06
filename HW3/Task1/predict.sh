#!/bin/bash

source /nfs/Sif/victai/pytorch_v4/bin/activate

encoder='encoder.pt'
decoder='decoder.pt'

#wget "https://www.dropbox.com/s/pnd3c2yzl7s9165/encoder_all_rhyme.pt?dl=1" -O $encoder
#wget "https://www.dropbox.com/s/60ejpae723ygc2a/decoder_all_rhyme.pt?dl=1" -O $decoder

test_data=$1
output_file=$2

time python3 main_all.py    --test_data $test_data \
                            --output_file $output_file \
                            --encoder_path=$encoder \
                            --decoder_path=$decoder \
                            --use_cuda \
                            #--train

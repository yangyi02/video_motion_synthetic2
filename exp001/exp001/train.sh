#!/bin/bash

export PYTHONPATH=/home/yi/code/video_motion_synthetic2:$PYTHONPATH

# Create directories if not exist
MODEL_PATH=./model
if [[ ! -e $MODEL_PATH ]]; then
    mkdir -p $MODEL_PATH
else
    echo "$MODEL_PATH already exist!"
    exit
fi

CUDA_VISIBLE_DEVICES=1 python ../demo.py --train --data=box --method=unsupervised --train_epoch=10000 --test_interval=100 --test_epoch=10 --learning_rate=0.001 --batch_size=64 --image_size=32 --motion_range=1 --num_frame=3 2>&1 | tee $MODEL_PATH/train.log

#!/bin/bash

source ../../set_path.sh

MODEL_PATH="./model"
if [[ ! -e $MODEL_PATH ]]; then
    echo "$MODEL_PATH not exist!"
    exit
fi

python ../demo.py --test --init_model=./model/final.pth --test_epoch=1 --batch_size=64 --motion_range=1 --image_size=32 --num_frame=3 --display 2>&1 | tee $MODEL_PATH/test.log

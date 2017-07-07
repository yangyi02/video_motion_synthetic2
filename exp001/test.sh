#!/bin/bash

MODEL_PATH="./model"
if [[ ! -e $MODEL_PATH ]]; then
    echo "$MODEL_PATH not exist!"
    exit
fi

python main.py --test --init_model=./model/final.pth --test_epoch=1 --batch_size=64 --motion_range=1 --image_size=32 --num_channel=3 --num_inputs=2 --display 2>&1 | tee $MODEL_PATH/test.log
# python main.py --test --init_model=./model/final.pth --test_epoch=1 --batch_size=64 --motion_range=1 --image_size=64 --num_channel=3 --num_inputs=2 --display --bidirection 2>&1 | tee $MODEL_PATH/test.log

cp test.sh $MODEL_PATH

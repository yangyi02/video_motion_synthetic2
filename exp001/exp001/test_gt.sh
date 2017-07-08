#!/bin/bash

source ../../set_path.sh

python ../demo.py --test_gt --test_epoch=1 --batch_size=32 --motion_range=1 --image_size=32 --num_frame=3 --display

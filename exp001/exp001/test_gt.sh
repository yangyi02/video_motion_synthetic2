#!/bin/bash

export PYTHONPATH=/home/yi/code/video_motion_synthetic2:$PYTHONPATH

python ../demo.py --test_gt --test_epoch=1 --batch_size=32 --motion_range=1 --image_size=32 --num_frame=3 --display

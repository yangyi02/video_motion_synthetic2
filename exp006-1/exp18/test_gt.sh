#!/bin/bash

source ../../set_path.sh

python ../demo.py --test_gt --data=mnist --test_epoch=10 --batch_size=64 --motion_range=2 --image_size=64 --num_frame=3 --num_objects=2 --display --save_display --save_display_dir=./ 2>&1 | tee test_gt.log

sh trim.sh

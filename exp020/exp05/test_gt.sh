#!/bin/bash

source ../../set_path.sh

python ../demo.py --test_gt --data=box --test_epoch=10 --batch_size=64 --motion_range=1 --image_size=32 --num_frame=5 --num_objects=2 --display --save_display --save_display_dir=./ 2>&1 | tee test_gt.log

sh trim.sh

#!/bin/bash

source ../../set_path.sh

python ../demo.py --test --data=mnist --init_model=./model.pth --test_epoch=10 --batch_size=64 --motion_range=1 --image_size=32 --num_frame=5 --display --save_display --save_display_dir=./ 2>&1 | tee test.log

sh trim.sh

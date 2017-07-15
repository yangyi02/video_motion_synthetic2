#!/bin/bash

source ../set_path.sh

python ../data/box_data.py --num_objects=2 --num_frame=5 --batch_size=32 --image_size=32 --motion_range=3 --bg_move --save_display --save_display_dir=./

python ../data/mnist_data.py --num_objects=2 --num_frame=5 --batch_size=32 --image_size=32 --motion_range=3 --save_display --save_display_dir=./

python demo.py --test_gt --test_epoch=1 --batch_size=32 --motion_range=1 --image_size=32 --num_frame=5 --display --save_display --save_display_dir=./

python demo.py --train --data=box --method=unsupervised --save_dir=./ --train_epoch=5 --test_interval=5 --test_epoch=1 --learning_rate=0.001 --batch_size=32 --image_size=32 --motion_range=1 --num_frame=5

python demo.py --test --init_model=model.pth --test_epoch=1 --batch_size=64 --motion_range=1 --image_size=32 --num_frame=5 --display --save_display --save_display_dir=./

sh trim.sh

rm model.pth

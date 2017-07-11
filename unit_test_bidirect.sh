#!/bin/bash

source ./set_path.sh

python data/box_data_bidirect.py --num_objects=4 --num_frame=5 --batch_size=32 --image_size=32 --motion_range=3 --bg_move --save_display --save_display_dir=./

python data/mnist_data_bidirect.py --num_objects=4 --num_frame=5 --batch_size=32 --image_size=32 --motion_range=3 --save_display --save_display_dir=./

sh trim.sh


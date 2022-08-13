#!/bin/bash


#!/bin/bash

rm -rf $3
python ./dataset/tiny_data_gen.py --source_path $1 --data_num 800 --save_dir $2 --mode $3

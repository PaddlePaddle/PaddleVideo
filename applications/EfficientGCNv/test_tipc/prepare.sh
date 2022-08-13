#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1
# MODE be one of ['lite_train_lite_infer']
MODE=$2

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})

# The training params
model_name=$(func_parser_value "${lines[1]}")

trainer_list=$(func_parser_value "${lines[14]}")
echo trainner_list
# MODE be one of ['lite_train_lite_infer']
# if [ ${MODE} = "lite_train_lite_infer" ];then
#    rm -f ./data/ntu/tiny_dataset/*
#    python ./dataset/tiny_data_gen.py  --data_path $1 --label_path $2 --save_dir $3
#fi

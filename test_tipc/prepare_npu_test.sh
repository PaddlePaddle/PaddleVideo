#!/bin/bash

BASEDIR=$(dirname "$0")

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

REPO_ROOT_PATH=$(readlinkf ${BASEDIR}/../)

config_files=$(find ${REPO_ROOT_PATH}/test_tipc/configs -name "train_infer_python.txt")
for file in ${config_files}; do
   echo $file
   sed -i "s/Global.use_gpu/Global.use_npu/g" $file
   sed -i '16s/$/ -o use_npu=True/' $file
   sed -i '24s/$/ -o use_npu=True/' $file
   sed -i '40s/$/ --use_gpu=False/' $file
   sed -i "s/--use_gpu:True|False/--use_npu:True|False/g" $file
done

yaml_files=$(find ${REPO_ROOT_PATH}/configs/recognition/ -name "*.yaml")
for file in ${yaml_files}; do
   echo $file
   sed -i "s/num_workers: 4/num_workers: 0/g" $file
   sed -i "s/num_workers: 8/num_workers: 0/g" $file
done

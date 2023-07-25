#!/bin/bash
source test_tipc/common_func.sh

function readlinkf() {
    perl -MCwd -e "print Cwd::abs_path shift" "$1";
}

function func_parser_config() {
    strs=$1
    IFS=" "
    array=(${strs})
    tmp=${array[2]}
    echo ${tmp}
}

BASEDIR=$(dirname "$0")
REPO_ROOT_PATH=$(readlinkf ${BASEDIR}/../)

FILENAME=$1

# disable mkldnn on non x86_64 env
arch=$(uname -i)
if [ $arch != "x86_64" ]; then
    sed -i "s/--enable_mkldnn:True|False/--enable_mkldnn:False/g" $FILENAME
    sed -i "s/--enable_mkldnn:True/--enable_mkldnn:False/g" $FILENAME
fi

# change gpu to npu in tipc txt configs
sed -i "s/use_gpu/use_npu/g" $FILENAME
# disable benchmark as AutoLog required nvidia-smi command
sed -i "s/--enable_benchmark:True/--enable_benchmark:False/g" $FILENAME
# python has been updated to version 3.9 for npu backend
sed -i "s/python3.7/python3.9/g" $FILENAME
dataline=`cat $FILENAME`

# change gpu to npu in execution script
sed -i "s/\"gpu\"/\"npu\"/g" test_tipc/test_train_inference_python.sh

# pass parameters to test_train_inference_python.sh
cmd="bash test_tipc/test_train_inference_python.sh ${FILENAME} $2"
echo -e "\033[1;32m Started to run command: ${cmd}!  \033[0m"
eval $cmd

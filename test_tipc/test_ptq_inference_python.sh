#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1
# MODE be one of ['lite_train_lite_infer' 'lite_train_whole_infer' 'whole_train_whole_infer', 'whole_infer']
MODE=$2

dataline=$(awk 'NR==1, NR==32{print}'  $FILENAME)

# parser params
IFS=$'\n'
lines=(${dataline})

# The training params
model_name=$(func_parser_value "${lines[1]}")
python=$(func_parser_value "${lines[2]}")
use_gpu_key=$(func_parser_key "${lines[3]}")
use_gpu_value=$(func_parser_value "${lines[3]}")
quant_config_file_key=$(func_parser_key "${lines[4]}")
quant_config_file_value=$(func_parser_value "${lines[4]}")
model_path_key=$(func_parser_key "${lines[5]}")
model_path_value=$(func_parser_value "${lines[5]}")
output_dir_key=$(func_parser_key "${lines[6]}")
output_dir_value=$(func_parser_value "${lines[6]}")
data_dir_key=$(func_parser_key "${lines[7]}")
data_dir_value=$(func_parser_value "${lines[7]}")
data_anno_key=$(func_parser_key "${lines[8]}")
data_anno_value=$(func_parser_value "${lines[8]}")
batch_num_key=$(func_parser_key "${lines[9]}")
batch_num_value=$(func_parser_value "${lines[9]}")
quant_batch_size_key=$(func_parser_key "${lines[10]}")
quant_batch_size_value=$(func_parser_value "${lines[10]}")

# parser trainer
train_py=$(func_parser_value "${lines[13]}")

# parser inference
inference_py=$(func_parser_value "${lines[16]}")
use_gpu_key=$(func_parser_key "${lines[17]}")
use_gpu_list=$(func_parser_value "${lines[17]}")
infer_config_file_key=$(func_parser_key "${lines[18]}")
infer_config_file_value=$(func_parser_value "${lines[18]}")
infer_batch_size_key=$(func_parser_key "${lines[19]}")
infer_batch_size_list=$(func_parser_value "${lines[19]}")
infer_model_key=$(func_parser_key "${lines[20]}")
infer_model_value=$(func_parser_value "${lines[20]}")
infer_params_key=$(func_parser_key "${lines[21]}")
infer_params_value=$(func_parser_value "${lines[21]}")
infer_video_key=$(func_parser_key "${lines[22]}")
infer_video_dir=$(func_parser_value "${lines[22]}")
benchmark_key=$(func_parser_key "${lines[23]}")
benchmark_value=$(func_parser_value "${lines[23]}")


function func_inference(){
    IFS='|'
    _python=$1
    _script=$2
    _model_dir=$3
    _log_path=$4
    _img_dir=$5
    # inference
    for use_gpu in ${use_gpu_list[*]}; do
        # cpu
        if [ ${use_gpu} = "False" ] || [ ${use_gpu} = "cpu" ]; then
            for batch_size in ${infer_batch_size_list[*]}; do
                _save_log_path="${_log_path}/python_infer_cpu_batchsize_${batch_size}.log"
                set_infer_data=$(func_set_params "${infer_video_key}" "${_img_dir}")
                set_benchmark=$(func_set_params "${benchmark_key}" "${benchmark_value}")
                set_batchsize=$(func_set_params "${infer_batch_size_key}" "${batch_size}")
                set_model_file_path=$(func_set_params "${infer_model_key}" "${infer_model_value}")
                set_params_file_path=$(func_set_params "${infer_params_key}" "${infer_params_value}")
                set_config_file_path=$(func_set_params "${infer_config_file_key}" "${infer_config_file_value}")
                command="${_python} ${_script} ${use_gpu_key}=${use_gpu} ${set_config_file_path} ${set_model_file_path} ${set_params_file_path} ${set_batchsize} ${set_infer_data} ${set_benchmark} > ${_save_log_path} 2>&1 "
                # echo $command
                eval $command
                last_status=${PIPESTATUS[0]}
                eval "cat ${_save_log_path}"
                status_check $last_status "${command}" "${status_log}" "${model_name}"
            done
        # gpu
        elif [ ${use_gpu} = "True" ] || [ ${use_gpu} = "gpu" ]; then
            for batch_size in ${infer_batch_size_list[*]}; do
                _save_log_path="${_log_path}/python_infer_gpu_batchsize_${batch_size}.log"
                set_infer_data=$(func_set_params "${infer_video_key}" "${_img_dir}")
                set_benchmark=$(func_set_params "${benchmark_key}" "${benchmark_value}")
                set_batchsize=$(func_set_params "${infer_batch_size_key}" "${batch_size}")
                set_model_file_path=$(func_set_params "${infer_model_key}" "${infer_model_value}")
                set_params_file_path=$(func_set_params "${infer_params_key}" "${infer_params_value}")
                set_config_file_path=$(func_set_params "${infer_config_file_key}" "${infer_config_file_value}")
                command="${_python} ${_script} ${use_gpu_key}=${use_gpu} ${set_config_file_path} ${set_model_file_path} ${set_params_file_path} ${set_batchsize} ${set_infer_data} ${set_benchmark} > ${_save_log_path} 2>&1 "
                echo $command
                eval $command
                last_status=${PIPESTATUS[0]}
                eval "cat ${_save_log_path}"
                status_check $last_status "${command}" "${status_log}" "${model_name}"
            done
        else
            echo "Does not support hardware other than CPU and GPU Currently!"
        fi
    done
}

# log
LOG_PATH="./log/${model_name}/${MODE}"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_python.log"

if [ ${MODE} = "whole_infer" ]; then
    IFS="|"
    # run export
    set_output_dir=$(func_set_params "${output_dir_key}" "${output_dir_value}")
    set_data_dir=$(func_set_params "${data_dir_key}" "${data_dir_value}")
    set_data_anno=$(func_set_params "${data_anno_key}" "${data_anno_value}")
    set_batch_size=$(func_set_params "${quant_batch_size_key}" "${quant_batch_size_value}")
    set_batch_num=$(func_set_params "${batch_num_key}" "${batch_num_value}")
    set_model_path=$(func_set_params "${model_path_key}" "${model_path_value}")
    set_config_file=$(func_set_params "${quant_config_file_key}" "${quant_config_file_value}")
    set_use_gpu=$(func_set_params "${use_gpu_key}" "${use_gpu_value}")

    export_log_path="${LOG_PATH}/${MODE}_export_${Count}.log"
    export_cmd="${python} ${train_py} ${set_use_gpu} ${set_config_file} ${set_model_path} ${set_batch_num} ${set_batch_size} ${set_data_dir} ${set_data_anno} ${set_output_dir} > ${export_log_path} 2>&1 "
    echo $export_cmd
    eval $export_cmd
    status_export=$?
    status_check $status_export "${export_cmd}" "${status_log}" "${model_name}"

    save_infer_dir=${output_dir_value}
    #run inference
    func_inference "${python}" "${inference_py}" "${save_infer_dir}" "${LOG_PATH}" "${infer_video_dir}"

fi

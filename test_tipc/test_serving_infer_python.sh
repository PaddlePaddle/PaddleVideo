#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1
dataline=$(awk 'NR==1, NR==18{print}'  $FILENAME)
MODE=$2

# parser params
IFS=$'\n'
lines=(${dataline})

# parser serving
model_name=$(func_parser_value "${lines[1]}")
python_list=$(func_parser_value "${lines[2]}")
trans_model_py=$(func_parser_value "${lines[3]}")
infer_model_dir_key=$(func_parser_key "${lines[4]}")
infer_model_dir_value=$(func_parser_value "${lines[4]}")
model_filename_key=$(func_parser_key "${lines[5]}")
model_filename_value=$(func_parser_value "${lines[5]}")
params_filename_key=$(func_parser_key "${lines[6]}")
params_filename_value=$(func_parser_value "${lines[6]}")
serving_server_key=$(func_parser_key "${lines[7]}")
serving_server_value=$(func_parser_value "${lines[7]}")
serving_client_key=$(func_parser_key "${lines[8]}")
serving_client_value=$(func_parser_value "${lines[8]}")
serving_dir_value=$(func_parser_value "${lines[9]}")
web_service_py=$(func_parser_value "${lines[10]}")
pipeline_py=$(func_parser_value "${lines[11]}")
video_dir_key=$(func_parser_key "${lines[12]}")
video_dir_value=$(func_parser_value "${lines[12]}")


LOG_PATH="./test_tipc/output/log/${model_name}/${MODE}"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_serving.log"


function func_serving(){
    IFS='|'
    _python=$1
    _script=$2
    _model_dir=$3

    # python serving code
    set_dirname=$(func_set_params "${infer_model_dir_key}" "${infer_model_dir_value}")
    set_model_filename=$(func_set_params "${model_filename_key}" "${model_filename_value}")
    set_params_filename=$(func_set_params "${params_filename_key}" "${params_filename_value}")
    set_serving_server=$(func_set_params "${serving_server_key}" "${serving_server_value}")

    set_serving_client=$(func_set_params "${serving_client_key}" "${serving_client_value}")
    python_list=(${python_list})
    python=${python_list[0]}
    trans_log="${LOG_PATH}/python_trans_model.log"
    trans_model_cmd="${python} ${trans_model_py} ${set_dirname} ${set_model_filename} ${set_params_filename} ${set_serving_server} ${set_serving_client} > ${trans_log} 2>&1 "

    eval ${trans_model_cmd}

    # modify the alias name of fetch_var to "outputs"
    server_fetch_var_line_cmd="sed -i '/fetch_var/,/is_lod_tensor/s/alias_name: .*/alias_name: \"outputs\"/' $serving_server_value/serving_server_conf.prototxt"
    eval ${server_fetch_var_line_cmd}
    client_fetch_var_line_cmd="sed -i '/fetch_var/,/is_lod_tensor/s/alias_name: .*/alias_name: \"outputs\"/' $serving_client_value/serving_client_conf.prototxt"
    eval ${client_fetch_var_line_cmd}

    cd ${serving_dir_value}
    echo 'PWD= '$PWD
    unset https_proxy
    unset http_proxy

    server_log_path="${LOG_PATH}/python_server_gpu.log"
    web_service_cmd="${python} ${web_service_py} > ${server_log_path} 2>&1 &"
    eval $web_service_cmd
    last_status=${PIPESTATUS[0]}
    status_check $last_status "${web_service_cmd}" "${status_log}" "${model_name}"
    sleep 30s # not too short is ok

    _save_log_path="../../${LOG_PATH}/python_server_infer_gpu_batchsize_1.log"
    set_video_dir=$(func_set_params "${video_dir_key}" "${video_dir_value}")
    pipeline_cmd="${python} ${pipeline_py} ${set_video_dir} > ${_save_log_path} 2>&1 "

    eval $pipeline_cmd
    last_status=${PIPESTATUS[0]}

    eval "cat ${_save_log_path}"
    cd ../../
    status_check $last_status "${pipeline_cmd}" "${status_log}" "${model_name}"
    ps ux | grep -E 'web_service|pipeline' | awk '{print $2}' | xargs kill -s 9
}


# set cuda device
GPUID=$3
if [ ${#GPUID} -le 0 ];then
    env=" "
else
    env="export CUDA_VISIBLE_DEVICES=${GPUID}"
fi
set CUDA_VISIBLE_DEVICES
eval $env


echo "################### run test ###################"

export Count=0
IFS="|"
func_serving "${web_service_cmd}"


#!/bin/bash

#0.数据集 PaddleSeg/test_tipc/prepare.sh
######################################## base ######################################
# base 参数: share_mem 默认是true
# num_workers 默认是8
# shuffle默认是true
#python main.py  -c configs/recognition/tsm/tsm_k400_frames.yaml \
#  --seed 1234 --max_iters=150 \
#  -o log_interval=10      \
#  -o output_dir=./test_tipc/output/TSM/norm_train_gpus_0_autocast_fp32 \
#  -o epochs=1 -o MODEL.backbone.pretrained='data/ResNet50_pretrain.pdparams' \
#  -o DATASET.batch_size=30 -o DATASET.train.file_path='data/k400/train_small_frames.list' \
#  -o DATASET.valid.file_path='data/k400/val_small_frames.list' \
#  -o DATASET.test.file_path='data/k400/val_small_frames.list'

#python main.py  -c configs/recognition/tsm/tsm_k400_frames.yaml \
#   --seed 1234 --max_iters=150 \
#   -o log_interval=10 \
#   --amp    \
#   -o output_dir=./test_tipc/output/TSM/norm_train_gpus_0_autocast_fp16 \
#   -o epochs=1 -o MODEL.backbone.pretrained='data/ResNet50_pretrain.pdparams' \
#   -o DATASET.batch_size=30 -o DATASET.train.file_path='data/k400/train_small_frames.list' \
#   -o DATASET.valid.file_path='data/k400/val_small_frames.list' \
#   -o DATASET.test.file_path='data/k400/val_small_frames.list'  
#============================================Perf Summary============================================
#Reader Ratio: 3.955%
#Time Unit: s, IPS Unit: samples/s
#|                 |       avg       |       max       |       min       |
#|   reader_cost   |     0.01459     |     0.02408     |     0.00011     |
#|    batch_cost   |     0.36893     |     0.37670     |     0.34942     |
#|       ips       |     81.89640    |     85.85752    |     79.63950    |
###################################### base end ######################################

batch_size=${1:-4}
precision=${2:-"fp32"} # tf32, fp16_o1, fp16_o2
opt_level=${3:-4}
data_format=${4:-"NCHW"}
if [ "${data_format}" = "NHWC" ]; then
  data_layout="_nhwc"
else
  data_layout=""
fi

echo "data_format = ${data_layout}, batch_size=${batch_size}, precision=${precision}, opt_level=${opt_level}"
export CUDA_VISIBLE_DEVICES="5"
train_precision=fp32
log_iter=10

WORK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/" && pwd )"
export PYTHONPATH=${WORK_ROOT}:${PYTHONPATH}

# opt 1
if [ ${opt_level} -ge 1 ]; then
  use_shared_memory=True
else
  use_shared_memory=False
fi
use_shared_memory=True
  
# opt 2
if [ ${opt_level} -ge 2 ]; then
  num_workers=12
else
  num_workers=4
fi

# opt 3
if [ ${opt_level} -ge 3 ]; then
  export FLAGS_use_autotune=1
  #export FLAGS_cudnn_exhaustive_search=1
else
  export FLAGS_use_autotune=0
fi
export GLOG_vmodule=switch_autotune=3

# opt 4
if [ ${opt_level} -ge 4 ]; then
  export FLAGS_conv_workspace_size_limit=4000 #MB
fi

# opt 5
if [ ${opt_level} -ge 5 ]; then
  prof="_old_prof"
  prof_type="--profiler_options=\"batch_range=[100,110];tracer_option=OpDetail;profile_path=model.profile\""
else
  prof="_new_prof"
  prof_type=""
fi
echo "prof_type ${prof_type}"
export FLAGS_enable_eager_mode=1
if [ "${precision}" = "fp32" ]; then
  export NVIDIA_TF32_OVERRIDE=0
else
  unset NVIDIA_TF32_OVERRIDE
  if [ "${precision}" = "fp16_o1" ]; then
    train_precision=fp16
    amp_args="--amp"
    amp_level_args="--amp_level O1"
  elif [ "${precision}" = "fp16_o2" ]; then
    train_precision=fp16
    amp_args="--amp"
    amp_level_args="--amp_level O2"
  else
    amp_args=""
    amp_level_args=""
  fi
fi

echo "lalala ${amp_args} ${amp_level_args}"
suffix="" #"_profile"
subdir_name=tmp

#all_bn_nhwc_sim_bn_true_profile_true_smi_true #logs_commit
model_name=tSM_bs30_fp32
output_filename=${model_name}_bs${batch_size}_${precision}_logiter${log_iter}.opt_${opt_level}
output_root=/root/paddlejob/workspace/work/niuliling/PaddlePerf/ModelPerf-AMP/PaddleVideo/${model_name}
mkdir -p ${output_root}/${subdir_name}


#opt 6 gpu_percent = True
collect_gpu_status=False
smi=""
if [ "${collect_gpu_status}" = "True" ]; then
  smi="_smi"
  nvidia-smi -i ${CUDA_VISIBLE_DEVICES} --query-gpu=utilization.gpu,utilization.memory --format=csv -lms 100 > ${output_root}/${subdir_name}/gpu_usage_${output_filename}.txt 2>&1 &
  gpu_query_pid=$!
  echo "gpu_query_pid=${gpu_query_pid}"
fi

log_root=${output_root}/${subdir_name}/log_${output_filename}${data_layout}${smi}${prof}

# opt 7 nsys
#nsys_args="nsys profile --stats true -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --capture-range=cudaProfilerApi -x true --force-overwrite true -o ${log_root}"
echo "log_root : ${log_root}"
# opt 8 log_root
file_root_name=${log_root}.txt
echo "======================================================"
echo "NVIDIA_TF32_OVERRIDE            : ${NVIDIA_TF32_OVERRIDE}"
echo "FLAGS_enable_eager_mode         : ${FLAGS_enable_eager_mode}"
echo "FLAGS_use_autotune              : ${FLAGS_use_autotune}"
echo "FLAGS_conv_workspace_size_limit : ${FLAGS_conv_workspace_size_limit}"
echo "use_shared_memory               : ${use_shared_memory}"
echo "num_workers                     : ${num_workers}"
echo "model_name                      : ${model_name}"
echo "output_filename                 : ${output_filename}"
echo "nsys_args                       : ${nsys_args}"
echo "train_precision                 : ${train_precision}"
echo "file_root_name                  : ${file_root_name}"
echo "======================================================"
echo "configs/recognition/tsm/tsm_k400_frames${data_layout}.yaml"

${nsys_args} python main.py  -c configs/recognition/tsm/tsm_k400_frames${data_layout}.yaml \
   --seed 1234 --max_iters=150 \
   -o log_interval=10  ${amp_args} ${amp_level_args} \
   -o output_dir=./test_tipc/output/TSM/norm_train_gpus_0_autocast_${train_precision} \
   -o epochs=1 -o MODEL.backbone.pretrained='data/ResNet50_pretrain.pdparams' \
   -o DATASET.batch_size=30 -o DATASET.train.file_path='data/k400/train_small_frames.list' \
   -o DATASET.valid.file_path='data/k400/val_small_frames.list' \
   -o DATASET.test.file_path='data/k400/val_small_frames.list'  \
   -o DATASET.num_workers=${num_workers} ${prof_type} \
        | tee ${file_root_name} 2>&1

if [ "${collect_gpu_status}" = "True" ]; then
  kill ${gpu_query_pid}
fi

 #--profiler_options="batch_range=[50, 60]; tracer_option=OpDetail; profile_path=model.profile"\

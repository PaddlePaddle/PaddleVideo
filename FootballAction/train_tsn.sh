export CUDA_VISIBLE_DEVICES=0,1

export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.8

home_dir="/home/work"
work_dir="$home_dir/train_proposal"
save_dir="$home_dir/checkpoints/models_tsn"
if [ ! -d "$save_dir" ]; then
    mkdir "$save_dir"
fi

cd $work_dir
# train
LOG="$save_dir/log_train"
python -u train.py \
       --model_name=TSN \
       --config=$work_dir/configs/tsn_football.yaml \
       --save_dir=$save_dir \
       2>&1 | tee $LOG

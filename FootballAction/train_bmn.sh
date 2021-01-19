
export CUDA_VISIBLE_DEVICES=0,1,2,3
home_dir="/home/work"
work_dir="$home_dir/train_proposal"
save_dir="$home_dir/checkpoints/models_bmn"
if [ ! -d "$save_dir" ]; then
    mkdir "$save_dir"
fi

cd $work_dir
# train
LOG="$save_dir/log_train"
python -u train.py \
        --model_name=BMN \
        --config=$work_dir/configs/bmn_football.yaml \
        --save_dir=$save_dir \
        2>&1 | tee $LOG

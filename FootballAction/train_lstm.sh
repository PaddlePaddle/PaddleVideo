
export CUDA_VISIBLE_DEVICES=0,1
home_dir="/home/work"
work_dir="$home_dir/train_lstm"
save_dir="$home_dir/checkpoints/models_lstm"
if [ ! -d "$save_dir" ]; then
    mkdir "$save_dir"
fi

cd $work_dir
LOG="$save_dir/log_train"
python -u scenario_lib/train.py \
    --model_name=ActionNet \
    --config=$work_dir/conf/conf.txt \
    --save_dir=$save_dir \
    --log_interval=5 \
    --valid_interval=1 \
    2>&1 | tee $LOG

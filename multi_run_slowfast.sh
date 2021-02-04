export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

start_time=$(date +%s)

# run slowfast training
python3.7 -B -m paddle.distributed.launch --selected_gpus="0,1,2,3,4,5,6,7" --log_dir=log-slowfast main.py --validate --multigrid -c configs/recognition/slowfast/slowfast_multigrid.yaml -o log_level="INFO"

end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "8 card bs=64, max_epoch=239, epoch_factor=1.5,  warmup_epoch=34, all 400 class, preciseBN 200 iter, valid 10 epoch: build kernel time is $(($cost_time/60))min $(($cost_time%60))s"

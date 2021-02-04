export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# run slowfast multi-grid test
python3.7 -B -m paddle.distributed.launch --selected_gpus="0,1,2,3,4,5,6,7" --log_dir=logs_test test.py  -c ../configs/recognition/slowfast/slowfast_multigrid.yaml -w "../output/SlowFast/SlowFast0_00358.pdparams"

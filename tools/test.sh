export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3.7 -B -m paddle.distributed.launch --selected_gpus="0,1,2,3,4,5,6,7" --log_dir=logs_test test.py  -c ../configs/recognition/slowfast/slowfast.yaml -w "../output/SlowFast/SlowFast_epoch_196.pdparams"

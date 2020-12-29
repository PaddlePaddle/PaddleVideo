export CUDA_VISIBLE_DEVICES=0,1

# run slowfast training
python3.7 -B -m paddle.distributed.launch --selected_gpus="0,1" --log_dir=log_slowfast main.py --validate --multigrid -c configs/recognition/slowfast/slowfast_multigrid.yaml -o log_level="INFO"

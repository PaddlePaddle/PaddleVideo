export CUDA_VISIBLE_DEVICES=0
python3 -B -m paddle.distributed.launch --selected_gpus=0 main.py -c configs/frame.yaml -o log_level="INFO"

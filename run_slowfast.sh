export CUDA_VISIBLE_DEVICES=0,1,2,3 #,4,5,6,7

# run slowfast training
python3.7 -B -m paddle.distributed.launch --selected_gpus="0,1,2,3" --log_dir=log_slowfast  main.py --validate -c configs/recognition/slowfast/slowfast.yaml -o log_level="INFO"

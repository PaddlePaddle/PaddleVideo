export CUDA_VISIBLE_DEVICES=0
#python3.7 -B -m paddle.distributed.launch --selected_gpus="0" --log_dir=log_slowfast  main.py --parallel --validate -c configs/recognition/slowfast/slowfast.yaml -o log_level="INFO" -o DATASET.batch_size=8

python3.7 main.py --validate -c configs/recognition/slowfast/slowfast.yaml -o log_level="INFO" -o DATASET.batch_size=8


#python3 -B -m paddle.distributed.launch --selected_gpus="0,3" main.py --parallel -c configs/example.yaml -o log_level="INFO" -o DATASET.batch_size=16

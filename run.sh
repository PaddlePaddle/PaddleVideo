export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -B -m paddle.distributed.launch --selected_gpus="2,3" --log_dir=log_tsm  main.py --parallel --validate -c configs/recognition/tsm/tsm.yaml -o log_level="INFO" -o DATASET.batch_size=16

#python3 -B -m paddle.distributed.launch --selected_gpus="0,3" main.py --parallel -c configs/example.yaml -o log_level="INFO" -o DATASET.batch_size=16


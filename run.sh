export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# run tsm training
#python3 -B -m paddle.distributed.launch --selected_gpus="0,1,2,3"  main.py  --validate -c configs/recognition/tsm/tsm.yaml -o log_level="INFO" -o DATASET.batch_size=16

# run slowfast training
#python3.7 -B -m paddle.distributed.launch --selected_gpus="0,1,2,3,4,5,6,7" --log_dir=log_slowfast  main.py --validate -c configs/recognition/slowfast/slowfast.yaml -o log_level="INFO"

# run bmn training
#python3.7 -B -m paddle.distributed.launch --selected_gpus="0,1,2,3"  --log_dir=log_bmn main.py  --validate -c configs/localization/bmn.yaml

#python3 -B -m paddle.distributed.launch --selected_gpus="0,3" main.py --parallel -c configs/example.yaml -o log_level="INFO" -o DATASET.batch_size=16

# run attention_lstm training
#python3.7 -B -m paddle.distributed.launch --started_port 1989 --log_dir=log.attetion_lstm  main.py  --validate -c configs/recognition/attention_lstm/attention_lstm.yaml -o log_level="INFO"

# run pptsm training
python3.7 -B -m paddle.distributed.launch --selected_gpus="0,1,2,3"  --log_dir=log_tsm.pptsm  main.py  --validate -c configs/recognition/tsm/pptsm.yaml -o log_level="INFO"

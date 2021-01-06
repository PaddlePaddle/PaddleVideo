export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# run tsm training
#python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3" --log_dir=log_tsm main.py  --validate -c configs/recognition/tsm/tsm.yaml -o log_level="INFO" -o DATASET.batch_size=16

# run tsn training
#python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3" --log_dir=log_tsn main.py  --validate -c configs/recognition/tsn/tsn.yaml -o log_level="INFO"

# run slowfast training
#python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_slowfast  main.py --validate -c configs/recognition/slowfast/slowfast.yaml -o log_level="INFO"

# run bmn training
#python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3"  --log_dir=log_bmn main.py  --validate -c configs/localization/bmn.yaml

# run attention_lstm training
#python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_attetion_lstm  main.py  --validate -c configs/recognition/attention_lstm/attention_lstm.yaml -o log_level="INFO"

# run pp-tsm training
python3.7 -B -m paddle.distributed.launch --selected_gpus="0,1,2,3"  --log_dir=log_pptsm  main.py  --validate -c configs/recognition/tsm/pptsm.yaml -o log_level="INFO"

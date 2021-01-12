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


# test.sh
# just use `example` as example, please replace to real name.
#python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_test test.py  -c configs/example.yaml -w "output/example/example_best.pdparams"

#NOTE: run bmn test, only support single card, bs=1
#python3.7 test.py -c configs/localization/bmn.yaml -w output/BMN/BMN_epoch_00010.pdparams  -o DATASET.batch_size=1

# export_models script
# just use `example` as example, please replace to real name.
#python3.7 tools/export_model.py -c configs/example.yaml -p output/example/example_best.pdparams -o ./inference

# predict script
# just use `example` as example, please replace to real name.
#python3.7 tools/predict.py -v example.avi --model_file "./inference/example.pdmodel" --params_file "./inference/example.pdiparams" --enable_benchmark=False --model="example" --num_seg=8

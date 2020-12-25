export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# run slowfast test
python3 -B -m paddle.distributed.launch --gpus="0,1,2,3" --log_dir=log_test tools/test.py  -c configs/recognition/tsn/tsn.yaml -w "output/TSN/TSN_epoch_00080.pdparams" -o DATASET.batch_size=2

#python3.7 -B -m paddle.distributed.launch --selected_gpus="0,1,2,3,4,5,6,7" --log_dir=logs_test test.py  -c ../configs/recognition/slowfast/slowfast.yaml -w "../output/SlowFast/SlowFast_epoch_00196.pdparams"

# run bmn test, only support single card, bs=1
#python3.7 test.py -c ../configs/localization/bmn.yaml -w ../output/BMN/BMN_epoch_00001.pdparams  -o DATASET.batch_size=1

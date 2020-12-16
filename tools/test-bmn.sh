export CUDA_VISIBLE_DEVICES=0

# only support single card, bs=1 test
python3.7 test.py -c ../configs/localization/bmn.yaml -w ../output/BMN/BMN_epoch_00001.pdparams  -o DATASET.batch_size=1

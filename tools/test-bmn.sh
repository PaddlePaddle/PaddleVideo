export CUDA_VISIBLE_DEVICES=0


# only support single card test
python3.7 test.py  --validate -c ../configs/localization/bmn.yaml -w ../output/BMN/BMN_epoch_9.pdparams

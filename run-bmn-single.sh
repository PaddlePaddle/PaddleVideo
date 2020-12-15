export CUDA_VISIBLE_DEVICES=0
python3.7 main.py --validate -c configs/localization/bmn.yaml

#python3.7 main.py  --validate -c configs/localization/bmn.yaml
#TODO: no top1 when valid, modify train.py

#python3 -B -m paddle.distributed.launch --selected_gpus="0,3" main.py --parallel -c configs/example.yaml -o log_level="INFO" -o DATASET.batch_size=16

export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -B -m paddle.distributed.launch --selected_gpus="2"  test.py  --validate -c configs/recognition/tsm/tsm.yaml -o log_level="INFO" -o DATASET.batch_size=16 -o MODEL.backbone.pretrained=""

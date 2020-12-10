export CUDA_VISIBLE_DEVICES=0
python3.7 test.py  -c ../configs/recognition/slowfast/slowfast.yaml -o DATASET.batch_size=4 -w "/workspace/huangjun12/PaddleProject/PaddleVideo/Pr/PaddleVideo/output/SlowFast/SlowFast_epoch_10.pdparams"

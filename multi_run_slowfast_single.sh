export CUDA_VISIBLE_DEVICES=0

# run slowfast training
python3.7 -B main.py --validate --multigrid -c configs/recognition/slowfast/slowfast_multi.yaml -o log_level="INFO"

#python3.7 -B main.py --validate -c configs/recognition/slowfast/slowfast.yaml -o log_level="INFO"

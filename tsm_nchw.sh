export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/workspace/PaddleVideo/:$PYTHONPATH
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_cudnn_batchnorm_spatial_persistent=1
python main.py  --amp  -c configs/recognition/tsm/tsm_ucf101_frames.yaml #configs/recognition/tsm/tsm_k400_frames.yaml

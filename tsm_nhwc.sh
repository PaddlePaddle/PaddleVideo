export FLAGS_conv_workspace_size_limit=1000 #MB
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_cudnn_batchnorm_spatial_persistent=1
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/workspace/PaddleVideo/:$PYTHONPATH
python main.py --amp  -c configs/recognition/tsm/tsm_ucf101_frames_nhwc.yaml

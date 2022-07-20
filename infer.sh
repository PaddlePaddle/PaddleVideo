python3.7 tools/predict.py --input_file data/example.avi \
                           --config configs/recognition/movinet/movinet_k400_frame.yaml \
                           --model_file inference/MoViNetA0/MoViNet.pdmodel \
                           --params_file inference/MoViNetA0/MoViNet.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False

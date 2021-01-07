export CUDA_VISIBLE_DEVICES=0 #,1,2,3,4,5,6,7

# run slowfast test
#python3.7 video_tag.py  -c configs/recognition/tsn/tsn_video_tag.yaml --weights "/workspace/huangjun12/PaddleProject/PaddleVideo/Pr/videotag/weights/tsn.pdparams"

python3.7 video_tag.py  -ec configs/recognition/tsn/tsn_video_tag.yaml -pc configs/recognition/attention_lstm/attention_lstm_video_tag.yaml --weights "/workspace/huangjun12/PaddleProject/PaddleVideo/Pr/videotag/weights/tsn.pdparams"

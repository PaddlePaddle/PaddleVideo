CURRENT=`date "+%Y-%m-%d %H:%M:%S"`
echo "start time: $CURRENT"

export CUDA_VISIBLE_DEVICES=4,5,6,7
python3.7 -B -m paddle.distributed.launch --log_dir=log_pptsm  main.py  --validate -c configs/recognition/tsm/pptsm.yaml -o log_level="INFO" 

CURRENT=`date "+%Y-%m-%d %H:%M:%S"`
echo "end time: $CURRENT"

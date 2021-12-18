export CUDA_VISIBLE_DEVICES=0

# run  training
python3.7 -B -m paddle.distributed.launch --gpus="0"  --log_dir=log_pptsm  main.py --amp  --validate -c configs/recognition/tsm/pptsm_regression.yaml

# run testing
#python3.7 -m paddle.distributed.launch --gpus="0,1,2,3" --log_dir=log_pptsm main.py -c configs/recognition/tsm/pptsm_regression.yaml --test --weights=output/model_name/ppTSM_best.pdparams

#finetune
#python3 -m paddle.distributed.launch --gpus="0,1,2,3" main.py --amp -c ./configs/recognition/tsm/pptsm_regression.yaml --validate --weights=./output/model_name/ppTSM_best.pdparams

#resume
#python3 -m paddle.distributed.launch --gpus="0,1,2,3" main.py --amp -c ./configs/recognition/tsm/pptsm_regression.yaml --validate -o resume_epoch=2
# export_models script
# just use `example` as example, please replace to real name.
#python3.7 tools/export_model.py -c configs/example.yaml -p output/model_name/ppTSM_best.pdparams -o ./inference

# predict script
# just use `example` as example, please replace to real name.
#python3.7 tools/predict.py -v example.avi --model_file "./inference/example.pdmodel" --params_file "./inference/example.pdiparams" --enable_benchmark=False --model="example" --num_seg=8

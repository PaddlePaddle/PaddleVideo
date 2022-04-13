PRETRAIN_MODEL='/home/lc/PaddleVideo/applications/Ma-Net/saved_model/DeeplabV3_coco.pdparams'
VOS_SAVE_RESULT_DIR='/home/lc/PaddleVideo/applications/Ma-Net/saved_model/MaNet_davis2017_stage1.pdparams'
#VOS_SAVE_RESULT_DIR='/home/lc/PaddleVideo/applications/Ma-Net/saved_model/stage1'
INT_SAVE_RESULT_DIR='/home/lc/PaddleVideo/applications/Ma-Net/saved_model/MANet_davis2017.pdparams'
#INT_SAVE_RESULT_DIR='/home/lc/PaddleVideo/applications/Ma-Net/saved_model/stage2'
INT_RESULT_DIR='/home/lc/PaddleVideo/applications/Ma-Net/saved_model/result'
RESCALE=416
RANDOMCROP=416
DATA_ROOT='/home/lc/PaddleVideo/data/DAVIS'
echo 'Stage1 training'
CUDA_VISIBLE_DEVICE=3 python train_stage1.py --SAVE_RESULT_DIR $VOS_SAVE_RESULT_DIR --PRETRAINED_MODEL $PRETRAIN_MODEL --DATA_ROOT $DATA_ROOT --TRAIN_BATCH_SIZE 2 --DATA_RESCALE $RESCALE --DATA_RANDOMCROP $RANDOMCROP --TRAIN_LR 0.0007  --MODEL_MAX_LOCAL_DISTANCE 12
echo 'Stage2 training'
python train_stage2.py --SAVE_RESULT_DIR $INT_SAVE_RESULT_DIR --SAVE_VOS_RESULT_DIR $VOS_SAVE_RESULT_DIR --DATA_ROOT $DATA_ROOT --DATA_RESCALE $RESCALE --DATA_RANDOMCROP $RANDOMCROP  --PRETRAINED_MODEL $PRETRAIN_MODEL
echo 'Testing'
python test.py --DATA_ROOT $DATA_ROOT --SAVE_RESULT_DIR $INT_SAVE_RESULT_DIR  --RESULT_ROOT $INT_RESULT_DIR --MODEL_USEIntSeg False --TEST_MODE True

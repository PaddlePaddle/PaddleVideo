import paddle
import argparse
import os
import sys
import cv2
import time


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='intvos config')
parser.add_argument('--ROOT_DIR',
                    type=str,
                    default=os.path.abspath(
                        os.path.join(os.path.dirname("__file__"))))
parser.add_argument('--EXP_NAME', type=str, default='deeplabv3+coco')
parser.add_argument('--SAVE_RESULT_DIR', type=str, default='../afs/result/')
parser.add_argument('--SAVE_VOS_RESULT_DIR', type=str, default='')
parser.add_argument('--NUM_WORKER', type=int, default=4)
parser.add_argument('--KNNS', type=int, default=1)
parser.add_argument('--PRETRAINED_MODEL',
                    type=str,
                    default='./model_best.pth.tar')
parser.add_argument(
    '--RESULT_ROOT',
    type=str,
    default=os.path.join('../afs/vos_result/result_total_80000'))
######DATA_CONFIG
parser.add_argument('--DATA_NAME', type=str, default='COCO2017')
parser.add_argument('--DATA_AUG', type=str2bool, default=True)
parser.add_argument('--DATA_WORKERS', type=int, default=4)
parser.add_argument('--DATA_RESCALE', type=int, default=416)
parser.add_argument('--DATA_RANDOMCROP', type=int, default=416)
parser.add_argument('--DATA_RANDOMROTATION', type=int, default=0)
parser.add_argument('--DATA_RANDOM_H', type=int, default=10)
parser.add_argument('--DATA_RANDOM_S', type=int, default=10)
parser.add_argument('--DATA_RANDOM_V', type=int, default=10)
parser.add_argument('--DATA_RANDOMFLIP', type=float, default=0.5)
parser.add_argument('--DATA_ROOT', type=str, default='../data/DAVIS')

######MODEL_CONFIG
parser.add_argument('--MODEL_NAME', type=str, default='deeplabv3plus')
parser.add_argument('--MODEL_BACKBONE', type=str, default='res101_atrous')
parser.add_argument('--MODEL_OUTPUT_STRIDE', type=int, default=16)
parser.add_argument('--MODEL_ASPP_OUTDIM', type=int, default=256)
parser.add_argument('--MODEL_SHORTCUT_DIM', type=int, default=48)
parser.add_argument('--MODEL_SHORTCUT_KERNEL', type=int, default=1)
parser.add_argument('--MODEL_NUM_CLASSES', type=int, default=21)
parser.add_argument('--MODEL_SEMANTIC_EMBEDDING_DIM', type=int, default=100)
parser.add_argument('--MODEL_HEAD_EMBEDDING_DIM', type=int, default=256)
parser.add_argument('--MODEL_LOCAL_DOWNSAMPLE', type=str2bool, default=True)
parser.add_argument('--MODEL_MAX_LOCAL_DISTANCE', type=int, default=12)
parser.add_argument('--MODEL_SELECT_PERCENT', type=float, default=0.8)
parser.add_argument('--MODEL_USEIntSeg', type=str2bool, default=False)

######TRAIN_CONFIG
parser.add_argument('--TRAIN_LR', type=float, default=0.0007)
parser.add_argument('--TRAIN_LR_GAMMA', type=float, default=0.1)
parser.add_argument('--TRAIN_MOMENTUM', type=float, default=0.9)
parser.add_argument('--TRAIN_WEIGHT_DECAY', type=float, default=0.00004)
parser.add_argument('--TRAIN_POWER', type=float, default=0.9)
parser.add_argument('--TRAIN_BATCH_SIZE', type=int, default=2)
parser.add_argument('--TRAIN_SHUFFLE', type=str2bool, default=True)
parser.add_argument('--TRAIN_CLIP_GRAD_NORM', type=float, default=5.)
parser.add_argument('--TRAIN_MINEPOCH', type=int, default=9)
parser.add_argument('--TRAIN_TOTAL_STEPS', type=int, default=101000)
parser.add_argument('--TRAIN_LOSS_LAMBDA', type=int, default=0)
parser.add_argument('--TRAIN_TBLOG', type=str2bool, default=False)
parser.add_argument('--TRAIN_BN_MOM', type=float,
                    default=0.9997)  # fixed. difs between paddle and torch.
parser.add_argument('--TRAIN_TOP_K_PERCENT_PIXELS', type=float, default=0.15)
parser.add_argument('--TRAIN_HARD_MINING_STEP', type=int, default=50000)
parser.add_argument('--TRAIN_LR_STEPSIZE', type=int, default=2000)
parser.add_argument('--TRAIN_INTER_USE_TRUE_RESULT',
                    type=str2bool,
                    default=True)
parser.add_argument('--TRAIN_RESUME_DIR', type=str, default='')

parser.add_argument('--LOG_DIR', type=str, default=os.path.join('./log'))

parser.add_argument('--TEST_CHECKPOINT',
                    type=str,
                    default='save_step_100000.pth')
parser.add_argument('--TEST_MODE', type=str2bool, default=False)

cfg = parser.parse_args()
cfg.TRAIN_EPOCHS = int(200000 * cfg.TRAIN_BATCH_SIZE / 60.)

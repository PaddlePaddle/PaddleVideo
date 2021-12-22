# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import paddle
import os,sys
import copy as cp
import cv2
import math

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from paddlevideo.modeling.builder import build_model
from paddlevideo.utils import get_config
from paddlevideo.loader.builder import build_dataloader, build_dataset, build_pipeline
from paddlevideo.metrics.ava_utils import read_labelmap

import time
from os import path as osp
import numpy as np
from paddlevideo.utils import get_config
import pickle

from paddlevideo.utils import (get_logger,load, mkdir, save)

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1

FPS = 30 # 每秒抽取多少帧图像

def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))

plate_blue = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
plate_blue = plate_blue.split('-')
plate_blue = [hex2color(h) for h in plate_blue]
plate_green = '004b23-006400-007200-008000-38b000-70e000'
plate_green = plate_green.split('-')
plate_green = [hex2color(h) for h in plate_green]

def abbrev(name):
    """Get the abbreviation of label name:
    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find('(') != -1:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed + 1:]
    return name

# annotations is pred results
def visualize(frames, annotations, plate=plate_blue, max_num=5):
    """Visualize frames with predicted annotations.
    Args:
        frames (list[np.ndarray]): Frames for visualization, note that
            len(frames) % len(annotations) should be 0.
        annotations (list[list[tuple]]): The predicted results.
        plate (str): The plate used for visualization. Default: plate_blue.
        max_num (int): Max number of labels to visualize for a person box.
            Default: 5，目前不能大于5.
    Returns:
        list[np.ndarray]: Visualized frames.
    """

    assert max_num + 1 <= len(plate)
    plate = [x[::-1] for x in plate]
    frames_ = cp.deepcopy(frames)
    nf, na = len(frames), len(annotations)
    assert nf % na == 0
    nfpa = len(frames) // len(annotations)
    anno = None
    h, w, _ = frames[0].shape
    # proposals被归一化需要还原真实坐标值
    scale_ratio = np.array([w, h, w, h])
   
    for i in range(na):
        anno = annotations[i]
        if anno is None:
            continue
        for j in range(nfpa):
            ind = i * nfpa + j
            frame = frames_[ind]
            for ann in anno:
                box = ann[0]
                label = ann[1]
                if not len(label):
                    continue
                score = ann[2]
                box = (box * scale_ratio).astype(np.int64)
                st, ed = tuple(box[:2]), tuple(box[2:])
                cv2.rectangle(frame, st, ed, plate[0], 2)
                for k, lb in enumerate(label):
                    if k >= max_num:
                        break
                    text = abbrev(lb)
                    text = ': '.join([text, str(score[k])])
                    location = (0 + st[0], 18 + k * 18 + st[1])
                    textsize = cv2.getTextSize(text, FONTFACE, FONTSCALE,
                                               THICKNESS)[0]
                    textwidth = textsize[0]
                    diag0 = (location[0] + textwidth, location[1] - 14)
                    diag1 = (location[0], location[1] + 2)
                    cv2.rectangle(frame, diag0, diag1, plate[k + 1], -1)
                    cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                                FONTCOLOR, THICKNESS, LINETYPE)

    return frames_


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    # general params
    parser = argparse.ArgumentParser("PaddleVideo Inference model script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/example.yaml',
                        help='config file path')

    parser.add_argument('--video', help='video file/url')

    parser.add_argument('-o',
                        '--override',
                        action='append',
                        default=[],
                        help='config options to be overridden')
    parser.add_argument('-w',
                        '--weights',
                        type=str,
                        help='weights for finetuning or testing')

    parser.add_argument("-i", "--input_file", type=str, help="input file path")
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--params_file", type=str)

    # detection_result_dir,frame_dir
    parser.add_argument('--detection_result_dir', help='the object detection result dir of extracted frames')
    parser.add_argument('--frame_dir', help='the dir of frames extracted with FPS frame rate ')

    # params for predict
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=8000)
    parser.add_argument("--enable_benchmark", type=str2bool, default=False)
    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=None)
    parser.add_argument(
        '--out-filename',
        default='ava_det_demo.mp4',
        help='output filename')
    parser.add_argument(
        '--predict-stepsize',
        default=8,
        type=int,
        help='give out a prediction per n frames')
    parser.add_argument(
        '--output-stepsize',
        default=4,
        type=int,
        help=('show one frame per n frames in the demo, we should have: '
              'predict_stepsize % output_stepsize == 0'))
    parser.add_argument(
        '--output-fps',
        default=1,
        type=int,
        help='the fps of demo video output')

    return parser.parse_args()

# 一帧的结果。根据概率大小进行排序
def pack_result(human_detection, result):
    """Short summary.
    Args:
        human_detection (np.ndarray): Human detection result.
        result (type): The predicted label of each human proposal.
    Returns:
        tuple: Tuple of human proposal, label name and label score.
    """
    results = []
    if result is None:
        return None

    for prop, res in zip(human_detection, result):
        res.sort(key=lambda x: -x[1])
        
        results.append(
            (prop, [x[0] for x in res], [x[1] for x in res]))
    
    return results

# 构造数据处理需要的results
def get_timestep_result(frame_dir,timestamp,clip_len,frame_interval):
    result = {}

    result["frame_dir"] = frame_dir

    frame_num = len(os.listdir(frame_dir))

    dir_name = frame_dir.split("/")[-1]
    result["video_id"] = dir_name

    result['timestamp']=timestamp

    timestamp_str = '{:04d}'.format(timestamp)
    img_key=dir_name+","+timestamp_str
    result['img_key'] = img_key

    result['shot_info']= (1, frame_num)
    result['fps']= FPS

    result['suffix'] = '{:05}.jpg'

    result['timestamp_start'] =1
    result['timestamp_end'] =int(frame_num/result['fps'])

    return result


class PackOutput(object):
    """
    In slowfast model, we want to get slow pathway from fast pathway based on
    alpha factor.
    Args:
        alpha(int): temporal length of fast/slow
    """
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, fast_pathway):
        # sample num points between start and end
        slow_idx_start = 0
        slow_idx_end = fast_pathway.shape[0] - 1
        slow_idx_num = fast_pathway.shape[0] // self.alpha
        slow_idxs_select = np.linspace(slow_idx_start, slow_idx_end,
                                       slow_idx_num).astype("int64")
        slow_pathway = fast_pathway[slow_idxs_select]

        # T H W C -> C T H W.
        slow_pathway = slow_pathway.transpose(3, 0, 1, 2)
        fast_pathway = fast_pathway.transpose(3, 0, 1, 2)

        # slow + fast
        frames_list = [slow_pathway, fast_pathway]
        results['imgs'] = frames_list
        return results

def get_detection_result(txt_file_path,img_h,img_w,person_det_score_thr):
    """
    根据检测结果文件得到图像中人的检测框(proposals)和置信度（scores）
    txt_file_path:检测结果存放路径
    img_h:图像高度
    img_w:图像宽度
    """

    proposals = []
    scores = []

    with open(txt_file_path,'r') as detection_file:
        lines = detection_file.readlines()
        for line in lines: # person 0.9842637181282043 0.0 469.1407470703125 944.7770385742188 831.806396484375
            items = line.split(" ")
            if items[0]!='person': #只要人
                continue
            
            score = items[1]

            if (float)(score)<person_det_score_thr:
                continue

            x1 = (float(items[2]))/img_w
            y1 = ((float)(items[3]))/img_h
            box_w = ((float)(items[4]))
            box_h = ((float)(items[5]))

            x2 = (float(items[2])+box_w)/img_w
            y2 = (float(items[3])+box_h)/img_h

            scores.append(score)

            proposals.append([x1,y1,x2,y2])

    return np.array(proposals),np.array(scores)
    
@paddle.no_grad()
def main(args): #detection_result_dir,frame_dir
    """
    detection_result_dir: 目标检测结果所在文件夹
    frame_dir: 视频以FPS抽帧结果所在文件夹
    """

    detection_result_dir = args.detection_result_dir
    frame_dir = args.frame_dir

    config = get_config(args.config, show=False)#解析配置文件

    # FPS帧率抽帧得到的帧列表
    frame_name_list = os.listdir(frame_dir)
    original_frames = []
    frame_paths = [] # 帧的全路径
    for frame_name in frame_name_list:
        full_path = os.path.join(frame_dir,frame_name)
        frame_paths.append(full_path)

        frame = cv2.imread(full_path)
        original_frames.append(frame)
    #按照帧名字排个序，后面需要按照顺序拿    
    frame_paths.sort() 
    num_frame = len(frame_paths) #视频秒数*FPS
    # 帧图像高度和宽度
    h, w, _ = original_frames[0].shape

    # Get clip_len, frame_interval and calculate center index of each clip
    data_process_pipeline = build_pipeline(config.PIPELINE.test) #测试时输出处理流水配置
    
    clip_len = config.PIPELINE.test.sample['clip_len']
    assert clip_len % 2 == 0, 'We would like to have an even clip_len'
    frame_interval = config.PIPELINE.test.sample['frame_interval']
    
    # 此处关键帧每秒取一个
    timestamps = np.arange(1,math.ceil(num_frame/FPS)+1)
    print("*** timetamps:",timestamps)
    
    # Load label_map
    label_map_path = config.DATASET.test['label_file']
    categories, class_whitelist = read_labelmap(open(label_map_path))
    label_map = {}
    for item in categories:
        id = item['id']
        name = item['name']
        label_map[id] = name

    # Construct model.
    if config.MODEL.backbone.get('pretrained'):
        config.MODEL.backbone.pretrained = ''  # disable pretrain model init
    model = build_model(config.MODEL)

    model.eval()
    state_dicts = load(args.weights)
    model.set_state_dict(state_dicts)

    print('Performing SpatioTemporal Action Detection for each clip')

    # 存储所有关键帧人的检测结果
    human_detections = []
    # 模型输出
    predictions = []

    # 遍历每个时间戳
    for timestamp in timestamps:
        frame_name = "{:05}.jpg".format(timestamp)
        frame_path = os.path.join(detection_result_dir,frame_name)

        detection_txt_path = frame_path.replace("jpg","txt")
        detection_txt_path = os.path.join(detection_result_dir,detection_txt_path.split("/")[-1])
        if not os.path.exists(detection_txt_path):
            print(detection_txt_path,"not exists!")
            continue

        # proposals需要归一化
        proposals,scores = get_detection_result(detection_txt_path,h,w,(float)(config.DATASET.test['person_det_score_thr']))

        human_detections.append(proposals)

        if proposals.shape[0] == 0:
            predictions.append(None)
            continue
        
        # 获取训练、评估格式的results
        result = get_timestep_result(frame_dir,timestamp,clip_len,frame_interval)
        result["proposals"] = proposals
        result["scores"] = scores
        
        new_result = data_process_pipeline(result)
        proposals = new_result['proposals']# 此过程中，proposals经过reshape

        img_slow = new_result['imgs'][0]
        # 添加第0维
        img_slow = img_slow[np.newaxis, :]
        img_fast = new_result['imgs'][1]
        # 添加第0维
        img_fast = img_fast[np.newaxis, :]

        # 添加第0维，batch维度
        proposals=proposals[np.newaxis, :]
        
        # 添加第0维
        scores = scores[np.newaxis, :]

        img_shape = np.asarray(new_result['img_shape'])
        img_shape = img_shape[np.newaxis, :]
        
        #print("**** proposals:",proposals)
        data = [paddle.to_tensor(img_slow,dtype='float32'),paddle.to_tensor(img_fast,dtype='float32'),paddle.to_tensor(proposals,dtype='float32'),scores,paddle.to_tensor(img_shape,dtype='int32')]

        with paddle.no_grad():
            result = model(data,mode='infer') #推理

            #*** result: <class 'list'> 1 <class 'list'> 80 <class 'numpy.ndarray'> (2, 5)
            #print(" *** result:",type(result),len(result),type(result[0]),len(result[0]),type(result[0][0]),result[0][0].shape)
           
            result = result[0] # batch维度第一个，batch_size= 1
            prediction = []

            person_num = proposals.shape[1]
            # N proposals
            for i in range(person_num): # proposals(batch,person_num,bbox)(1,2,4)表示有2个人
                prediction.append([])
            
            # Perform action score thr
            for i in range(len(result)): # 80个类别
                if i + 1 not in class_whitelist:
                    continue
                for j in range(person_num):
                    if result[i][j, 4] > config.MODEL.head['action_thr']:
                        prediction[j].append((label_map[i + 1], result[i][j,4])) #label_map[i + 1]，+1是因为label_map中index从1开始
            predictions.append(prediction)

    #print("*** human_detections:",type(human_detections[0]),human_detections[0].shape,human_detections[0])
    #print("*** predictions:",type(predictions[0]),predictions[0],len(predictions[0]))

    results = []
    for human_detection, prediction in zip(human_detections, predictions):
        results.append(pack_result(human_detection, prediction))

    def dense_timestamps(timestamps, n):
        """Make it nx frames."""
        old_frame_interval = (timestamps[1] - timestamps[0])
        start = timestamps[0] - old_frame_interval / n * (n - 1) / 2
        new_frame_inds = np.arange(
            len(timestamps) * n) * old_frame_interval / n + start
        return new_frame_inds.astype(np.int)

    dense_n = 30 #int(args.predict_stepsize / args.output_stepsize)
    new_frame_inds = np.arange(2,len(frame_paths),FPS)
    #print("*** new_frame_inds",new_frame_inds) #  [  2  32  62  92 122 152 182 212 242 272 302 332 362 392 422 452 482 512]
    for index in new_frame_inds:
        print(frame_paths[index-1])
    frames = [
        cv2.imread(frame_paths[i - 1]) # frame_paths是30fps的
        for i in new_frame_inds
    ]
    print('Performing visualization')
    #print("*** results:",results)
    vis_frames = visualize(frames, results)

    try:
        import moviepy.editor as mpy
    except ImportError:
        raise ImportError('Please install moviepy to enable output file')

    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames],
                                fps=args.output_fps)
    vid.write_videofile(args.out_filename)

if __name__ == '__main__':
    args = parse_args() #解析参数 
    main(args)


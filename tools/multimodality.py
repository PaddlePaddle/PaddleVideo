# -*- coding: utf-8 -*-
import sys
import os
import datetime
import argparse
import json

import cv2
import operator
import numpy as np


from scipy.signal import argrelextrema
from paddleocr import PaddleOCR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paddlevideo.utils import get_config

ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # Paddle-ocr: a lightweight and efficient model

def LCstring(string1,string2):
    len1 = len(string1)
    len2 = len(string2)
    res = [[0 for i in range(len1+1)] for j in range(len2+1)]
    result = 0
    for i in range(1,len2+1):
        for j in range(1,len1+1):
            if string2[i-1] == string1[j-1]:
                res[i][j] = res[i-1][j-1]+1
                result = max(result,res[i][j])
    return result

#Class to hold information about each frame
class Frame:
    def __init__(self, id, frame, value):
        self.id = id
        self.frame = frame
        self.value = value

    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return self.id == other.id and self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)

def parse_args():
    parser = argparse.ArgumentParser("PaddleVideo multimodality script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/multimodality/modality.yaml',
                        help='config file path')
    parser.add_argument('-v',
                        '--video_path',
                        type=str,
                        default='data/xiaodu.mp4',
                        help='video file path')
    args = parser.parse_args()
    return args

def postprocess(results):
    """
    delete repeate content and reformat the results
    """
    last = []
    res = []
    if cfg._COR:
        return results
    else:
        for item in results:
            tmp = []
            data = item['content']
            for d in data:
                if d and d[1][1]>cfg.THRESH:
                    d.pop(0)
                    tmp.append(d)
                else:
                    continue
            # if tmp and LCstring(tmp[0][0][0],last) < 2:
            #     item['content'] = tmp
            #     res.append(item)
            #     last = tmp[0][0][0]
            # else:
            #     continue
    return res

def smooth(x, window_len=13, window='hanning'):
    """
    smooth the data with the given window size
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.") 

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[2 * x[0] - x[window_len:1:-1],
              x, 2 * x[-1] - x[-1:-window_len:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')

    return y[window_len - 1:-window_len + 1]

def ocr_im(name):
    img_path = cfg.dir+name
    #print(img_path)
    text = ocr.ocr(img_path, cls=True)
    return text

def save_results(results,time):
    f = open('./results.txt','w')
    f.write('*'*30+'OCR Results'+'*'*30+'\n')
    for item in results:
        f.write(json.dumps(str(item), ensure_ascii=False) + '\n')
    f.write('*'*50+'\n')
    f.write('Time Consume: '+str(time))
    f.close()

def video_ocr(frames,frame_diffs,fps):
    print("Extracting key frames and ocr, waiting...")
    num_frames = len(frames)
    total_time = num_frames/fps

    frame_indexes = []
    if cfg.USE_LOCAL_MAXIMA:
        diff_array = np.array(frame_diffs)
        sm_diff_array = smooth(diff_array, cfg.smooth_window_length) #smoothing the frame diff
        frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]  # return extrema frame index

    elif cfg.USE_ONE_SECOND:
        start = int(fps)//2
        frame_indexes = np.arange(start,len(frames),int(fps))

    print("Num of key frames: {}".format(len(frame_indexes)))

    total_results = []
    for i in frame_indexes:
        timestamp = round(total_time * i / float(num_frames), 1)
        name = "frame_" + str(frames[i - 1].id) + ".jpg"
        cv2.imwrite(cfg.dir + name, frames[i - 1].frame)  # save a frame temporaly

        text = ocr_im(name)
        # Check for repeated subtitles
        if text:
            tmp = {'timestamp': str(timestamp) + 's', 'content': text}
            total_results.append(tmp)
        os.remove(cfg.dir + name)
    res = postprocess(total_results)
    return res

def main():

    # check temporary frames dir
    if not os.path.exists(cfg.dir):
       os.mkdir(cfg.dir)

    print('*'*30+"Decording video waiting"+'*'*30)
    print("[video path] "+args.video_path)
    print("[config file] "+args.config)


    start=datetime.datetime.now()
    #decode video
    cap = cv2.VideoCapture(str(args.video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("[FPS] "+str(fps))

    curr_frame = None
    prev_frame = None

    frame_diffs = []
    frames = []
    ret, frame = cap.read()
    i = 1

    while (ret):
       luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
       curr_frame = luv
       if curr_frame is not None and prev_frame is not None:
            #logic here
           diff = cv2.absdiff(curr_frame, prev_frame)  # conculate difference between adjent frames
           count = np.sum(diff)
           frame_diffs.append(count)
           frame = Frame(i, frame, count)  # Instance a Frame class
           frames.append(frame)
       prev_frame = curr_frame
       i = i + 1
       ret, frame = cap.read()
    cap.release()

    #print('*'*30+"Finish video decord"+'*'*30)
    #print('Decord video time consuming: {}'.format(datetime.datetime.now()-start))
    
    print('='*30+'Start Video OCR '+'='*30)
    total_results = video_ocr(frames,frame_diffs,fps) #ocr a video
    end = datetime.datetime.now()
    save_results(total_results,end-start)

    os.removedirs(cfg.dir)

    if cfg.USE_VIDEO_TAG:
        print('='*30+'Start Video Tag '+'='*30)
        root = os.path.abspath(os.path.dirname(__file__))
        command = 'python '+root+'/VideoTag/videotag_test.py --filelist '+args.video_path
        os.system(command)
    print('='*30+'Finish All Process '+'='*30)
    print('TIME Consuming: ',(datetime.datetime.now()-start))

if __name__ == "__main__":
    args = parse_args()
    cfg = get_config(args.config,show=False)
    main()

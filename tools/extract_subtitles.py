# -*- coding: utf-8 -*-

import cv2
import operator
import numpy as np
import sys
import os
from scipy.signal import argrelextrema
from paddleocr import PaddleOCR
import datetime
import argparse

from paddlevideo.utils import get_config

ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # need to run only once to download and load model into memory


#Directory to store the processed frames
dir = "frames/"


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
                        default='configs/example.yaml',
                        help='config file path')
    parser.add_argument('-v',
                        '--video_path',
                        type=str,
                        default='data/xiaodu.mp4',
                        help='video file path')
    parser.add_argument('-w',
                        '--window_length',
                        type=int,
                        default=100,
                        help='smooth window length')
    args = parser.parse_args()
    return args

def postprocess(results):
    if _COR:
        return results
    else:
        for item in results:
            data = item['content']
            for res in data:
                if res:
                    res.pop(0)
                else:
                    continue
            item['content'] = data
    return results

def smooth(x, window_len=13, window='hanning'):
    print(len(x), window_len)
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
    #print(len(s))

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]

def rel_change(a, b):
   x = (b - a) / max(a, b)
   return x

def ocr_im(name):
    img_path = dir+name
    print('img_path',img_path) #改成帧
    text = ocr.ocr(img_path, cls=True)
    return text

def save_results(results,time):
    import json
    f = open('./results.txt','w')
    for item in results:
        f.write(json.dumps(str(item),ensure_ascii=False)+'\n')
    f.write('*'*50+'\n')
    f.write('Time Consume: '+str(time))
    f.close()

def video_ocr(frames,frame_diffs,fps):
    print("Extracting key frames, waiting...")
    print('-----len(frames)-----',len(frames))
    num_frames = len(frames)
    total_time = num_frames/fps

    last_subtitle = ""

    if USE_TOP_ORDER:
        # sort the list in descending order
        frames.sort(key=operator.attrgetter("value"), reverse=True)
        for keyframe in frames[:NUM_TOP_FRAMES]:
            name = "frame_" + str(keyframe.id) + ".jpg"
            cv2.imwrite(dir + "/" + name, keyframe.frame)

    if USE_THRESH:
        for i in range(1, len(frames)):
            if (rel_change(np.float(frames[i - 1].value), np.float(frames[i].value)) >= THRESH):
                # print("prev_frame:"+str(frames[i-1].value)+"  curr_frame:"+str(frames[i].value))
                name = "frame_" + str(frames[i].id) + ".jpg"
                cv2.imwrite(dir + "/" + name, frames[i].frame)

    if USE_LOCAL_MAXIMA:
        diff_array = np.array(frame_diffs)
        sm_diff_array = smooth(diff_array, len_window) #smoothing the frame diff
        frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]  # return extrema frame index
        total_results = []
        for i in frame_indexes:
            timestamp = round(total_time*i/float(num_frames),1)
            name = "frame_" + str(frames[i - 1].id) + ".jpg"
            cv2.imwrite(dir + name, frames[i - 1].frame)

            text = ocr_im(name)
            # Check for repeated subtitles
            if text and text != last_subtitle:
                last_subtitle = text
                tmp = {'timestamp':timestamp,'content':text}
                total_results.append(tmp)
            os.remove(dir + name)
        postprocess(total_results)
        return total_results

def main():
    #Print infos
    args = parse_args()
    cfg = get_config(args.config)
    print(cfg)

    import sys
    sys.exit(0)

    print("[Video Path] " + videopath)
    print("[Frame Directory] " + dir)
    print('*'*50+"Decording video waiting"+'*'*50)

    start=datetime.datetime.now()
    #decode video
    cap = cv2.VideoCapture(str(videopath))

    fps = cap.get(cv2.CAP_PROP_FPS)

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
            # logic here
            diff = cv2.absdiff(curr_frame, prev_frame)  # conculate difference between adjent frames
            count = np.sum(diff)
            frame_diffs.append(count)
            frame = Frame(i, frame, count)  # Instance a Frame class
            frames.append(frame)
        prev_frame = curr_frame
        i = i + 1
        ret, frame = cap.read()
    cap.release()

    print('*'*50+"Finish video decord"+'*'*50)
    print('Decord video time consuming: {}'.format(datetime.datetime.now()-start))

    total_results = video_ocr(frames,frame_diffs,fps) #ocr a video
    end = datetime.datetime.now()
    print('TIME Consuming: ',(end-start))
    save_results(total_results,end-start)

if __name__ == "__main__":
    main()
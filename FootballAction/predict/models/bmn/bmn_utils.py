"""
bmn common
"""
#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import os
import json
import multiprocessing as mp

import pandas as pd
import numpy as np
from paddle.fluid.initializer import Uniform


def boundary_choose(score_list):
    """
    boundary_choose
    :param score_list:
    :return:
    """
    max_score = max(score_list)
    mask_high = (score_list > max_score * 0.5)
    score_list = list(score_list)
    score_middle = np.array([0.0] + score_list + [0.0])
    score_front = np.array([0.0, 0.0] + score_list)
    score_back = np.array(score_list + [0.0, 0.0])
    mask_peak = ((score_middle > score_front) & (score_middle > score_back))
    mask_peak = mask_peak[1:-1]
    mask = (mask_high | mask_peak).astype('float32')
    return mask


def temporal_nms(df, alpha_low):
    """
    # One-dimensional non-maximal suppression
    #:param bboxes: [[st, ed, score], ...]
    #:param thresh:
    #:return:
    """

    thresh = alpha_low
    fps = 5
    topk = int(df.max()[1] / fps)  # 最长视频时长， fps = 5
    df_new = df[df["score"] > 0.01]
    df_new = df_new[df_new["xmax"] - df_new["xmin"] > 5]
    df_new = df_new.sort_values(by="score", ascending=False)
    tstart = np.array(df_new.xmin.values[:])
    tend = np.array(df_new.xmax.values[:])
    tscore = np.array(df_new.score.values[:])

    durations = tstart - tend + 1
    order = tscore.argsort()[::-1]

    keep = []
    while order.size > 0:
        import time
        start_time = time.time()
        i = order[0]
        keep.append(i)
        tt1 = np.maximum(tstart[i], tstart[order[1:]])
        tt2 = np.minimum(tend[i], tend[order[1:]])
        intersection = tt2 - tt1 + 1
        IoU = intersection / (durations[i] + durations[order[1:]] -
                              intersection).astype(float)

        inds = np.where(IoU <= thresh)[0]
        order = order[inds + 1]
        end_time = time.time()
    return [df_new[i] for i in keep]


def video_process_longvideo(video_list,
                            output_path,
                            snms_alpha=0.4,
                            snms_t1=0.55,
                            snms_t2=0.9):
    """
    video_process_longvideo
    :param video_list:
    :param output_path:
    :param snms_alpha:
    :param snms_t1:
    :param snms_t2:
    :return:
    """

    for video_name in video_list:
        df = pd.read_csv(os.path.join(output_path, video_name + ".csv"))
        if len(df) > 1:
            alpha_low_iou = 0.7
            df = temporal_nms(df, alpha_low_iou)

        proposal_list = []
        for idx in range(len(df)):
            tmp_prop = {"score": df.score.values[idx],\
                        "segment": [max(0, df.xmin.values[idx]),\
                                    df.xmax.values[idx]]}
            proposal_list.append(tmp_prop)
        result_dict[video_name] = proposal_list


def bmn_post_processing_video(video_list, subset, output_path, result_path):
    """
    bmn_post_processing_video
    :param video_list:
    :param subset:
    :param output_path:
    :param result_path:
    :return:
    """
    global result_dict
    result_dict = mp.Manager().dict()
    pp_num = 4

    num_videos = len(video_list)
    num_videos_per_thread = int(num_videos / pp_num)
    processes = []
    for tid in range(pp_num - 1):
        tmp_video_list = video_list[tid * num_videos_per_thread:(tid + 1) *
                                    num_videos_per_thread]
        p = mp.Process(target=video_process_longvideo,
                       args=(
                           tmp_video_list,
                           output_path,
                       ))
        p.start()
        processes.append(p)
    tmp_video_list = video_list[(pp_num - 1) * num_videos_per_thread:]
    p = mp.Process(target=video_process_longvideo,
                   args=(tmp_video_list, output_path))
    p.start()
    processes.append(p)
    for p in processes:
        p.join()

    result_dict = dict(result_dict)
    output_dict = {
        "version": "VERSION 1.3",
        "results": result_dict,
        "external_data": {}
    }
    outfile = open(os.path.join(result_path, "bmn_results_%s.json" % subset),
                   "w")

    json.dump(output_dict, outfile)
    outfile.close()


def gen_props_video(pred_bm, pred_start, pred_end, \
    max_window = 200, min_window = 5, output_path = "tmp/", video_name="test"):
    """
    gen_props_video
    """
    video_len = min(pred_bm.shape[-1],
                    min(pred_start.shape[-1], pred_end.shape[-1]))
    pred_bm = pred_bm[0, 0, :, :] * pred_bm[0, 1, :, :]
    start_mask = boundary_choose(pred_start)
    start_mask[0] = 1.
    end_mask = boundary_choose(pred_end)
    end_mask[-1] = 1.
    score_vector_list = []
    for idx in range(min_window, max_window):
        for jdx in range(video_len):
            start_index = jdx
            end_index = start_index + idx
            if end_index < video_len and start_mask[
                    start_index] == 1 and end_mask[end_index] == 1:
                xmin = start_index
                xmax = end_index
                xmin_score = pred_start[start_index]
                xmax_score = pred_end[end_index]
                bm_score = pred_bm[idx, jdx]
                conf_score = xmin_score * xmax_score * bm_score
                score_vector_list.append([xmin, xmax, conf_score])
    return_score = score_vector_list
    #score_vector_list = np.stack(score_vector_list)
    #cols = ["xmin", "xmax", "score"]
    #video_df = pd.DataFrame(score_vector_list, columns=cols)
    #video_df.to_csv(
    #    os.path.join(output_path, "%s.csv" % video_name), index=False)
    return return_score


def accumulate_infer_results_video(fetch_list):
    """
    accumulate_infer_results_video
    """
    pred_bm = np.array(fetch_list[0])
    pred_start = np.array(fetch_list[1][0])
    pred_end = np.array(fetch_list[2][0])
    result = gen_props_video(pred_bm, pred_start, pred_end)
    return result

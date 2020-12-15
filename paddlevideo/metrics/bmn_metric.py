# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from .registry import METRIC
from paddlevideo.utils import get_logger

import pandas as pd
import multiprocessing as mp
import json
import os
import numpy as np

logger = get_logger("paddlevideo")
""" BMNMetric for BMN.
"""


@METRIC.register
class BMNMetric(object):
    def __init__(self,
                 data_size,
                 batch_size,
                 world_size,
                 tscale,
                 dscale,
                 anno_file,
                 subset,
                 output_path,
                 result_path,
                 log_interval=1):
        """prepare for metrics
        """
        self.data_size = data_size
        self.batch_size = batch_size
        self.world_size = 1  # Now we just support single-card eval
        self.log_interval = log_interval

        self.tscale = tscale
        self.dscale = dscale
        self.anno_file = anno_file
        self.subset = subset
        self.output_path = output_path
        self.result_path = result_path

        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)
        if not os.path.isdir(self.result_path):
            os.makedirs(self.result_path)

        self.video_dict, self.video_list = self.get_dataset_dict(
            self.anno_file, self.subset)

    def get_dataset_dict(self, anno_file, subset):
        annos = json.load(open(anno_file))
        video_dict = {}
        for video_name in annos.keys():
            video_subset = annos[video_name]["subset"]
            if subset in video_subset:
                video_dict[video_name] = annos[video_name]
        video_list = list(video_dict.keys())
        video_list.sort()
        return video_dict, video_list

    def boundary_choose(self, score_list):
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

    def update(self, batch_id, data, outputs):
        """update metrics during each iter
        """
        fid = data[4].numpy()
        pred_bm, pred_start, pred_end = outputs

        snippet_xmins = [1.0 / self.tscale * i for i in range(self.tscale)]
        snippet_xmaxs = [
            1.0 / self.tscale * i for i in range(1, self.tscale + 1)
        ]
        cols = ["xmin", "xmax", "score"]

        video_name = self.video_list[fid]
        pred_bm = pred_bm[0, 0, :, :] * pred_bm[0, 1, :, :]
        start_mask = self.boundary_choose(pred_start)
        start_mask[0] = 1.
        end_mask = self.boundary_choose(pred_end)
        end_mask[-1] = 1.
        score_vector_list = []
        for idx in range(self.dscale):
            for jdx in range(self.tscale):
                start_index = jdx
                end_index = start_index + idx
                if end_index < self.tscale and start_mask[
                        start_index] == 1 and end_mask[end_index] == 1:
                    xmin = snippet_xmins[start_index]
                    xmax = snippet_xmaxs[end_index]
                    xmin_score = pred_start[start_index]
                    xmax_score = pred_end[end_index]
                    bm_score = pred_bm[idx, jdx]
                    conf_score = xmin_score * xmax_score * bm_score
                    score_vector_list.append([xmin, xmax, conf_score])

        score_vector_list = np.stack(score_vector_list)
        video_df = pd.DataFrame(score_vector_list, columns=cols)
        video_df.to_csv(os.path.join(self.output_path, "%s.csv" % video_name),
                        index=False)

        if batch_id % self.log_interval == 0:
            logger.info("Processing................ batch {}".format(batch_id))

    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        # check clip index of each video
        self.bmn_post_processing(self.video_dict, self.subset, self.output_path,
                                 self.result_path)
        logger.info("[EVAL] eval finished")

    def bmn_post_processing(self, video_dict, subset, output_path, result_path):
        video_list = list(video_dict.keys())
        global result_dict
        result_dict = mp.Manager().dict()
        pp_num = 12

        num_videos = len(video_list)
        num_videos_per_thread = int(num_videos / pp_num)
        processes = []
        for tid in range(pp_num - 1):
            tmp_video_list = video_list[tid * num_videos_per_thread:(tid + 1) *
                                        num_videos_per_thread]
            p = mp.Process(target=self.video_process,
                           args=(tmp_video_list, video_dict, output_path,
                                 result_dict))
            p.start()
            processes.append(p)
        tmp_video_list = video_list[(pp_num - 1) * num_videos_per_thread:]
        p = mp.Process(target=self.video_process,
                       args=(tmp_video_list, video_dict, output_path,
                             result_dict))
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
        outfile = open(
            os.path.join(result_path, "bmn_results_%s.json" % subset), "w")

        json.dump(output_dict, outfile)
        outfile.close()

    def video_process(self,
                      video_list,
                      video_dict,
                      output_path,
                      result_dict,
                      snms_alpha=0.4,
                      snms_t1=0.55,
                      snms_t2=0.9):

        for video_name in video_list:
            logger.info("Processing video........" + video_name)
            df = pd.read_csv(os.path.join(output_path, video_name + ".csv"))
            if len(df) > 1:
                df = self.soft_nms(df, snms_alpha, snms_t1, snms_t2)

            video_duration = video_dict[video_name]["duration_second"]
            proposal_list = []
            for idx in range(min(100, len(df))):
                tmp_prop={"score":df.score.values[idx], \
                          "segment":[max(0,df.xmin.values[idx])*video_duration, \
                                     min(1,df.xmax.values[idx])*video_duration]}
                proposal_list.append(tmp_prop)
            result_dict[video_name[2:]] = proposal_list

    def soft_nms(self, df, alpha, t1, t2):
        '''
        df: proposals generated by network;
        alpha: alpha value of Gaussian decaying function;
        t1, t2: threshold for soft nms.
        '''
        df = df.sort_values(by="score", ascending=False)
        tstart = list(df.xmin.values[:])
        tend = list(df.xmax.values[:])
        tscore = list(df.score.values[:])

        rstart = []
        rend = []
        rscore = []

        while len(tscore) > 1 and len(rscore) < 101:
            max_index = tscore.index(max(tscore))
            tmp_iou_list = self.iou_with_anchors(np.array(tstart),
                                                 np.array(tend),
                                                 tstart[max_index],
                                                 tend[max_index])
            for idx in range(0, len(tscore)):
                if idx != max_index:
                    tmp_iou = tmp_iou_list[idx]
                    tmp_width = tend[max_index] - tstart[max_index]
                    if tmp_iou > t1 + (t2 - t1) * tmp_width:
                        tscore[idx] = tscore[idx] * np.exp(
                            -np.square(tmp_iou) / alpha)

            rstart.append(tstart[max_index])
            rend.append(tend[max_index])
            rscore.append(tscore[max_index])
            tstart.pop(max_index)
            tend.pop(max_index)
            tscore.pop(max_index)

        newDf = pd.DataFrame()
        newDf['score'] = rscore
        newDf['xmin'] = rstart
        newDf['xmax'] = rend
        return newDf

    def iou_with_anchors(self, anchors_min, anchors_max, box_min, box_max):
        """Compute jaccard score between a box and the anchors.
        """
        len_anchors = anchors_max - anchors_min
        int_xmin = np.maximum(anchors_min, box_min)
        int_xmax = np.minimum(anchors_max, box_max)
        inter_len = np.maximum(int_xmax - int_xmin, 0.)
        union_len = len_anchors - inter_len + box_max - box_min
        jaccard = np.divide(inter_len, union_len)
        return jaccard

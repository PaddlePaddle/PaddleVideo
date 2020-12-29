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

import os
import json
import numpy as np
import urllib.request as urllib2
import pandas as pd
import multiprocessing as mp

from .registry import METRIC
from .base import BaseMetric
from paddlevideo.utils import get_logger

logger = get_logger("paddlevideo")


class ANETproposal(object):
    """
    This class is used for calculating AR@N and AUC;
    Code transfer from ActivityNet Gitub repository](https://github.com/activitynet/ActivityNet.git)
    """
    GROUND_TRUTH_FIELDS = ['database', 'taxonomy', 'version']
    PROPOSAL_FIELDS = ['results', 'version', 'external_data']
    API = 'http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/challenge19/api.py'

    def __init__(self,
                 ground_truth_filename=None,
                 proposal_filename=None,
                 ground_truth_fields=GROUND_TRUTH_FIELDS,
                 proposal_fields=PROPOSAL_FIELDS,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                 max_avg_nr_proposals=None,
                 subset='validation',
                 verbose=False,
                 check_status=True):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not proposal_filename:
            raise IOError('Please input a valid proposal file.')
        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.max_avg_nr_proposals = max_avg_nr_proposals
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = proposal_fields
        self.recall = None
        self.avg_recall = None
        self.proposals_per_video = None
        self.check_status = check_status
        # Retrieve blocked videos from server.
        if self.check_status:
            self.blocked_videos = self.get_blocked_videos()
        else:
            self.blocked_videos = list()
        # Import ground truth and proposals.
        self.ground_truth, self.activity_index = self._import_ground_truth(
            ground_truth_filename)
        self.proposal = self._import_proposal(proposal_filename)

        if self.verbose:
            print('[INIT] Loaded annotations from {} subset.'.format(subset))
            nr_gt = len(self.ground_truth)
            print('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.proposal)
            print('\tNumber of proposals: {}'.format(nr_pred))
            print('\tFixed threshold for tiou score: {}'.format(
                self.tiou_thresholds))

    def _import_ground_truth(self, ground_truth_filename):
        """
        Reads ground truth file, checks if it is well formatted, and returns
        the ground truth instances and the activity classes.

        Parameters:
        ground_truth_filename (str): full path to the ground truth json file.
        Returns:
        ground_truth (df): Data frame containing the ground truth instances.
        activity_index (dict): Dictionary containing class index.
        """
        with open(ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format
        if not all([field in data.keys() for field in self.gt_fields]):
            raise IOError('Please input a valid ground truth file.')

        # Read ground truth data.
        activity_index, cidx = {}, 0
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        for videoid, v in data['database'].items():
            if self.subset != v['subset']:
                continue
            if videoid in self.blocked_videos:
                continue
            for ann in v['annotations']:
                if ann['label'] not in activity_index:
                    activity_index[ann['label']] = cidx
                    cidx += 1
                video_lst.append(videoid)
                t_start_lst.append(float(ann['segment'][0]))
                t_end_lst.append(float(ann['segment'][1]))
                label_lst.append(activity_index[ann['label']])

        ground_truth = pd.DataFrame({
            'video-id': video_lst,
            't-start': t_start_lst,
            't-end': t_end_lst,
            'label': label_lst
        })
        return ground_truth, activity_index

    def _import_proposal(self, proposal_filename):
        """
        Reads proposal file, checks if it is well formatted, and returns
        the proposal instances.

        Parameters:
        proposal_filename (str): Full path to the proposal json file.
        Returns:
        proposal (df): Data frame containing the proposal instances.
        """
        with open(proposal_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format...
        if not all([field in data.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid proposal file.')

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        score_lst = []
        for videoid, v in data['results'].items():
            if videoid in self.blocked_videos:
                continue
            for result in v:
                video_lst.append(videoid)
                t_start_lst.append(float(result['segment'][0]))
                t_end_lst.append(float(result['segment'][1]))
                score_lst.append(result['score'])
        proposal = pd.DataFrame({
            'video-id': video_lst,
            't-start': t_start_lst,
            't-end': t_end_lst,
            'score': score_lst
        })
        return proposal

    def evaluate(self):
        """
        Evaluates a proposal file. To measure the performance of a
        method for the proposal task, we computes the area under the
        average recall vs average number of proposals per video curve.
        """
        recall, avg_recall, proposals_per_video = self.average_recall_vs_avg_nr_proposals(
            self.ground_truth,
            self.proposal,
            max_avg_nr_proposals=self.max_avg_nr_proposals,
            tiou_thresholds=self.tiou_thresholds)

        area_under_curve = np.trapz(avg_recall, proposals_per_video)

        if self.verbose:
            print('[RESULTS] Performance on ActivityNet proposal task.')
            print('\tArea Under the AR vs AN curve: {}%'.format(
                100. * float(area_under_curve) / proposals_per_video[-1]))

        self.recall = recall
        self.avg_recall = avg_recall
        self.proposals_per_video = proposals_per_video

    def average_recall_vs_avg_nr_proposals(self,
                                           ground_truth,
                                           proposals,
                                           max_avg_nr_proposals=None,
                                           tiou_thresholds=np.linspace(
                                               0.5, 0.95, 10)):
        """
        Computes the average recall given an average number of
        proposals per video.

        Parameters:
        ground_truth(df): Data frame containing the ground truth instances.
            Required fields: ['video-id', 't-start', 't-end']
        proposal(df): Data frame containing the proposal instances.
            Required fields: ['video-id, 't-start', 't-end', 'score']
        tiou_thresholds(1d-array | optional): array with tiou thresholds.

        Returns:
        recall(2d-array): recall[i,j] is recall at ith tiou threshold at the jth
            average number of average number of proposals per video.
        average_recall(1d-array): recall averaged over a list of tiou threshold.
            This is equivalent to recall.mean(axis=0).
        proposals_per_video(1d-array): average number of proposals per video.
        """

        # Get list of videos.
        video_lst = ground_truth['video-id'].unique()

        if not max_avg_nr_proposals:
            max_avg_nr_proposals = float(
                proposals.shape[0]) / video_lst.shape[0]

        ratio = max_avg_nr_proposals * float(
            video_lst.shape[0]) / proposals.shape[0]

        # Adaptation to query faster
        ground_truth_gbvn = ground_truth.groupby('video-id')
        proposals_gbvn = proposals.groupby('video-id')

        # For each video, computes tiou scores among the retrieved proposals.
        score_lst = []
        total_nr_proposals = 0
        for videoid in video_lst:
            # Get ground-truth instances associated to this video.
            ground_truth_videoid = ground_truth_gbvn.get_group(videoid)
            this_video_ground_truth = ground_truth_videoid.loc[:, [
                't-start', 't-end'
            ]].values

            # Get proposals for this video.
            try:
                proposals_videoid = proposals_gbvn.get_group(videoid)
            except:
                n = this_video_ground_truth.shape[0]
                score_lst.append(np.zeros((n, 1)))
                continue

            this_video_proposals = proposals_videoid.loc[:,
                                                         ['t-start', 't-end'
                                                          ]].values

            if this_video_proposals.shape[0] == 0:
                n = this_video_ground_truth.shape[0]
                score_lst.append(np.zeros((n, 1)))
                continue

            # Sort proposals by score.
            sort_idx = proposals_videoid['score'].argsort()[::-1]
            this_video_proposals = this_video_proposals[sort_idx, :]

            if this_video_proposals.ndim != 2:
                this_video_proposals = np.expand_dims(this_video_proposals,
                                                      axis=0)
            if this_video_ground_truth.ndim != 2:
                this_video_ground_truth = np.expand_dims(
                    this_video_ground_truth, axis=0)

            nr_proposals = np.minimum(
                int(this_video_proposals.shape[0] * ratio),
                this_video_proposals.shape[0])
            total_nr_proposals += nr_proposals
            this_video_proposals = this_video_proposals[:nr_proposals, :]

            # Compute tiou scores.
            tiou = self.wrapper_segment_iou(this_video_proposals,
                                            this_video_ground_truth)
            score_lst.append(tiou)

        # Given that the length of the videos is really varied, we
        # compute the number of proposals in terms of a ratio of the total
        # proposals retrieved, i.e. average recall at a percentage of proposals
        # retrieved per video.

        # Computes average recall.
        pcn_lst = np.arange(1, 101) / 100.0 * (max_avg_nr_proposals * float(
            video_lst.shape[0]) / total_nr_proposals)
        matches = np.empty((video_lst.shape[0], pcn_lst.shape[0]))
        positives = np.empty(video_lst.shape[0])
        recall = np.empty((tiou_thresholds.shape[0], pcn_lst.shape[0]))
        # Iterates over each tiou threshold.
        for ridx, tiou in enumerate(tiou_thresholds):

            # Inspect positives retrieved per video at different
            # number of proposals (percentage of the total retrieved).
            for i, score in enumerate(score_lst):
                # Total positives per video.
                positives[i] = score.shape[0]
                # Find proposals that satisfies minimum tiou threshold.
                true_positives_tiou = score >= tiou
                # Get number of proposals as a percentage of total retrieved.
                pcn_proposals = np.minimum(
                    (score.shape[1] * pcn_lst).astype(np.int), score.shape[1])

                for j, nr_proposals in enumerate(pcn_proposals):
                    # Compute the number of matches for each percentage of the proposals
                    matches[i, j] = np.count_nonzero(
                        (true_positives_tiou[:, :nr_proposals]).sum(axis=1))

            # Computes recall given the set of matches per video.
            recall[ridx, :] = matches.sum(axis=0) / positives.sum()

        # Recall is averaged.
        avg_recall = recall.mean(axis=0)

        # Get the average number of proposals per video.
        proposals_per_video = pcn_lst * (float(total_nr_proposals) /
                                         video_lst.shape[0])

        return recall, avg_recall, proposals_per_video

    def get_blocked_videos(self, api=API):
        api_url = '{}?action=get_blocked'.format(api)
        req = urllib2.Request(api_url)
        response = urllib2.urlopen(req)
        return json.loads(response.read())

    def wrapper_segment_iou(self, target_segments, candidate_segments):
        """
        Compute intersection over union btw segments
        Parameters:
        target_segments(nd-array): 2-dim array in format [m x 2:=[init, end]]
        candidate_segments(nd-array): 2-dim array in format [n x 2:=[init, end]]
        Returns:
        tiou(nd-array): 2-dim array [n x m] with IOU ratio.
        Note: It assumes that candidate-segments are more scarce that target-segments
        """
        if candidate_segments.ndim != 2 or target_segments.ndim != 2:
            raise ValueError('Dimension of arguments is incorrect')

        n, m = candidate_segments.shape[0], target_segments.shape[0]
        tiou = np.empty((n, m))
        for i in range(m):
            tiou[:, i] = self.segment_iou(target_segments[i, :],
                                          candidate_segments)

        return tiou

    def segment_iou(self, target_segment, candidate_segments):
        """
        Compute the temporal intersection over union between a
        target segment and all the test segments.

        Parameters:
        target_segment(1d-array): Temporal target segment containing [starting, ending] times.
        candidate_segments(2d-array): Temporal candidate segments containing N x [starting, ending] times.

        Returns:
        tiou(1d-array): Temporal intersection over union score of the N's candidate segments.
        """
        tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
        tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
        # Intersection including Non-negative overlap score.
        segments_intersection = (tt2 - tt1).clip(0)
        # Segment union.
        segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
                         + (target_segment[1] - target_segment[0]) - segments_intersection
        # Compute overlap as the ratio of the intersection
        # over union of two segments.
        tIoU = segments_intersection.astype(float) / segments_union
        return tIoU


@METRIC.register
class BMNMetric(BaseMetric):
    """
    Metrics for BMN. Two Stages in this metric:
    (1) Get test results using trained model, results will be saved in BMNMetric.result_path;
    (2) Calculate metrics using results file from stage (1).
    """
    def __init__(self,
                 data_size,
                 batch_size,
                 world_size,
                 tscale,
                 dscale,
                 anno_file,
                 ground_truth_filename,
                 subset,
                 output_path,
                 result_path,
                 get_metrics=True,
                 log_interval=1):
        """
        Init for BMN metrics.
        Params:
            get_metrics: whether to calculate AR@N and AUC metrics or not, default True.
        """
        super().__init__(data_size, batch_size, world_size, log_interval)
        assert self.batch_size == 1, " Now we just support batch_size==1 test"
        assert self.world_size == 1, " Now we just support single-card test"

        self.tscale = tscale
        self.dscale = dscale
        self.anno_file = anno_file
        self.ground_truth_filename = ground_truth_filename
        self.subset = subset
        self.output_path = output_path
        self.result_path = result_path
        self.get_metrics = get_metrics

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
        pred_bm = pred_bm.numpy()
        pred_start = pred_start[0].numpy()
        pred_end = pred_end[0].numpy()

        snippet_xmins = [1.0 / self.tscale * i for i in range(self.tscale)]
        snippet_xmaxs = [
            1.0 / self.tscale * i for i in range(1, self.tscale + 1)
        ]
        cols = ["xmin", "xmax", "score"]

        video_name = self.video_list[fid[0]]
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
        #Stage1
        self.bmn_post_processing(self.video_dict, self.subset, self.output_path,
                                 self.result_path)
        if self.get_metrics:
            logger.info("[TEST] calculate metrics...")
            #Stage2
            uniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid = self.cal_metrics(
                self.ground_truth_filename,
                os.path.join(self.result_path, "bmn_results_validation.json"),
                max_avg_nr_proposals=100,
                tiou_thresholds=np.linspace(0.5, 0.95, 10),
                subset='validation')
            logger.info("AR@1; AR@5; AR@10; AR@100")
            logger.info("%.02f %.02f %.02f %.02f" %
                        (100 * np.mean(uniform_recall_valid[:, 0]),
                         100 * np.mean(uniform_recall_valid[:, 4]),
                         100 * np.mean(uniform_recall_valid[:, 9]),
                         100 * np.mean(uniform_recall_valid[:, -1])))

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

    def cal_metrics(self,
                    ground_truth_filename,
                    proposal_filename,
                    max_avg_nr_proposals=100,
                    tiou_thresholds=np.linspace(0.5, 0.95, 10),
                    subset='validation'):

        anet_proposal = ANETproposal(ground_truth_filename,
                                     proposal_filename,
                                     tiou_thresholds=tiou_thresholds,
                                     max_avg_nr_proposals=max_avg_nr_proposals,
                                     subset=subset,
                                     verbose=True,
                                     check_status=False)
        anet_proposal.evaluate()
        recall = anet_proposal.recall
        average_recall = anet_proposal.avg_recall
        average_nr_proposals = anet_proposal.proposals_per_video

        return (average_nr_proposals, average_recall, recall)

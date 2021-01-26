# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
# limitations under the License.

import os.path as osp
import copy
import random
import numpy as np

from ..registry import DATASETS
from .base import BaseDataset
from ...utils import get_logger
from collections import defaultdict
logger = get_logger("paddlevideo")


@DATASETS.register()
class AVADataset(BaseDataset):
    """AVA dataset for spatial temporal detection.

    Based on official AVA annotation files, the dataset loads raw frames,
    bounding boxes, proposals and applies specified transformations to return
    a dict containing the frame tensors and other information.

    This datasets can load information from the following files:

    .. code-block:: txt

        ann_file -> ava_{train, val}_{v2.1, v2.2}.csv
        exclude_file -> ava_{train, val}_excluded_timestamps_{v2.1, v2.2}.csv
        label_file -> ava_action_list_{v2.1, v2.2}.pbtxt /
                      ava_action_list_{v2.1, v2.2}_for_activitynet_2019.pbtxt
        proposal_file -> ava_dense_proposals_{train, val}.FAIR.recall_93.9.pkl

    Particularly, the proposal_file is a pickle file which contains
    ``img_key`` (in format of ``{video_id},{timestamp}``). Example of a pickle
    file:

    .. code-block:: JSON

        {
            ...
            '0f39OWEqJ24,0902':
                array([[0.011   , 0.157   , 0.655   , 0.983   , 0.998163]]),
            '0f39OWEqJ24,0912':
                array([[0.054   , 0.088   , 0.91    , 0.998   , 0.068273],
                       [0.016   , 0.161   , 0.519   , 0.974   , 0.984025],
                       [0.493   , 0.283   , 0.981   , 0.984   , 0.983621]]),
            ...
        }

    Args:
        ann_file (str): Path to the annotation file like
            ``ava_{train, val}_{v2.1, v2.2}.csv``.
        exclude_file (str): Path to the excluded timestamp file like
            ``ava_{train, val}_excluded_timestamps_{v2.1, v2.2}.csv``.
        pipeline (list[dict | callable]): A sequence of data transforms.
        label_file (str): Path to the label file like
            ``ava_action_list_{v2.1, v2.2}.pbtxt`` or
            ``ava_action_list_{v2.1, v2.2}_for_activitynet_2019.pbtxt``.
            Default: None.
        suffix (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        proposal_file (str): Path to the proposal file like
            ``ava_dense_proposals_{train, val}.FAIR.recall_93.9.pkl``.
            Default: None.
        person_det_score_thr (float): The threshold of person detection scores,
            bboxes with scores above the threshold will be used. Default: 0.9.
            Note that 0 <= person_det_score_thr <= 1. If no proposal has
            detection score larger than the threshold, the one with the largest
            detection score will be used.
        num_classes (int): The number of classes of the dataset. Default: 81.
            (AVA has 80 action classes, another 1-dim is added for potential
            usage)
        data_prefix (str): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
                        Default: 'RGB'.
        num_max_proposals (int): Max proposals number to store. Default: 1000.
        timestamp_start (int): The start point of included timestamps. The
            default value is referred from the official website. Default: 902.
        timestamp_end (int): The end point of included timestamps. The
            default value is referred from the official website. Default: 1798.
    """

    _FPS = 30

    def __init__(self,
                 pipeline,
                 ann_file=None,
                 exclude_file=None,
                 label_file=None,
                 suffix='img_{:05}.jpg',
                 proposal_file=None,
                 person_det_score_thr=0.9,
                 num_classes=81,
                 data_prefix=None,
                 test_mode=False,
                 num_max_proposals=1000,
                 timestamp_start=900,
                 timestamp_end=1800):
        # since it inherits from `BaseDataset`, some arguments
        # should be assigned before performing `load_annotations()`
        self.exclude_file = exclude_file
        self.label_file = label_file
        self.proposal_file = proposal_file
        assert 0 <= person_det_score_thr <= 1, (
            'The value of '
            'person_det_score_thr should in [0, 1]. ')
        self.person_det_score_thr = person_det_score_thr
        self.num_classes = num_classes
        self.suffix = suffix
        self.num_max_proposals = num_max_proposals
        self.timestamp_start = timestamp_start
        self.timestamp_end = timestamp_end
        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,)

        if self.proposal_file is not None:
            self.proposals = self._load(self.proposal_file)
        else:
            self.proposals = None

        if not test_mode:
            valid_indexes = self.filter_exclude_file()
            self.logger.info(
                f'{len(valid_indexes)} out of {len(self.info)} '
                f'frames are valid.')
            self.info = self.info = [
                self.info[i] for i in valid_indexes
            ]

    def _load(self,path):
        import pickle
        f = open(path,'rb')
        res = pickle.load(f)
        f.close()
        return res

    def parse_img_record(self, img_records):
        bboxes, labels, entity_ids = [], [], []
        while len(img_records) > 0:
            img_record = img_records[0]
            num_img_records = len(img_records)
            selected_records = list(
                filter(
                    lambda x: np.array_equal(x['entity_box'], img_record[
                        'entity_box']), img_records))
            num_selected_records = len(selected_records)
            img_records = list(
                filter(
                    lambda x: not np.array_equal(x['entity_box'], img_record[
                        'entity_box']), img_records))
            assert len(img_records) + num_selected_records == num_img_records

            bboxes.append(img_record['entity_box'])
            valid_labels = np.array([
                selected_record['label']
                for selected_record in selected_records
            ])

            # The format can be directly used by BCELossWithLogits
            label = np.zeros(self.num_classes, dtype=np.float32)
            label[valid_labels] = 1.

            labels.append(label)
            entity_ids.append(img_record['entity_id'])

        bboxes = np.stack(bboxes)
        labels = np.stack(labels)
        entity_ids = np.stack(entity_ids)
        return bboxes, labels, entity_ids

    def filter_exclude_file(self):
        valid_indexes = []
        if self.exclude_file is None:
            valid_indexes = list(range(len(self.info)))
        else:
            exclude_video_infos = [
                x.strip().split(',') for x in open(self.exclude_file)
            ]
            for i, video_info in enumerate(self.info):
                valid_indexes.append(i)
                for video_id, timestamp in exclude_video_infos:
                    if (video_info['video_id'] == video_id
                            and video_info['timestamp'] == int(timestamp)):
                        valid_indexes.pop()
                        break
        return valid_indexes
    # def load_annotations(self):
    #     info = []
    #     records_dict_by_img = defaultdict(list)
    #     with open(self.ann_file, 'r') as fin:
    #         for line in fin:
    #             line_split = line.strip().split(',')
    #
    #             video_id = line_split[0]
    #             timestamp = int(line_split[1])
    #             img_key = f'{video_id},{timestamp:04d}'
    #
    #             entity_box = np.array(list(map(float, line_split[2:6])))
    #             label = int(line_split[6])
    #             entity_id = int(line_split[7])
    #             shot_info = (0, (self.timestamp_end - self.timestamp_start) *
    #                          self._FPS)
    #
    #             video_info = dict(
    #                 video_id=video_id,
    #                 timestamp=timestamp,
    #                 entity_box=entity_box,
    #                 label=label,
    #                 entity_id=entity_id,
    #                 shot_info=shot_info)
    #             records_dict_by_img[img_key].append(video_info)
    #
    #     for img_key in records_dict_by_img:
    #         video_id, timestamp = img_key.split(',')
    #         bboxes, labels, entity_ids = self.parse_img_record(
    #             records_dict_by_img[img_key])
    #         ann = dict(
    #             gt_bboxes=bboxes, gt_labels=labels, entity_ids=entity_ids)
    #         frame_dir = video_id
    #         if self.data_prefix is not None:
    #             frame_dir = osp.join(self.data_prefix, frame_dir)
    #         video_info = dict(
    #             frame_dir=frame_dir,
    #             video_id=video_id,
    #             timestamp=int(timestamp),
    #             img_key=img_key,
    #             shot_info=shot_info,
    #             fps=self._FPS,
    #             ann=ann)
    #         info.append(video_info)
    #
    #     return info
    #
    # def prepare_train_frames(self, idx):
    #     """Prepare the frames for training given the index."""
    #     results = copy.deepcopy(self.info[idx])
    #     img_key = results['img_key']
    #
    #     results['suffix'] = self.suffix
    #     results['modality'] = self.modality
    #     results['start_index'] = self.start_index
    #     results['timestamp_start'] = self.timestamp_start
    #     results['timestamp_end'] = self.timestamp_end
    #
    #     if self.proposals is not None:
    #         if img_key not in self.proposals:
    #             results['proposals'] = np.array([[0, 0, 1, 1]])
    #             results['scores'] = np.array([1])
    #         else:
    #             proposals = self.proposals[img_key]
    #             assert proposals.shape[-1] in [4, 5]
    #             if proposals.shape[-1] == 5:
    #                 thr = min(self.person_det_score_thr, max(proposals[:, 4]))
    #                 positive_inds = (proposals[:, 4] >= thr)
    #                 proposals = proposals[positive_inds]
    #                 proposals = proposals[:self.num_max_proposals]
    #                 results['proposals'] = proposals[:, :4]
    #                 results['scores'] = proposals[:, 4]
    #             else:
    #                 proposals = proposals[:self.num_max_proposals]
    #                 results['proposals'] = proposals
    #
    #     ann = results.pop('ann')
    #     results['gt_bboxes'] = ann['gt_bboxes']
    #     results['gt_labels'] = ann['gt_labels']
    #     results['entity_ids'] = ann['entity_ids']
    #
    #     return self.pipeline(results)
    #
    # def prepare_test_frames(self, idx):
    #     """Prepare the frames for testing given the index."""
    #     results = copy.deepcopy(self.info[idx])
    #     img_key = results['img_key']
    #
    #     results['suffix'] = self.suffix
    #     results['modality'] = self.modality
    #     results['start_index'] = self.start_index
    #     results['timestamp_start'] = self.timestamp_start
    #     results['timestamp_end'] = self.timestamp_end
    #
    #     if self.proposals is not None:
    #         if img_key not in self.proposals:
    #             results['proposals'] = np.array([[0, 0, 1, 1]])
    #             results['scores'] = np.array([1])
    #         else:
    #             proposals = self.proposals[img_key]
    #             assert proposals.shape[-1] in [4, 5]
    #             if proposals.shape[-1] == 5:
    #                 thr = min(self.person_det_score_thr, max(proposals[:, 4]))
    #                 positive_inds = (proposals[:, 4] >= thr)
    #                 proposals = proposals[positive_inds]
    #                 proposals = proposals[:self.num_max_proposals]
    #                 results['proposals'] = proposals[:, :4]
    #                 results['scores'] = proposals[:, 4]
    #             else:
    #                 proposals = proposals[:self.num_max_proposals]
    #                 results['proposals'] = proposals
    #
    #     ann = results.pop('ann')
    #     # Follow the mmdet variable naming style.
    #     results['gt_bboxes'] = ann['gt_bboxes']
    #     results['gt_labels'] = ann['gt_labels']
    #     results['entity_ids'] = ann['entity_ids']
    #
    #     return self.pipeline(results)
    def load_file(self):
        """Load index file to get video information."""
        info = []
        records_dict_by_img = defaultdict(list)
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split(',')

                video_id = line_split[0]
                timestamp = int(line_split[1])
                img_key = f'{video_id},{timestamp:04d}'

                entity_box = np.array(list(map(float, line_split[2:6])))
                label = int(line_split[6])
                entity_id = int(line_split[7])
                shot_info = (0, (self.timestamp_end - self.timestamp_start) *
                             self._FPS)

                video_info = dict(
                    video_id=video_id,
                    timestamp=timestamp,
                    entity_box=entity_box,
                    label=label,
                    entity_id=entity_id,
                    shot_info=shot_info)
                records_dict_by_img[img_key].append(video_info)

        for img_key in records_dict_by_img:
            video_id, timestamp = img_key.split(',')
            bboxes, labels, entity_ids = self.parse_img_record(
                records_dict_by_img[img_key])
            ann = dict(
                gt_bboxes=bboxes, gt_labels=labels, entity_ids=entity_ids)
            frame_dir = video_id
            if self.data_prefix is not None:
                frame_dir = osp.join(self.data_prefix, frame_dir)
            video_info = dict(
                frame_dir=frame_dir,
                video_id=video_id,
                timestamp=int(timestamp),
                img_key=img_key,
                shot_info=shot_info,
                fps=self._FPS,
                ann=ann)
            info.append(video_info)

        return info

    def prepare_train(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.info[idx])
        img_key = results['img_key']

        results['suffix'] = self.suffix
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results['timestamp_start'] = self.timestamp_start
        results['timestamp_end'] = self.timestamp_end

        if self.proposals is not None:
            if img_key not in self.proposals:
                results['proposals'] = np.array([[0, 0, 1, 1]])
                results['scores'] = np.array([1])
            else:
                proposals = self.proposals[img_key]
                assert proposals.shape[-1] in [4, 5]
                if proposals.shape[-1] == 5:
                    thr = min(self.person_det_score_thr, max(proposals[:, 4]))
                    positive_inds = (proposals[:, 4] >= thr)
                    proposals = proposals[positive_inds]
                    proposals = proposals[:self.num_max_proposals]
                    results['proposals'] = proposals[:, :4]
                    results['scores'] = proposals[:, 4]
                else:
                    proposals = proposals[:self.num_max_proposals]
                    results['proposals'] = proposals

        ann = results.pop('ann')
        results['gt_bboxes'] = ann['gt_bboxes']
        results['gt_labels'] = ann['gt_labels']
        results['entity_ids'] = ann['entity_ids']

        return self.pipeline(results)

    def prepare_test(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.info[idx])
        img_key = results['img_key']

        results['suffix'] = self.suffix
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results['timestamp_start'] = self.timestamp_start
        results['timestamp_end'] = self.timestamp_end

        if self.proposals is not None:
            if img_key not in self.proposals:
                results['proposals'] = np.array([[0, 0, 1, 1]])
                results['scores'] = np.array([1])
            else:
                proposals = self.proposals[img_key]
                assert proposals.shape[-1] in [4, 5]
                if proposals.shape[-1] == 5:
                    thr = min(self.person_det_score_thr, max(proposals[:, 4]))
                    positive_inds = (proposals[:, 4] >= thr)
                    proposals = proposals[positive_inds]
                    proposals = proposals[:self.num_max_proposals]
                    results['proposals'] = proposals[:, :4]
                    results['scores'] = proposals[:, 4]
                else:
                    proposals = proposals[:self.num_max_proposals]
                    results['proposals'] = proposals

        ann = results.pop('ann')
        # Follow the mmdet variable naming style.
        results['gt_bboxes'] = ann['gt_bboxes']
        results['gt_labels'] = ann['gt_labels']
        results['entity_ids'] = ann['entity_ids']

        return self.pipeline(results)
from __future__ import division
import json
import os
import shutil
import numpy as np
import paddle, cv2
from .base import BaseDataset
import json
from PIL import Image
from davisinteractive.utils.scribbles import scribbles2mask, annotated_frames
import sys
import copy
from ..registry import DATASETS

from ...utils import get_logger

logger = get_logger("paddlevideo")

import time


# stage1
@DATASETS.register()
class DAVIS2017_VOS_TrainDataset(BaseDataset):
    """DAVIS2017 dataset for training

    Return: imgs: N*2*3*H*W,label: N*2*1*H*W, seq-name: N, frame_num:N
    """
    def __init__(self,
                 file_path,
                 pipeline=None,
                 data_prefix=None,
                 test_mode=False,
                 split='train',
                 transform=None,
                 rgb=False):
        self.split = split
        self.rgb = rgb
        self.pipeline = transform
        self.seq_list_file = os.path.join(
            file_path, 'ImageSets', '2017',
            '_'.join(self.split) + '_instances.txt')
        super().__init__(file_path, pipeline, data_prefix, test_mode=test_mode)

    def load_file(self):
        info = []
        for splt in self.split:
            with open(
                    os.path.join(self.file_path, 'ImageSets', '2017',
                                 self.split + '.txt')) as f:
                seqs_tmp = f.readlines()
            seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
            info.extend(seqs_tmp)
        self.imglistdic = {}
        if not self._check_preprocess():
            self._preprocess()
        for seq_name in info:
            images = np.sort(
                os.listdir(
                    os.path.join(self.file_path, 'JPEGImages/480p/',
                                 seq_name.strip())))
            images_path = list(
                map(
                    lambda x: os.path.join('JPEGImages/480p/', seq_name.strip(),
                                           x), images))
            lab = np.sort(
                os.listdir(
                    os.path.join(self.file_path, 'Annotations/480p/',
                                 seq_name.strip())))
            lab_path = list(
                map(
                    lambda x: os.path.join('Annotations/480p/', seq_name.strip(
                    ), x), lab))
            self.imglistdic[seq_name] = (images, lab)
        return info

    def prepare_train(self, idx):
        seqname = self.info[idx]
        imagelist, lablist = self.imglistdic[seqname]
        prev_img = np.random.choice(imagelist[:-1], 1)
        prev_img = prev_img[0]
        frame_num = int(prev_img.split('.')[0]) + 1
        next_frame = str(frame_num)
        while len(next_frame) != 5:
            next_frame = '0' + next_frame

        ###############################Processing two adjacent frames and labels
        img2path = os.path.join('JPEGImages/480p/', seqname,
                                next_frame + '.' + prev_img.split('.')[-1])
        img2 = cv2.imread(os.path.join(self.file_path, img2path))
        img2 = np.array(img2, dtype=np.float32)

        imgpath = os.path.join('JPEGImages/480p/', seqname, prev_img)
        img1 = cv2.imread(os.path.join(self.file_path, imgpath))
        img1 = np.array(img1, dtype=np.float32)
        ###############
        labelpath = os.path.join(
            'Annotations/480p/', seqname,
            prev_img.split('.')[0] + '.' + lablist[0].split('.')[-1])
        label1 = Image.open(os.path.join(self.file_path, labelpath))
        label2path = os.path.join('Annotations/480p/', seqname,
                                  next_frame + '.' + lablist[0].split('.')[-1])
        label2 = Image.open(os.path.join(self.file_path, label2path))

        label1 = np.array(label1, dtype=np.uint8)
        label2 = np.array(label2, dtype=np.uint8)

        ###################
        ref_img = np.random.choice(imagelist, 1)
        ref_img = ref_img[0]
        ref_img_name = ref_img
        ref_scribble_label = Image.open(
            os.path.join(
                self.file_path, 'Annotations/480p/', seqname,
                ref_img_name.split('.')[0] + '.' + lablist[0].split('.')[-1]))
        ref_scribble_label = np.array(ref_scribble_label, dtype=np.uint8)

        while len(np.unique(ref_scribble_label)) < self.seq_dict[seqname][
                -1] + 1 or ref_img == prev_img or ref_img == (
                    next_frame + '.' + prev_img.split('.')[-1]):
            ref_img = np.random.choice(imagelist, 1)
            ref_img = ref_img[0]
            ref_img_name = ref_img
            ref_scribble_label = Image.open(
                os.path.join(
                    self.file_path, 'Annotations/480p/', seqname,
                    ref_img_name.split('.')[0] + '.' +
                    lablist[0].split('.')[-1]))
            ref_scribble_label = np.array(ref_scribble_label, dtype=np.int64)
        ref_img = os.path.join('JPEGImages/480p/', seqname, ref_img)
        ref_img = cv2.imread(os.path.join(self.file_path, ref_img))
        ref_img = np.array(ref_img, dtype=np.float32)
        ####
        ###################
        if self.rgb:
            img1 = img1[:, :, [2, 1, 0]]
            img2 = img2[:, :, [2, 1, 0]]
            ref_img = ref_img[:, :, [2, 1, 0]]
        obj_num = self.seq_dict[seqname][-1]

        sample = {
            'ref_img': ref_img,
            'img1': img1,
            'img2': img2,
            'ref_scribble_label': ref_scribble_label,
            'label1': label1,
            'label2': label2
        }

        sample['meta'] = {
            'seq_name': seqname,
            'frame_num': frame_num,
            'obj_num': obj_num
        }
        # self.pipeline=None
        if self.pipeline is not None:
            sample = self.pipeline(sample)
        sample['ref_scribble_label'] = paddle.to_tensor(
            sample['ref_scribble_label'], dtype='int64')
        sample['label1'] = paddle.to_tensor(sample['label1'], dtype='int64')
        sample['label2'] = paddle.to_tensor(sample['label2'], dtype='int64')
        return sample

    def prepare_test(self, idx):
        return self.prepare_test(idx)

    ########################

    def _check_preprocess(self):
        _seq_list_file = self.seq_list_file
        if not os.path.isfile(_seq_list_file):
            return False
        else:
            self.seq_dict = json.load(open(self.seq_list_file, 'r'))
            return True

    def _preprocess(self):
        self.seq_dict = {}
        for seq in self.info:
            # Read object masks and get number of objects
            name_label = np.sort(
                os.listdir(
                    os.path.join(self.file_path, 'Annotations/480p/', seq)))
            label_path = os.path.join(self.file_path, 'Annotations/480p/', seq,
                                      name_label[0])
            _mask = np.array(Image.open(label_path))
            _mask_ids = np.unique(_mask)
            n_obj = _mask_ids[-1]

            self.seq_dict[seq] = list(range(1, n_obj + 1))

        with open(self.seq_list_file, 'w') as outfile:
            outfile.write('{{\n\t"{:s}": {:s}'.format(
                self.info[0], json.dumps(self.seq_dict[self.info[0]])))
            for ii in range(1, len(self.info)):
                outfile.write(',\n\t"{:s}": {:s}'.format(
                    self.info[ii], json.dumps(self.seq_dict[self.info[ii]])))
            outfile.write('\n}\n')

        print('Preprocessing finished')


# stage2
@DATASETS.register()
class DAVIS2017_TrainDataset(BaseDataset):
    """DAVIS2017 dataset for training

    Return: imgs: N*2*3*H*W,label: N*2*1*H*W, seq-name: N, frame_num:N
    """
    def __init__(self,
                 file_path,
                 pipeline=None,
                 data_prefix=None,
                 test_mode=False,
                 split='train',
                 transform=None,
                 rgb=False):
        self.split = split
        self.rgb = rgb
        self.pipeline = transform
        self.seq_list_file = os.path.join(
            file_path, 'ImageSets', '2017',
            '_'.join(self.split) + '_instances.txt')
        self.seqs = []
        for splt in self.split:
            with open(
                    os.path.join(file_path, 'ImageSets', '2017',
                                 self.split + '.txt')) as f:
                seqs_tmp = f.readlines()
            seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
            self.seqs.extend(seqs_tmp)

        if not self._check_preprocess():
            self._preprocess()
        super().__init__(file_path, pipeline, data_prefix, test_mode=test_mode)

    def load_file(self):
        info = []
        for seq_name in self.seqs:
            images = np.sort(
                os.listdir(
                    os.path.join(self.file_path, 'JPEGImages/480p/',
                                 seq_name.strip())))
            images_path = list(
                map(
                    lambda x: os.path.join('JPEGImages/480p/', seq_name.strip(),
                                           x), images))
            lab = np.sort(
                os.listdir(
                    os.path.join(self.file_path, 'Annotations/480p/',
                                 seq_name.strip())))
            lab_path = list(
                map(
                    lambda x: os.path.join('Annotations/480p/', seq_name.strip(
                    ), x), lab))

            for img_path, label_path in zip(images_path[:-1], lab_path[:-1]):
                tmp_dic = {
                    'img': img_path,
                    'label': label_path,
                    'seq_name': seq_name,
                    'frame_num': img_path.split('/')[-1].split('.')[0]
                }
                info.append(tmp_dic)
        return info

    def prepare_train(self, idx):
        tmp_sample = self.info[idx]
        imgpath = tmp_sample['img']
        labelpath = tmp_sample['label']
        seqname = tmp_sample['seq_name']
        frame_num = int(tmp_sample['frame_num']) + 1

        next_frame = str(frame_num)
        while len(next_frame) != 5:
            next_frame = '0' + next_frame
        ###############################Processing two adjacent frames and labels
        img2path = os.path.join('JPEGImages/480p/', seqname,
                                next_frame + '.' + imgpath.split('.')[-1])
        img2 = cv2.imread(os.path.join(self.file_path, img2path))
        img2 = np.array(img2, dtype=np.float32)

        img1 = cv2.imread(os.path.join(self.file_path, imgpath))
        img1 = np.array(img1, dtype=np.float32)
        ###############
        label1 = Image.open(os.path.join(self.file_path, labelpath))
        label2path = os.path.join('Annotations/480p/', seqname,
                                  next_frame + '.' + labelpath.split('.')[-1])
        label2 = Image.open(os.path.join(self.file_path, label2path))

        label1 = np.array(
            label1, dtype=np.int32
        )  # fixed, uint8->int32, because layers.stack does not support uint8
        label2 = np.array(
            label2, dtype=np.int32
        )  # fixed, uint8->int32, because layers.stack does not support uint8
        ###################
        ref_tmp_dic = self.ref_frame_dic[seqname]
        ref_img = ref_tmp_dic['ref_frame']
        ref_scribble_label = ref_tmp_dic['scribble_label']
        ref_img = cv2.imread(os.path.join(self.file_path, ref_img))
        ref_img = np.array(ref_img, dtype=np.float32)
        ref_frame_gt = ref_tmp_dic['ref_frame_gt']
        ref_frame_gt = Image.open(os.path.join(self.file_path, ref_frame_gt))
        ref_frame_gt = np.array(
            ref_frame_gt, dtype=np.int32
        )  # fixed, uint8->int32, because layers.stack does not support uint8
        ref_frame_num = ref_tmp_dic['ref_frame_num']

        ###################
        if self.rgb:
            img1 = img1[:, :, [2, 1, 0]]
            img2 = img2[:, :, [2, 1, 0]]
            ref_img = ref_img[:, :, [2, 1, 0]]
        obj_num = self.seq_dict[seqname][-1]
        sample = {
            'ref_img': ref_img,
            'img1': img1,
            'img2': img2,
            'ref_scribble_label': ref_scribble_label,
            'label1': label1,
            'label2': label2,
            'ref_frame_gt': ref_frame_gt
        }
        if 'prev_round_label' in ref_tmp_dic:
            prev_round_label = ref_tmp_dic['prev_round_label']
            prev_round_label = prev_round_label.squeeze()
            prev_round_label = prev_round_label.numpy()
            sample = {
                'ref_img': ref_img,
                'img1': img1,
                'img2': img2,
                'ref_scribble_label': ref_scribble_label,
                'label1': label1,
                'label2': label2,
                'ref_frame_gt': ref_frame_gt,
                'prev_round_label': prev_round_label
            }

        sample['meta'] = {
            'seq_name': seqname,
            'frame_num': frame_num,
            'obj_num': obj_num,
            'ref_frame_num': ref_frame_num
        }
        if self.pipeline is not None:
            sample = self.pipeline(sample)

        return sample

    def prepare_test(self, idx):
        return self.prepare_test(idx)

    def update_ref_frame_and_label(self,
                                   round_scribble=None,
                                   frame_num=None,
                                   prev_round_label_dic=None):
        ##########Update reference frame and scribbles
        for seq in self.seqs:
            scribble = round_scribble[seq]
            if frame_num is None:
                scr_frame = annotated_frames(scribble)[0]
            else:
                scr_frame = frame_num[seq]
                scr_frame = int(scr_frame)
            scr_f = str(scr_frame)
            while len(scr_f) != 5:
                scr_f = '0' + scr_f
            ref_frame_path = os.path.join('JPEGImages/480p', seq,
                                          scr_f + '.jpg')
            #######################
            ref_frame_gt = os.path.join('Annotations/480p/', seq,
                                        scr_f + '.png')
            #########################
            ref_tmp = cv2.imread(os.path.join(self.file_path, ref_frame_path))
            h_, w_ = ref_tmp.shape[:2]
            scribble_masks = scribbles2mask(scribble, (h_, w_))
            if frame_num is None:

                scribble_label = scribble_masks[scr_frame]
            else:
                scribble_label = scribble_masks[0]
            self.ref_frame_dic[seq] = {
                'ref_frame': ref_frame_path,
                'scribble_label': scribble_label,
                'ref_frame_gt': ref_frame_gt,
                'ref_frame_num': scr_frame
            }
            if prev_round_label_dic is not None:
                self.ref_frame_dic[seq] = {
                    'ref_frame': ref_frame_path,
                    'scribble_label': scribble_label,
                    'ref_frame_gt': ref_frame_gt,
                    'ref_frame_num': scr_frame,
                    'prev_round_label': prev_round_label_dic[seq]
                }

    def init_ref_frame_dic(self):
        self.ref_frame_dic = {}
        for seq in self.seqs:
            selected_json = np.random.choice(
                ['001.json', '002.json', '003.json'], 1)
            selected_json = selected_json[0]
            scribble = os.path.join(self.file_path, 'Scribbles', seq,
                                    selected_json)
            with open(scribble) as f:
                scribble = json.load(f)
                scr_frame = annotated_frames(scribble)[0]
                scr_f = str(scr_frame)
                while len(scr_f) != 5:
                    scr_f = '0' + scr_f

                ref_frame_path = os.path.join('JPEGImages/480p', seq,
                                              scr_f + '.jpg')
                ref_tmp = cv2.imread(
                    os.path.join(self.file_path, ref_frame_path))
                h_, w_ = ref_tmp.shape[:2]
                scribble_masks = scribbles2mask(scribble, (h_, w_))
                ########################
                ref_frame_gt = os.path.join('Annotations/480p/', seq,
                                            scr_f + '.png')
                scribble_label = scribble_masks[scr_frame]
                self.ref_frame_dic[seq] = {
                    'ref_frame': ref_frame_path,
                    'scribble_label': scribble_label,
                    'ref_frame_gt': ref_frame_gt,
                    'ref_frame_num': scr_frame
                }

    def _check_preprocess(self):
        _seq_list_file = self.seq_list_file
        if not os.path.isfile(_seq_list_file):
            return False
        else:
            self.seq_dict = json.load(open(self.seq_list_file, 'r'))
            return True

    def _preprocess(self):
        self.seq_dict = {}
        for seq in self.seqs:
            # Read object masks and get number of objects
            name_label = np.sort(
                os.listdir(
                    os.path.join(self.file_path, 'Annotations/480p/', seq)))
            label_path = os.path.join(self.file_path, 'Annotations/480p/', seq,
                                      name_label[0])
            _mask = np.array(Image.open(label_path))
            _mask_ids = np.unique(_mask)
            n_obj = _mask_ids[-1]

            self.seq_dict[seq] = list(range(1, n_obj + 1))

        with open(self.seq_list_file, 'w') as outfile:
            outfile.write('{{\n\t"{:s}": {:s}'.format(
                self.seqs[0], json.dumps(self.seq_dict[self.seqs[0]])))
            for ii in range(1, len(self.seqs)):
                outfile.write(',\n\t"{:s}": {:s}'.format(
                    self.seqs[ii], json.dumps(self.seq_dict[self.seqs[ii]])))
            outfile.write('\n}\n')

        print('Preprocessing finished')


# test
@DATASETS.register()
class DAVIS2017_Feature_ExtractDataset(BaseDataset):
    def __init__(self,
                 file_path,
                 pipeline=None,
                 data_prefix=None,
                 test_mode=False,
                 split='val',
                 rgb=False,
                 seq_name=None):
        self.split = split
        self.rgb = rgb
        self.seq_name = seq_name
        super().__init__(file_path, pipeline, data_prefix, test_mode=test_mode)

    def load_file(self):
        info = np.sort(
            os.listdir(
                os.path.join(self.file_path, 'JPEGImages/480p/',
                             str(self.seq_name))))
        return info

    def prepare_train(self, idx):
        img = self.info[idx]
        imgpath = os.path.join(self.file_path, 'JPEGImages/480p/',
                               str(self.seq_name), img)
        current_img = cv2.imread(imgpath)
        current_img = np.array(current_img, dtype=np.float32)
        h, w, _ = current_img.shape
        sample = {'img1': current_img}
        sample['meta'] = {
            'seq_name': self.seq_name,
            'h_w': (h, w),
            'img_path': imgpath
        }
        if self.pipeline is not None:
            sample = self.pipeline(sample)
        return sample

    def prepare_test(self, idx):
        return self.prepare_train(idx)

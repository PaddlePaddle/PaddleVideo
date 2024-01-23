from __future__ import division
import json
import os
import shutil
import numpy as np
import paddle, cv2
from random import choice
from paddle.io import Dataset
import json
from PIL import Image
from davisinteractive.utils.scribbles import scribbles2mask, annotated_frames
import sys

sys.path.append("..")
from config import cfg
import time


class DAVIS2017_Test_Manager():
    def __init__(self,
                 split='val',
                 root=cfg.DATA_ROOT,
                 transform=None,
                 rgb=False,
                 seq_name=None):
        self.split = split
        self.db_root_dir = root

        self.rgb = rgb
        self.transform = transform
        self.seq_name = seq_name

    def get_image(self, idx):
        frame_name = str(idx)
        while len(frame_name) != 5:
            frame_name = '0' + frame_name
        imgpath = os.path.join(self.db_root_dir, 'JPEGImages/480p/',
                               str(self.seq_name), frame_name + '.jpg')
        img = cv2.imread(imgpath)
        img = np.array(img, dtype=np.float32)
        sample = {'img': img}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class DAVIS2017_Feature_Extract(Dataset):
    def __init__(self,
                 split='val',
                 root=cfg.DATA_ROOT,
                 transform=None,
                 rgb=False,
                 seq_name=None):
        self.split = split
        self.db_root_dir = root

        self.rgb = rgb
        self.transform = transform
        self.seq_name = seq_name
        self.img_list = np.sort(
            os.listdir(
                os.path.join(self.db_root_dir, 'JPEGImages/480p/',
                             str(seq_name))))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        imgpath = os.path.join(self.db_root_dir, 'JPEGImages/480p/',
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
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class DAVIS2017_VOS_Test(Dataset):
    """
    """
    def __init__(self,
                 split='val',
                 root=cfg.DATA_ROOT,
                 transform=None,
                 rgb=False,
                 result_root=None,
                 seq_name=None):
        self.split = split
        self.db_root_dir = root
        self.result_root = result_root
        self.rgb = rgb
        self.transform = transform
        self.seq_name = seq_name
        self.seq_list_file = os.path.join(
            self.db_root_dir, 'ImageSets', '2017',
            '_'.join(self.split) + '_instances.txt')

        self.seqs = []
        for splt in self.split:
            with open(
                    os.path.join(self.db_root_dir, 'ImageSets', '2017',
                                 self.split + '.txt')) as f:
                seqs_tmp = f.readlines()
            seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
            self.seqs.extend(seqs_tmp)

        if not self._check_preprocess():
            self._preprocess()

        assert self.seq_name in self.seq_dict.keys(
        ), '{} not in {} set.'.format(self.seq_name, '_'.join(self.split))
        names_img = np.sort(
            os.listdir(
                os.path.join(self.db_root_dir, 'JPEGImages/480p/',
                             str(seq_name))))
        img_list = list(
            map(lambda x: os.path.join('JPEGImages/480p/', str(seq_name), x),
                names_img))
        name_label = np.sort(
            os.listdir(
                os.path.join(self.db_root_dir, 'Annotations/480p/',
                             str(seq_name))))
        labels = list(
            map(lambda x: os.path.join('Annotations/480p/', str(seq_name), x),
                name_label))

        if not os.path.isfile(
                os.path.join(self.result_root, seq_name, name_label[0])):
            if not os.path.exists(os.path.join(self.result_root, seq_name)):
                os.makedirs(os.path.join(self.result_root, seq_name))

                shutil.copy(
                    os.path.join(self.db_root_dir, labels[0]),
                    os.path.join(self.result_root, seq_name, name_label[0]))
            else:
                shutil.copy(
                    os.path.join(self.db_root_dir, labels[0]),
                    os.path.join(self.result_root, seq_name, name_label[0]))
        self.first_img = names_img[0]
        self.first_label = name_label[0]
        self.img_list = names_img[1:]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        img = self.img_list[idx]
        imgpath = os.path.join(self.db_root_dir, 'JPEGImages/480p/',
                               str(self.seq_name), img)

        num_frame = int(img.split('.')[0])
        ref_img = os.path.join(self.db_root_dir, 'JPEGImages/480p/',
                               str(self.seq_name), self.first_img)
        prev_frame = num_frame - 1
        prev_frame = str(prev_frame)
        while len(prev_frame) != 5:
            prev_frame = '0' + prev_frame
        prev_img = os.path.join(self.db_root_dir, 'JPEGImages/480p/',
                                str(self.seq_name),
                                prev_frame + '.' + img.split('.')[-1])

        current_img = cv2.imread(imgpath)
        current_img = np.array(current_img, dtype=np.float32)

        ref_img = cv2.imread(ref_img)
        ref_img = np.array(ref_img, dtype=np.float32)

        prev_img = cv2.imread(prev_img)
        prev_img = np.array(prev_img, dtype=np.float32)

        ref_label = os.path.join(self.db_root_dir, 'Annotations/480p/',
                                 str(self.seq_name), self.first_label)
        ref_label = Image.open(ref_label)
        ref_label = np.array(ref_label, dtype=np.uint8)

        prev_label = os.path.join(
            self.result_root, str(self.seq_name),
            prev_frame + '.' + self.first_label.split('.')[-1])
        prev_label = Image.open(prev_label)
        prev_label = np.array(prev_label, dtype=np.uint8)

        obj_num = self.seq_dict[self.seq_name][-1]
        sample = {
            'ref_img': ref_img,
            'prev_img': prev_img,
            'current_img': current_img,
            'ref_label': ref_label,
            'prev_label': prev_label
        }
        sample['meta'] = {
            'seq_name': self.seq_name,
            'frame_num': num_frame,
            'obj_num': obj_num,
            'current_name': img
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

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
                    os.path.join(self.db_root_dir, 'Annotations/480p/', seq)))
            label_path = os.path.join(self.db_root_dir, 'Annotations/480p/',
                                      seq, name_label[0])
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


class DAVIS2017_VOS_Train(Dataset):
    """DAVIS2017 dataset for training

    Return: imgs: N*2*3*H*W,label: N*2*1*H*W, seq-name: N, frame_num:N
    """
    def __init__(self,
                 split='train',
                 root=cfg.DATA_ROOT,
                 transform=None,
                 rgb=False):
        self.split = split
        self.db_root_dir = root
        self.rgb = rgb
        self.transform = transform
        self.seq_list_file = os.path.join(
            self.db_root_dir, 'ImageSets', '2017',
            '_'.join(self.split) + '_instances.txt')
        self.seqs = []
        for splt in self.split:
            with open(
                    os.path.join(self.db_root_dir, 'ImageSets', '2017',
                                 self.split + '.txt')) as f:
                seqs_tmp = f.readlines()
            seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
            self.seqs.extend(seqs_tmp)
        self.imglistdic = {}
        if not self._check_preprocess():
            self._preprocess()
        self.sample_list = []
        for seq_name in self.seqs:
            images = np.sort(
                os.listdir(
                    os.path.join(self.db_root_dir, 'JPEGImages/480p/',
                                 seq_name.strip())))
            images_path = list(
                map(
                    lambda x: os.path.join('JPEGImages/480p/', seq_name.strip(),
                                           x), images))
            lab = np.sort(
                os.listdir(
                    os.path.join(self.db_root_dir, 'Annotations/480p/',
                                 seq_name.strip())))
            lab_path = list(
                map(
                    lambda x: os.path.join('Annotations/480p/', seq_name.strip(
                    ), x), lab))
            self.imglistdic[seq_name] = (images, lab)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seqname = self.seqs[idx]
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
        img2 = cv2.imread(os.path.join(self.db_root_dir, img2path))
        img2 = np.array(img2, dtype=np.float32)

        imgpath = os.path.join('JPEGImages/480p/', seqname, prev_img)
        img1 = cv2.imread(os.path.join(self.db_root_dir, imgpath))
        img1 = np.array(img1, dtype=np.float32)
        ###############
        labelpath = os.path.join(
            'Annotations/480p/', seqname,
            prev_img.split('.')[0] + '.' + lablist[0].split('.')[-1])
        label1 = Image.open(os.path.join(self.db_root_dir, labelpath))
        label2path = os.path.join('Annotations/480p/', seqname,
                                  next_frame + '.' + lablist[0].split('.')[-1])
        label2 = Image.open(os.path.join(self.db_root_dir, label2path))

        label1 = np.array(label1, dtype=np.uint8)
        label2 = np.array(label2, dtype=np.uint8)

        ###################
        ref_img = np.random.choice(imagelist, 1)
        ref_img = ref_img[0]
        ref_img_name = ref_img
        ref_scribble_label = Image.open(
            os.path.join(
                self.db_root_dir, 'Annotations/480p/', seqname,
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
                    self.db_root_dir, 'Annotations/480p/', seqname,
                    ref_img_name.split('.')[0] + '.' +
                    lablist[0].split('.')[-1]))
            ref_scribble_label = np.array(ref_scribble_label, dtype=np.int64)
        ref_img = os.path.join('JPEGImages/480p/', seqname, ref_img)
        ref_img = cv2.imread(os.path.join(self.db_root_dir, ref_img))
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
        if self.transform is not None:
            sample = self.transform(sample)
        sample['ref_scribble_label'] = paddle.to_tensor(
            sample['ref_scribble_label'], dtype='int64')
        sample['label1'] = paddle.to_tensor(sample['label1'], dtype='int64')
        sample['label2'] = paddle.to_tensor(sample['label2'], dtype='int64')
        return sample

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
        for seq in self.seqs:
            # Read object masks and get number of objects
            name_label = np.sort(
                os.listdir(
                    os.path.join(self.db_root_dir, 'Annotations/480p/', seq)))
            label_path = os.path.join(self.db_root_dir, 'Annotations/480p/',
                                      seq, name_label[0])
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


class DAVIS2017_Train(Dataset):
    """DAVIS2017 dataset for training

    Return: imgs: N*2*3*H*W,label: N*2*1*H*W, seq-name: N, frame_num:N
    """
    def __init__(self,
                 split='train',
                 root=cfg.DATA_ROOT,
                 transform=None,
                 rgb=False):
        self.split = split
        self.db_root_dir = root
        self.rgb = rgb
        self.transform = transform
        self.seq_list_file = os.path.join(
            self.db_root_dir, 'ImageSets', '2017',
            '_'.join(self.split) + '_instances.txt')
        self.seqs = []
        for splt in self.split:
            with open(
                    os.path.join(self.db_root_dir, 'ImageSets', '2017',
                                 self.split + '.txt')) as f:
                seqs_tmp = f.readlines()
            seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
            self.seqs.extend(seqs_tmp)

        if not self._check_preprocess():
            self._preprocess()
        self.sample_list = []
        for seq_name in self.seqs:
            images = np.sort(
                os.listdir(
                    os.path.join(self.db_root_dir, 'JPEGImages/480p/',
                                 seq_name.strip())))
            images_path = list(
                map(
                    lambda x: os.path.join('JPEGImages/480p/', seq_name.strip(),
                                           x), images))
            lab = np.sort(
                os.listdir(
                    os.path.join(self.db_root_dir, 'Annotations/480p/',
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
                self.sample_list.append(tmp_dic)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        tmp_sample = self.sample_list[idx]
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
        img2 = cv2.imread(os.path.join(self.db_root_dir, img2path))
        img2 = np.array(img2, dtype=np.float32)

        img1 = cv2.imread(os.path.join(self.db_root_dir, imgpath))
        img1 = np.array(img1, dtype=np.float32)
        ###############
        label1 = Image.open(os.path.join(self.db_root_dir, labelpath))
        label2path = os.path.join('Annotations/480p/', seqname,
                                  next_frame + '.' + labelpath.split('.')[-1])
        label2 = Image.open(os.path.join(self.db_root_dir, label2path))

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
        ref_img = cv2.imread(os.path.join(self.db_root_dir, ref_img))
        ref_img = np.array(ref_img, dtype=np.float32)
        ref_frame_gt = ref_tmp_dic['ref_frame_gt']
        ref_frame_gt = Image.open(os.path.join(self.db_root_dir, ref_frame_gt))
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
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

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
            ref_tmp = cv2.imread(os.path.join(self.db_root_dir, ref_frame_path))
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
        scribbles_path = os.path.join(self.db_root_dir, 'Scribbles')
        for seq in self.seqs:
            selected_json = np.random.choice(
                ['001.json', '002.json', '003.json'], 1)
            selected_json = selected_json[0]
            scribble = os.path.join(self.db_root_dir, 'Scribbles', seq,
                                    selected_json)
            with open(scribble) as f:
                scribble = json.load(f)
                #    print(scribble)
                scr_frame = annotated_frames(scribble)[0]
                scr_f = str(scr_frame)
                while len(scr_f) != 5:
                    scr_f = '0' + scr_f

                ref_frame_path = os.path.join('JPEGImages/480p', seq,
                                              scr_f + '.jpg')
                ref_tmp = cv2.imread(
                    os.path.join(self.db_root_dir, ref_frame_path))
                h_, w_ = ref_tmp.shape[:2]
                scribble_masks = scribbles2mask(scribble, (h_, w_))
                ########################
                ref_frame_gt = os.path.join('Annotations/480p/', seq,
                                            scr_f + '.png')
                ########################

                scribble_label = scribble_masks[scr_frame]
                self.ref_frame_dic[seq] = {
                    'ref_frame': ref_frame_path,
                    'scribble_label': scribble_label,
                    'ref_frame_gt': ref_frame_gt,
                    'ref_frame_num': scr_frame
                }

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
        for seq in self.seqs:
            # Read object masks and get number of objects
            name_label = np.sort(
                os.listdir(
                    os.path.join(self.db_root_dir, 'Annotations/480p/', seq)))
            label_path = os.path.join(self.db_root_dir, 'Annotations/480p/',
                                      seq, name_label[0])
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

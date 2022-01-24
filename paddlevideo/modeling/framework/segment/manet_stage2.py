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
# See the License for the specific language governing permissions and
# limitations under the License.
import json

from paddlevideo.loader import build_dataloader, build_dataset
from paddlevideo.loader.sampler import RandomIdentitySampler
from paddlevideo.loader.builder import build_pipeline
from paddlevideo.solver import build_lr, build_optimizer
from paddlevideo.utils import get_logger, load
import paddle.distributed as dist
import os.path as osp
from paddlevideo.utils import (build_record, log_batch, save, mkdir)
from paddlevideo.loader.pipelines import ToTensor_manet

import os
import time
import timeit
import davisinteractive.robot.interactive_robot as interactive_robot
import cv2
import numpy as np
import paddle
from PIL import Image
from davisinteractive.session import DavisInteractiveSession
from davisinteractive.utils.scribbles import scribbles2mask, annotated_frames
from paddle import nn

from paddlevideo.utils.config import AttrDict
from paddlevideo.utils.manet_utils import float_, _palette, damage_masks, int_, long_, label2colormap, mask_damager, \
    rough_ROI
from paddlevideo.utils import manet_utils
from .base import BaseSegment
from ...builder import build_model
from ...registry import SEGMENT


@SEGMENT.register()
class ManetSegment_Stage2(BaseSegment):
    def train_step(self,
                   weights=None,
                   parallel=False,
                   validate=True,
                   amp=False,
                   max_iters=80001,
                   use_fleet=False,
                   profiler_options=None,
                   train_cfg={},
                   **cfg):
        logger = get_logger("paddlevideo")
        batch_size = cfg['DATASET'].get('batch_size', 8)
        valid_batch_size = cfg['DATASET'].get('valid_batch_size', batch_size)

        use_gradient_accumulation = cfg.get('GRADIENT_ACCUMULATION', None)
        if use_gradient_accumulation and dist.get_world_size() >= 1:
            global_batch_size = cfg['GRADIENT_ACCUMULATION'].get(
                'global_batch_size', None)
            num_gpus = dist.get_world_size()

            assert isinstance(
                global_batch_size, int
            ), f"global_batch_size must be int, but got {type(global_batch_size)}"
            assert batch_size <= global_batch_size, f"global_batch_size must not be less than batch_size"

            cur_global_batch_size = batch_size * num_gpus  # The number of batches calculated by all GPUs at one time
            assert global_batch_size % cur_global_batch_size == 0, \
                f"The global batchsize must be divisible by cur_global_batch_size, but \
                        {global_batch_size} % {cur_global_batch_size} != 0"

            cfg['GRADIENT_ACCUMULATION'][
                "num_iters"] = global_batch_size // cur_global_batch_size
            # The number of iterations required to reach the global batchsize
            logger.info(
                f"Using gradient accumulation training strategy, "
                f"global_batch_size={global_batch_size}, "
                f"num_gpus={num_gpus}, "
                f"num_accumulative_iters={cfg['GRADIENT_ACCUMULATION'].num_iters}"
            )

        places = paddle.set_device('gpu')

        # default num worker: 0, which means no subprocess will be created
        num_workers = cfg['DATASET'].get('num_workers', 0)
        valid_num_workers = cfg['DATASET'].get('valid_num_workers', num_workers)
        model_name = cfg['model_name']
        output_dir = cfg.get("output_dir", f"./output/{model_name}")
        mkdir(output_dir)
        interactor = interactive_robot.InteractiveScribblesRobot()
        # 1. Construct model
        model = build_model(cfg['MODEL'])
        if parallel:
            model = paddle.DataParallel(model)

        if use_fleet:
            model = paddle.distributed_model(model)
        # 2. Construct dataset and sampler
        train_dataset = build_dataset(
            (cfg['DATASET']['train'], cfg['PIPELINE']['train']))
        step = 0
        round_ = 3
        epoch_per_round = 30

        # 3. Construct solver.
        lr = build_lr(cfg['OPTIMIZER']['learning_rate'])
        optimizer = build_optimizer(cfg['OPTIMIZER'], lr, model, parallel=True)
        resume_epoch = cfg.get("resume_epoch", 0)
        if resume_epoch:
            filename = osp.join(output_dir,
                                model_name + f"_step_{resume_epoch:05d}")
            resume_model_dict = manet_utils.load(filename + '.pdparams')
            resume_opt_dict = load(filename + '.pdopt')
            model.set_state_dict(resume_model_dict)
            optimizer.set_state_dict(resume_opt_dict)
        # Finetune:
        if weights:
            model_dict = manet_utils.load(weights, model.head.state_dict())
            model.head.set_state_dict(model_dict)
        # 4. Train Model
        ###AMP###
        record_list = build_record(cfg['MODEL'])
        while step < max_iters:
            if step < resume_epoch:
                logger.info(
                    f"| step: [{step + 1}] <= resume_epoch: [{resume_epoch}], continue... "
                )
                continue
            model.train()

            if step >= max_iters:
                break
            for r in range(round_):
                if r == 0:  #### r==0: Train the interaction branch in the first round
                    global_map_tmp_dic = {}
                    train_dataset.pipeline.pipelines[2].step = None
                    train_dataset.init_ref_frame_dic()

                train_dataloader_setting = dict(
                    batch_size=batch_size,
                    num_workers=num_workers,
                    collate_fn_cfg=cfg.get('MIX', None),
                    places=places,
                    sampler=RandomIdentitySampler(train_dataset),
                    dataset=train_dataset)
                train_loader = build_dataloader(train_dataloader_setting)

                for epoch in range(epoch_per_round):
                    tic = time.time()
                    for ii, sample in enumerate(train_loader):
                        record_list['reader_time'].update(time.time() - tic)
                        ref_imgs = sample['ref_img']  # batch_size * 3 * h * w
                        ref_scribble_labels = sample[
                            'ref_scribble_label']  # batch_size * 1 * h * w
                        seq_names = sample['meta']['seq_name']
                        obj_nums = sample['meta']['obj_num']
                        ref_frame_nums = sample['meta']['ref_frame_num']
                        ref_frame_gts = sample['ref_frame_gt']
                        bs, _, h, w = ref_imgs.shape
                        inputs = ref_imgs
                        with paddle.no_grad():
                            if parallel:
                                for c in model.children():
                                    c.head.feature_extracter.eval()
                                    c.head.semantic_embedding.eval()
                                    ref_frame_embedding = c.head.extract_feature(
                                        inputs)
                            else:
                                model.head.feature_extracter.eval()
                                model.head.semantic_embedding.eval()
                                ref_frame_embedding = model.head.extract_feature(
                                    inputs)
                        if r == 0:
                            first_inter = True
                            if parallel:
                                for c in model.children():
                                    tmp_dic = c.head.int_seghead(
                                        ref_frame_embedding=ref_frame_embedding,
                                        ref_scribble_label=ref_scribble_labels,
                                        prev_round_label=None,
                                        normalize_nearest_neighbor_distances=
                                        True,
                                        global_map_tmp_dic={},
                                        seq_names=seq_names,
                                        gt_ids=obj_nums,
                                        k_nearest_neighbors=train_cfg.knns,
                                        frame_num=ref_frame_nums,
                                        first_inter=first_inter)
                            else:
                                if parallel:
                                    for c in model.children():
                                        tmp_dic = c.head.int_seghead(
                                            ref_frame_embedding=
                                            ref_frame_embedding,
                                            ref_scribble_label=
                                            ref_scribble_labels,
                                            prev_round_label=None,
                                            normalize_nearest_neighbor_distances
                                            =True,
                                            global_map_tmp_dic={},
                                            seq_names=seq_names,
                                            gt_ids=obj_nums,
                                            k_nearest_neighbors=train_cfg.knns,
                                            frame_num=ref_frame_nums,
                                            first_inter=first_inter)
                                else:
                                    tmp_dic = model.head.int_seghead(
                                        ref_frame_embedding=ref_frame_embedding,
                                        ref_scribble_label=ref_scribble_labels,
                                        prev_round_label=None,
                                        normalize_nearest_neighbor_distances=
                                        True,
                                        global_map_tmp_dic={},
                                        seq_names=seq_names,
                                        gt_ids=obj_nums,
                                        k_nearest_neighbors=train_cfg.knns,
                                        frame_num=ref_frame_nums,
                                        first_inter=first_inter)
                        else:
                            first_inter = False
                            prev_round_label = sample['prev_round_label']
                            prev_round_label = prev_round_label
                            if parallel:
                                for c in model.children():
                                    tmp_dic = c.head.int_seghead(
                                        ref_frame_embedding=ref_frame_embedding,
                                        ref_scribble_label=ref_scribble_labels,
                                        prev_round_label=prev_round_label,
                                        normalize_nearest_neighbor_distances=
                                        True,
                                        global_map_tmp_dic={},
                                        seq_names=seq_names,
                                        gt_ids=obj_nums,
                                        k_nearest_neighbors=train_cfg.knns,
                                        frame_num=ref_frame_nums,
                                        first_inter=first_inter)
                            else:
                                tmp_dic = model.head.int_seghead(
                                    ref_frame_embedding=ref_frame_embedding,
                                    ref_scribble_label=ref_scribble_labels,
                                    prev_round_label=prev_round_label,
                                    normalize_nearest_neighbor_distances=True,
                                    global_map_tmp_dic={},
                                    seq_names=seq_names,
                                    gt_ids=obj_nums,
                                    k_nearest_neighbors=train_cfg.knns,
                                    frame_num=ref_frame_nums,
                                    first_inter=first_inter)
                        label_and_obj_dic = {}
                        for i, seq_ in enumerate(seq_names):
                            label_and_obj_dic[seq_] = (ref_frame_gts[i],
                                                       obj_nums[i])
                        label_dic = {}
                        for seq_ in tmp_dic.keys():
                            tmp_pred_logits = tmp_dic[seq_]
                            tmp_pred_logits = nn.functional.interpolate(
                                tmp_pred_logits,
                                size=(h, w),
                                mode='bilinear',
                                align_corners=True)
                            tmp_dic[seq_] = tmp_pred_logits

                            label_tmp, obj_num = label_and_obj_dic[seq_]
                            label_dic[seq_] = long_(label_tmp)
                        outputs = {}
                        if parallel:
                            for c in model.children():
                                outputs['loss'] = c.head.loss(
                                    dic_tmp=tmp_dic,
                                    label_dic=label_dic,
                                    step=step)
                        else:
                            outputs['loss'] = model.head.loss(
                                dic_tmp=tmp_dic, label_tmp=label_dic, step=step)
                        # 4.2 backward
                        if use_gradient_accumulation and i == 0:  # Use gradient accumulation strategy
                            optimizer.clear_grad()
                        avg_loss = outputs['loss'] / bs
                        avg_loss.backward()

                        # 4.3 minimize
                        if use_gradient_accumulation:  # Use gradient accumulation strategy
                            if (i + 1
                                ) % cfg['GRADIENT_ACCUMULATION'].num_iters == 0:
                                for p in model.parameters():
                                    if p.grad is not None:
                                        p.grad.set_value(
                                            p.grad /
                                            cfg['GRADIENT_ACCUMULATION'].
                                            num_iters)
                                optimizer.step()
                                optimizer.clear_grad()
                        else:  # Common case
                            optimizer.step()
                            optimizer.clear_grad()
                            # log record
                            record_list['lr'].update(optimizer.get_lr(),
                                                     batch_size)
                            for name, value in outputs.items():
                                record_list[name].update(value, batch_size)

                            record_list['batch_time'].update(time.time() - tic)
                            tic = time.time()

                            # learning rate iter step
                            if cfg['OPTIMIZER'].learning_rate.get("iter_step"):
                                lr.step()

                            # learning rate epoch step
                        if not cfg['OPTIMIZER'].learning_rate.get("iter_step"):
                            lr.step()
                        if step % cfg.get("log_interval", 10) == 0:
                            ips = "ips: {:.5f} instance/sec.".format(
                                batch_size / record_list["batch_time"].val)
                            log_batch(record_list,
                                      step,
                                      epoch + 1,
                                      _,
                                      "train",
                                      ips,
                                      cur_step=step,
                                      tot_step=max_iters)
                        if step % cfg['save_step'] == 0 and step != 0:
                            save(
                                optimizer.state_dict(),
                                osp.join(
                                    output_dir,
                                    model_name + f"_step_{step + 1:05d}.pdopt"))
                            save(
                                model.state_dict(),
                                osp.join(
                                    output_dir, model_name +
                                    f"_step_{step + 1:05d}.pdparams"))

                        step += 1

                print('trainset evaluating...')
                print('*' * 100)
                # validate

                if train_cfg.get('train_inter_use_true_result'):
                    if r != round_ - 1:
                        if r == 0:
                            prev_round_label_dic = {}
                        if parallel:
                            for c in model.children():
                                c.head.eval()
                        else:
                            model.head.eval()
                        with paddle.no_grad():
                            round_scribble = {}
                            frame_num_dic = {}
                            train_dataset.pipeline = build_pipeline(
                                cfg['PIPELINE']['valid']
                                ['train_inter_use_true_result'])
                            valid_dataloader_setting = dict(
                                batch_size=valid_batch_size,
                                num_workers=valid_num_workers,
                                places=places,
                                drop_last=False,
                                shuffle=cfg['DATASET'].get(
                                    'shuffle_valid', False),
                                dataset=train_dataset)
                            valid_loader = build_dataloader(
                                **valid_dataloader_setting)
                            for ii, sample in enumerate(valid_loader):
                                ref_imgs = sample[
                                    'ref_img']  # batch_size * 3 * h * w
                                img1s = sample['img1']
                                img2s = sample['img2']
                                ref_scribble_labels = sample[
                                    'ref_scribble_label']  # batch_size * 1 * h * w
                                label1s = sample['label1']
                                label2s = sample['label2']
                                seq_names = sample['meta']['seq_name']
                                obj_nums = sample['meta']['obj_num']
                                frame_nums = sample['meta']['frame_num']
                                bs, _, h, w = img2s.shape
                                inputs = paddle.concat((ref_imgs, img1s, img2s),
                                                       0)
                                if r == 0:
                                    ref_scribble_labels = rough_ROI(
                                        ref_scribble_labels)
                                # print(seq_names[0])
                                label1s_tocat = None
                                for i in range(bs):
                                    l = label1s[i]
                                    l = l.unsqueeze(0)
                                    l = mask_damager(l, 0.0)
                                    l = paddle.to_tensor(l)

                                    l = l.unsqueeze(0).unsqueeze(0)

                                    if label1s_tocat is None:
                                        label1s_tocat = float_(l)
                                    else:
                                        label1s_tocat = paddle.concat(
                                            (label1s_tocat, float_(l)), 0)

                                label1s = label1s_tocat
                                if parallel:
                                    for c in model.children():
                                        tmp_dic, global_map_tmp_dic = c.head(
                                            inputs,
                                            ref_scribble_labels,
                                            label1s,
                                            seq_names=seq_names,
                                            gt_ids=obj_nums,
                                            k_nearest_neighbors=train_cfg.knns,
                                            global_map_tmp_dic=
                                            global_map_tmp_dic,
                                            frame_num=frame_nums)
                                else:
                                    tmp_dic, global_map_tmp_dic = model.head(
                                        inputs,
                                        ref_scribble_labels,
                                        label1s,
                                        seq_names=seq_names,
                                        gt_ids=obj_nums,
                                        k_nearest_neighbors=train_cfg.knns,
                                        global_map_tmp_dic=global_map_tmp_dic,
                                        frame_num=frame_nums)
                                pred_label = tmp_dic[
                                    seq_names[0]].detach().cpu()
                                pred_label = nn.functional.interpolate(
                                    pred_label,
                                    size=(h, w),
                                    mode='bilinear',
                                    align_corners=True)
                                pred_label = paddle.argmax(pred_label, axis=1)
                                pred_label = pred_label.unsqueeze(0)
                                try:
                                    pred_label = damage_masks(pred_label)
                                except:
                                    pred_label = pred_label
                                pred_label = pred_label.squeeze(0)
                                round_scribble[
                                    seq_names[0]] = interactor.interact(
                                        seq_names[0], pred_label.numpy(),
                                        float_(label2s).squeeze(0).numpy(),
                                        obj_nums)
                                frame_num_dic[seq_names[0]] = frame_nums[0]
                                pred_label = pred_label.unsqueeze(0)
                                img_ww = Image.open(
                                    os.path.join(
                                        cfg['DATASET']['valid']['file_path'],
                                        'JPEGImages/480p/', seq_names[0],
                                        '00000.jpg'))
                                img_ww = np.array(img_ww)
                                or_h, or_w = img_ww.shape[:2]
                                pred_label = paddle.nn.functional.interpolate(
                                    float_(pred_label), (or_h, or_w),
                                    mode='nearest')
                                prev_round_label_dic[
                                    seq_names[0]] = pred_label.squeeze(0)
                        train_dataset.update_ref_frame_and_label(
                            round_scribble, frame_num_dic, prev_round_label_dic)

                    print(f'round {r}', 'trainset evaluating finished!')
                    print('*' * 100)
                    if parallel:
                        for c in model.children():
                            c.head.train()
                    else:
                        model.head.train()
                    train_dataset.pipeline = build_pipeline(
                        cfg['PIPELINE']['train'])
                    print('updating ref frame and label')
                else:
                    if r != round_ - 1:
                        round_scribble = {}

                        if r == 0:
                            prev_round_label_dic = {}
                        frame_num_dic = {}
                        train_dataset.pipeline = build_pipeline(
                            cfg['PIPELINE']['valid'])
                        valid_dataloader_setting = dict(
                            batch_size=valid_batch_size,
                            num_workers=valid_num_workers,
                            places=places,
                            drop_last=False,
                            shuffle=cfg['DATASET'].get('shuffle_valid', False),
                            dataset=train_dataset)

                        valid_loader = build_dataloader(
                            **valid_dataloader_setting)
                        if parallel:
                            for c in model.children():
                                c.head.eval()
                        else:
                            model.head.eval()
                        with paddle.no_grad():
                            for ii, sample in enumerate(valid_loader):
                                ref_imgs = sample[
                                    'ref_img']  # batch_size * 3 * h * w
                                img1s = sample['img1']
                                img2s = sample['img2']
                                ref_scribble_labels = sample[
                                    'ref_scribble_label']  # batch_size * 1 * h * w
                                label1s = sample['label1']
                                label2s = sample['label2']
                                seq_names = sample['meta']['seq_name']
                                obj_nums = sample['meta']['obj_num']
                                frame_nums = sample['meta']['frame_num']
                                bs, _, h, w = img2s.shape

                                print(seq_names[0])
                                label2s_ = mask_damager(label2s, 0.1)
                                round_scribble[
                                    seq_names[0]] = interactor.interact(
                                        seq_names[0],
                                        np.expand_dims(label2s_, axis=0),
                                        float_(label2s).squeeze(0).numpy(),
                                        obj_nums)
                                label2s__ = paddle.to_tensor(label2s_)

                                frame_num_dic[seq_names[0]] = frame_nums[0]
                                prev_round_label_dic[seq_names[0]] = label2s__

                        print(f'round {r}', 'trainset evaluating finished!')
                        print('*' * 100)
                        print('updating ref frame and label')

                        train_dataset.update_ref_frame_and_label(
                            round_scribble, frame_num_dic, prev_round_label_dic)
                        if parallel:
                            for c in model.children():
                                c.head.train()
                        else:
                            model.head.train()
                        train_dataset.pipeline = build_pipeline(
                            cfg['PIPELINE']['train'])
                        print('updating ref frame and label finished!')

    @paddle.no_grad()
    def test_step(self, weights, parallel=False, test_cfg={}, **cfg):
        # 1. Construct model.
        cfg['MODEL'].head.pretrained = ''
        cfg['MODEL'].head.test_mode = True
        model = build_model(cfg['MODEL'])
        if parallel:
            model = paddle.DataParallel(model)

        # 2. Construct dataset and sampler.
        cfg['DATASET'].test.test_mode = True
        total_frame_num_dic = {}
        #################
        seqs = []
        with open(
                os.path.join(cfg['DATASET'].test.file_path, 'ImageSets', '2017',
                             'val' + '.txt')) as f:
            seqs_tmp = f.readlines()
            seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
            seqs.extend(seqs_tmp)
        h_w_dic = {}
        for seq_name in seqs:
            images = np.sort(
                os.listdir(
                    os.path.join(cfg['DATASET'].test.file_path,
                                 'JPEGImages/480p/', seq_name.strip())))
            total_frame_num_dic[seq_name] = len(images)
            im_ = cv2.imread(
                os.path.join(cfg['DATASET'].test.file_path, 'JPEGImages/480p/',
                             seq_name, '00000.jpg'))
            im_ = np.array(im_, dtype=np.float32)
            hh_, ww_ = im_.shape[:2]
            h_w_dic[seq_name] = (hh_, ww_)
        _seq_list_file = os.path.join(cfg['DATASET'].test.file_path,
                                      'ImageSets', '2017',
                                      'v_a_l' + '_instances.txt')
        seq_dict = json.load(open(_seq_list_file, 'r'))
        ##################
        seq_imgnum_dict_ = {}
        seq_imgnum_dict = os.path.join(cfg['DATASET'].test.file_path,
                                       'ImageSets', '2017', 'val_imgnum.txt')
        if os.path.isfile(seq_imgnum_dict):

            seq_imgnum_dict_ = json.load(open(seq_imgnum_dict, 'r'))
        else:
            for seq in os.listdir(
                    os.path.join(cfg['DATASET'].test.file_path,
                                 'JPEGImages/480p/')):
                seq_imgnum_dict_[seq] = len(
                    os.listdir(
                        os.path.join(cfg['DATASET'].test.file_path,
                                     'JPEGImages/480p/', seq)))
            with open(seq_imgnum_dict, 'w') as f:
                json.dump(seq_imgnum_dict_, f)

        ##################

        is_save_image = cfg.get('is_save_image',
                                False)  # Save the predicted masks
        report_save_dir = cfg.get("output_dir", f"./output/{cfg['model_name']}")
        if not os.path.exists(report_save_dir):
            os.makedirs(report_save_dir)
        # Configuration used in the challenges
        max_nb_interactions = 8  # Maximum number of interactions
        max_time_per_interaction = 30  # Maximum time per interaction per object
        # Total time available to interact with a sequence and an initial set of scribbles
        max_time = max_nb_interactions * max_time_per_interaction  # Maximum time per object
        # Interactive parameters
        subset = 'val'
        host = 'localhost'  # 'localhost' for subsets train and val.
        model.eval()

        model_dict = manet_utils.load(weights, model.head.state_dict())
        model.head.set_state_dict(model_dict)
        inter_file = open(
            os.path.join(cfg.get("output_dir", f"./output/{cfg['model_name']}"),
                         'inter_file.txt'), 'w')
        seen_seq = []

        with DavisInteractiveSession(host=host,
                                     davis_root=cfg['DATASET'].test.file_path,
                                     subset=subset,
                                     report_save_dir=report_save_dir,
                                     max_nb_interactions=max_nb_interactions,
                                     max_time=max_time,
                                     metric_to_optimize='J') as sess:
            while sess.next():

                t_total = timeit.default_timer()
                # Get the current iteration scribbles

                sequence, scribbles, first_scribble = sess.get_scribbles(
                    only_last=True)  # return only one scribble (the last one)
                print(sequence)
                h, w = h_w_dic[sequence]
                if 'prev_label_storage' not in locals().keys():
                    prev_label_storage = paddle.zeros(
                        [104, h,
                         w])  # because the maximum length of frames is 104.
                if len(annotated_frames(scribbles)) == 0:
                    # if no scribbles return, keep masks in previous round
                    final_masks = prev_label_storage[:
                                                     seq_imgnum_dict_[sequence]]
                    sess.submit_masks(final_masks.numpy())
                    continue

                start_annotated_frame = annotated_frames(scribbles)[0]

                pred_masks = []
                pred_masks_reverse = []

                if first_scribble:  # If in the first round, initialize memories
                    n_interaction = 1
                    eval_global_map_tmp_dic = {}
                    local_map_dics = ({}, {})
                    total_frame_num = total_frame_num_dic[sequence]
                    obj_nums = seq_dict[sequence][-1]

                else:
                    n_interaction += 1
                    with open(f'{n_interaction}.json', 'w') as f:
                        json.dump(scribbles, f)
                inter_file.write(sequence + ' ' + 'interaction' +
                                 str(n_interaction) + ' ' + 'frame' +
                                 str(start_annotated_frame) + '\n')

                if first_scribble:  # if in the first round, extract pixel embbedings.
                    if sequence not in seen_seq:
                        inter_turn = 1

                        seen_seq.append(sequence)
                        embedding_memory = []
                        cfg_dataset = cfg['DATASET'].test
                        cfg_dataset.update({'seq_name': sequence})
                        test_dataset = build_dataset(
                            (cfg_dataset, cfg['PIPELINE'].test))
                        batch_size = cfg['DATASET'].get("test_batch_size", 14)
                        places = paddle.set_device('gpu')
                        # default num worker: 0, which means no subprocess will be created
                        num_workers = cfg['DATASET'].get('num_workers', 0)
                        num_workers = cfg['DATASET'].get(
                            'test_num_workers', num_workers)
                        test_dataloader_setting = dict(batch_size=batch_size,
                                                       num_workers=num_workers,
                                                       places=places,
                                                       drop_last=False,
                                                       shuffle=False,
                                                       dataset=test_dataset)
                        test_loader = build_dataloader(
                            **test_dataloader_setting)
                        for ii, sample in enumerate(test_loader):
                            imgs = sample['img1']
                            if parallel:
                                for c in model.children():
                                    frame_embedding = c.head.extract_feature(
                                        imgs)
                            else:
                                frame_embedding = model.head.extract_feature(
                                    imgs)
                            embedding_memory.append(frame_embedding)
                        del frame_embedding

                        embedding_memory = paddle.concat(embedding_memory, 0)
                        _, _, emb_h, emb_w = embedding_memory.shape
                        ref_frame_embedding = embedding_memory[
                            start_annotated_frame]
                        ref_frame_embedding = ref_frame_embedding.unsqueeze(0)
                    else:
                        inter_turn += 1
                        ref_frame_embedding = embedding_memory[
                            start_annotated_frame]
                        ref_frame_embedding = ref_frame_embedding.unsqueeze(0)

                else:
                    ref_frame_embedding = embedding_memory[
                        start_annotated_frame]
                    ref_frame_embedding = ref_frame_embedding.unsqueeze(0)
                ########
                scribble_masks = scribbles2mask(scribbles, (emb_h, emb_w))
                scribble_label = scribble_masks[start_annotated_frame]
                scribble_sample = {'scribble_label': scribble_label}
                scribble_sample = ToTensor_manet()(scribble_sample)
                #                     print(ref_frame_embedding, ref_frame_embedding.shape)
                scribble_label = scribble_sample['scribble_label']

                scribble_label = scribble_label.unsqueeze(0)
                model_name = cfg['model_name']
                output_dir = cfg.get("output_dir", f"./output/{model_name}")
                inter_file_path = os.path.join(
                    output_dir, sequence, 'interactive' + str(n_interaction),
                    'turn' + str(inter_turn))
                if is_save_image:
                    ref_scribble_to_show = scribble_label.squeeze().numpy()
                    im_ = Image.fromarray(
                        ref_scribble_to_show.astype('uint8')).convert('P', )
                    im_.putpalette(_palette)
                    ref_img_name = str(start_annotated_frame)

                    if not os.path.exists(inter_file_path):
                        os.makedirs(inter_file_path)
                    im_.save(
                        os.path.join(inter_file_path,
                                     'inter_' + ref_img_name + '.png'))
                if first_scribble:
                    prev_label = None
                    prev_label_storage = paddle.zeros(
                        [104, h,
                         w])  # because the maximum length of frames is 104.
                else:
                    prev_label = prev_label_storage[start_annotated_frame]
                    prev_label = prev_label.unsqueeze(0).unsqueeze(0)

                # check if no scribbles.
                if not first_scribble and paddle.unique(
                        scribble_label).shape[0] == 1:
                    print(
                        'not first_scribble and paddle.unique(scribble_label).shape[0] == 1'
                    )
                    print(paddle.unique(scribble_label))
                    final_masks = prev_label_storage[:
                                                     seq_imgnum_dict_[sequence]]
                    sess.submit_masks(final_masks.numpy())
                    continue

                ###inteaction segmentation head
                if parallel:
                    for c in model.children():
                        tmp_dic, local_map_dics = c.head.int_seghead(
                            ref_frame_embedding=ref_frame_embedding,
                            ref_scribble_label=scribble_label,
                            prev_round_label=prev_label,
                            global_map_tmp_dic=eval_global_map_tmp_dic,
                            local_map_dics=local_map_dics,
                            interaction_num=n_interaction,
                            seq_names=[sequence],
                            gt_ids=paddle.to_tensor([obj_nums]),
                            frame_num=[start_annotated_frame],
                            first_inter=first_scribble)
                else:
                    tmp_dic, local_map_dics = model.head.int_seghead(
                        ref_frame_embedding=ref_frame_embedding,
                        ref_scribble_label=scribble_label,
                        prev_round_label=prev_label,
                        global_map_tmp_dic=eval_global_map_tmp_dic,
                        local_map_dics=local_map_dics,
                        interaction_num=n_interaction,
                        seq_names=[sequence],
                        gt_ids=paddle.to_tensor([obj_nums]),
                        frame_num=[start_annotated_frame],
                        first_inter=first_scribble)
                pred_label = tmp_dic[sequence]
                pred_label = nn.functional.interpolate(pred_label,
                                                       size=(h, w),
                                                       mode='bilinear',
                                                       align_corners=True)
                pred_label = paddle.argmax(pred_label, axis=1)
                pred_masks.append(float_(pred_label))
                prev_label_storage[start_annotated_frame] = float_(
                    pred_label[0])

                if is_save_image:  # save image
                    pred_label_to_save = pred_label.squeeze(0).numpy()
                    im = Image.fromarray(
                        pred_label_to_save.astype('uint8')).convert('P', )
                    im.putpalette(_palette)
                    imgname = str(start_annotated_frame)
                    while len(imgname) < 5:
                        imgname = '0' + imgname
                    if not os.path.exists(inter_file_path):
                        os.makedirs(inter_file_path)
                    im.save(os.path.join(inter_file_path, imgname + '.png'))
                #######################################
                if first_scribble:
                    scribble_label = rough_ROI(scribble_label)

                ##############################
                ref_prev_label = pred_label.unsqueeze(0)
                prev_label = pred_label.unsqueeze(0)
                prev_embedding = ref_frame_embedding
                # propagate from current_frame to the end.
                # Propagation ->
                for ii in range(start_annotated_frame + 1, total_frame_num):
                    current_embedding = embedding_memory[ii]
                    current_embedding = current_embedding.unsqueeze(0)
                    prev_label = prev_label
                    if parallel:
                        for c in model.children():
                            tmp_dic, eval_global_map_tmp_dic, local_map_dics = c.head.prop_seghead(
                                ref_frame_embedding,
                                prev_embedding,
                                current_embedding,
                                scribble_label,
                                prev_label,
                                normalize_nearest_neighbor_distances=True,
                                use_local_map=True,
                                seq_names=[sequence],
                                gt_ids=paddle.to_tensor([obj_nums]),
                                k_nearest_neighbors=test_cfg['knns'],
                                global_map_tmp_dic=eval_global_map_tmp_dic,
                                local_map_dics=local_map_dics,
                                interaction_num=n_interaction,
                                start_annotated_frame=start_annotated_frame,
                                frame_num=[ii],
                                dynamic_seghead=c.head.dynamic_seghead)
                    else:
                        tmp_dic, eval_global_map_tmp_dic, local_map_dics = model.head.prop_seghead(
                            ref_frame_embedding,
                            prev_embedding,
                            current_embedding,
                            scribble_label,
                            prev_label,
                            normalize_nearest_neighbor_distances=True,
                            use_local_map=True,
                            seq_names=[sequence],
                            gt_ids=paddle.to_tensor([obj_nums]),
                            k_nearest_neighbors=test_cfg['knns'],
                            global_map_tmp_dic=eval_global_map_tmp_dic,
                            local_map_dics=local_map_dics,
                            interaction_num=n_interaction,
                            start_annotated_frame=start_annotated_frame,
                            frame_num=[ii],
                            dynamic_seghead=model.head.dynamic_seghead)
                    pred_label = tmp_dic[sequence]
                    pred_label = nn.functional.interpolate(pred_label,
                                                           size=(h, w),
                                                           mode='bilinear',
                                                           align_corners=True)
                    pred_label = paddle.argmax(pred_label, axis=1)
                    pred_masks.append(float_(pred_label))
                    prev_label = pred_label.unsqueeze(0)
                    prev_embedding = current_embedding
                    prev_label_storage[ii] = float_(pred_label[0])
                    if is_save_image:
                        pred_label_to_save = pred_label.squeeze(0).numpy()
                        im = Image.fromarray(
                            pred_label_to_save.astype('uint8')).convert('P', )
                        im.putpalette(_palette)
                        imgname = str(ii)
                        while len(imgname) < 5:
                            imgname = '0' + imgname
                        if not os.path.exists(inter_file_path):
                            os.makedirs(inter_file_path)
                        im.save(os.path.join(inter_file_path, imgname + '.png'))
                #######################################
                prev_label = ref_prev_label
                prev_embedding = ref_frame_embedding
                #######
                # propagate from current_frame to the beginning.
                # Propagation <-
                for ii in range(start_annotated_frame):
                    current_frame_num = start_annotated_frame - 1 - ii
                    current_embedding = embedding_memory[current_frame_num]
                    current_embedding = current_embedding.unsqueeze(0)
                    prev_label = prev_label
                    if parallel:
                        for c in model.children():
                            tmp_dic, eval_global_map_tmp_dic, local_map_dics = c.head.prop_seghead(
                                ref_frame_embedding,
                                prev_embedding,
                                current_embedding,
                                scribble_label,
                                prev_label,
                                normalize_nearest_neighbor_distances=True,
                                use_local_map=True,
                                seq_names=[sequence],
                                gt_ids=paddle.to_tensor([obj_nums]),
                                k_nearest_neighbors=test_cfg['knns'],
                                global_map_tmp_dic=eval_global_map_tmp_dic,
                                local_map_dics=local_map_dics,
                                interaction_num=n_interaction,
                                start_annotated_frame=start_annotated_frame,
                                frame_num=[current_frame_num],
                                dynamic_seghead=c.head.dynamic_seghead)
                    else:
                        tmp_dic, eval_global_map_tmp_dic, local_map_dics = model.head.prop_seghead(
                            ref_frame_embedding,
                            prev_embedding,
                            current_embedding,
                            scribble_label,
                            prev_label,
                            normalize_nearest_neighbor_distances=True,
                            use_local_map=True,
                            seq_names=[sequence],
                            gt_ids=paddle.to_tensor([obj_nums]),
                            k_nearest_neighbors=test_cfg['knns'],
                            global_map_tmp_dic=eval_global_map_tmp_dic,
                            local_map_dics=local_map_dics,
                            interaction_num=n_interaction,
                            start_annotated_frame=start_annotated_frame,
                            frame_num=[current_frame_num],
                            dynamic_seghead=model.head.dynamic_seghead)
                    pred_label = tmp_dic[sequence]
                    pred_label = nn.functional.interpolate(pred_label,
                                                           size=(h, w),
                                                           mode='bilinear',
                                                           align_corners=True)

                    pred_label = paddle.argmax(pred_label, axis=1)
                    pred_masks_reverse.append(float_(pred_label))
                    prev_label = pred_label.unsqueeze(0)
                    prev_embedding = current_embedding
                    ####
                    prev_label_storage[current_frame_num] = float_(
                        pred_label[0])
                    ###
                    if is_save_image:
                        pred_label_to_save = pred_label.squeeze(0).numpy()
                        im = Image.fromarray(
                            pred_label_to_save.astype('uint8')).convert('P', )
                        im.putpalette(_palette)
                        imgname = str(current_frame_num)
                        while len(imgname) < 5:
                            imgname = '0' + imgname
                        if not os.path.exists(inter_file_path):
                            os.makedirs(inter_file_path)
                        im.save(os.path.join(inter_file_path, imgname + '.png'))
                pred_masks_reverse.reverse()
                pred_masks_reverse.extend(pred_masks)
                final_masks = paddle.concat(pred_masks_reverse, 0)
                sess.submit_masks(final_masks.numpy())

                if inter_turn == 3 and n_interaction == 8:
                    del eval_global_map_tmp_dic
                    del local_map_dics
                    del embedding_memory
                    del prev_label_storage
                t_end = timeit.default_timer()
                print('Total time for single interaction: ' +
                      str(t_end - t_total))

            report = sess.get_report()
            summary = sess.get_global_summary(
                save_file=os.path.join(report_save_dir, 'summary.json'))
        inter_file.close()

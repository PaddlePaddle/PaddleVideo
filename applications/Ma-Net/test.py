import cv2
import os
import json

import paddle
from PIL import Image
import timeit
import numpy as np
from paddle.vision import transforms

from dataloaders.davis_2017_f import DAVIS2017_Feature_Extract
import dataloaders.custom_transforms_f as tr
from davisinteractive.session import DavisInteractiveSession
from networks.deeplab import DeepLab
from networks.IntVOS import IntVOS
import time
from davisinteractive.utils.scribbles import scribbles2mask, annotated_frames
from config import cfg
from paddle import nn
from paddle.io import DataLoader

from utils.api import float_, byte_


@paddle.no_grad()
def main():
    paddle.set_device("gpu:0")
    total_frame_num_dic = {}
    #################
    seqs = []
    with open(os.path.join(cfg.DATA_ROOT, 'ImageSets', '2017',
                           'val' + '.txt')) as f:
        seqs_tmp = f.readlines()
        seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
        seqs.extend(seqs_tmp)
    h_w_dic = {}
    for seq_name in seqs:
        images = np.sort(
            os.listdir(
                os.path.join(cfg.DATA_ROOT, 'JPEGImages/480p/',
                             seq_name.strip())))
        total_frame_num_dic[seq_name] = len(images)
        im_ = cv2.imread(
            os.path.join(cfg.DATA_ROOT, 'JPEGImages/480p/', seq_name,
                         '00000.jpg'))
        im_ = np.array(im_, dtype=np.float32)
        hh_, ww_ = im_.shape[:2]
        h_w_dic[seq_name] = (hh_, ww_)
    _seq_list_file = os.path.join(cfg.DATA_ROOT, 'ImageSets', '2017',
                                  'v_a_l' + '_instances.txt')
    seq_dict = json.load(open(_seq_list_file, 'r'))
    ##################
    seq_imgnum_dict_ = {}
    seq_imgnum_dict = os.path.join(cfg.DATA_ROOT, 'ImageSets', '2017',
                                   'val_imgnum.txt')
    if os.path.isfile(seq_imgnum_dict):

        seq_imgnum_dict_ = json.load(open(seq_imgnum_dict, 'r'))
    else:
        for seq in os.listdir(os.path.join(cfg.DATA_ROOT, 'JPEGImages/480p/')):
            seq_imgnum_dict_[seq] = len(
                os.listdir(os.path.join(cfg.DATA_ROOT, 'JPEGImages/480p/',
                                        seq)))
        with open(seq_imgnum_dict, 'w') as f:
            json.dump(seq_imgnum_dict_, f)

    ##################

    is_save_image = False  # Save the predicted masks
    report_save_dir = cfg.RESULT_ROOT
    save_res_dir = cfg.SAVE_RESULT_DIR  # changed to path
    if not os.path.exists(cfg.RESULT_ROOT):
        os.makedirs(cfg.RESULT_ROOT)
        # Configuration used in the challenges
    max_nb_interactions = 8  # Maximum number of interactions
    max_time_per_interaction = 30  # Maximum time per interaction per object
    # Total time available to interact with a sequence and an initial set of scribbles
    max_time = max_nb_interactions * max_time_per_interaction  # Maximum time per object
    # Interactive parameters
    subset = 'val'
    host = 'localhost'  # 'localhost' for subsets train and val.

    feature_extracter = DeepLab(backbone='resnet', freeze_bn=False)
    model = IntVOS(cfg, feature_extracter)
    print('model loading...')

    saved_model_dict = save_res_dir
    pretrained_dict = paddle.load(saved_model_dict)
    load_network(model, pretrained_dict)

    print(f'model loading from {saved_model_dict} finished!')
    model.eval()
    inter_file = open(os.path.join(cfg.RESULT_ROOT, 'inter_file.txt'), 'w')
    resized_h, resized_w = 480, 854
    ###############################
    composed_transforms = transforms.Compose(
        [tr.Resize((resized_h, resized_w)),
         tr.ToTensor()])
    ###############################

    seen_seq = []
    n = 0
    max_n = 1
    with DavisInteractiveSession(host=host,
                                 davis_root=cfg.DATA_ROOT,
                                 subset=subset,
                                 report_save_dir=report_save_dir,
                                 max_nb_interactions=max_nb_interactions,
                                 max_time=max_time,
                                 metric_to_optimize='J') as sess:
        while sess.next():
            t_total = timeit.default_timer()
            # Get the current iteration scribbles

            sequence, scribbles, first_scribble = sess.get_scribbles(
                only_last=True)
            h, w = h_w_dic[sequence]
            if 'prev_label_storage' not in locals().keys():
                prev_label_storage = paddle.zeros(
                    [104, h, w])  # because the maximum length of frames is 104.
            print(sequence)
            h, w = h_w_dic[sequence]
            if len(
                    annotated_frames(scribbles)
            ) == 0:  # if no scribbles return, keep masks in previous round

                final_masks = prev_label_storage[:seq_imgnum_dict_[sequence]]
                sess.submit_masks(final_masks.numpy())
            else:

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
                ##
                inter_file.write(sequence + ' ' + 'interaction' +
                                 str(n_interaction) + ' ' + 'frame' +
                                 str(start_annotated_frame) + '\n')
                ##

                ##########################Reference image process

                if first_scribble:  # if in the first round, extract pixel embbedings.
                    if sequence not in seen_seq:
                        inter_turn = 1

                        seen_seq.append(sequence)
                        embedding_memory = []
                        test_dataset = DAVIS2017_Feature_Extract(
                            root=cfg.DATA_ROOT,
                            transform=composed_transforms,
                            seq_name=sequence)
                        testloader = DataLoader(test_dataset,
                                                batch_size=14,
                                                shuffle=False,
                                                num_workers=cfg.NUM_WORKER)
                        for ii, sample in enumerate(testloader):
                            imgs = sample['img1']
                            frame_embedding = model.extract_feature(imgs)
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
                scribble_sample = tr.ToTensor()(scribble_sample)
                #                     print(ref_frame_embedding, ref_frame_embedding.shape)
                scribble_label = scribble_sample['scribble_label']

                scribble_label = scribble_label.unsqueeze(0)

                ######
                if is_save_image:
                    ref_scribble_to_show = scribble_label.squeeze().numpy()
                    im_ = Image.fromarray(
                        ref_scribble_to_show.astype('uint8')).convert('P', )
                    im_.putpalette(_palette)
                    ref_img_name = str(start_annotated_frame)

                    if not os.path.exists(
                            os.path.join(cfg.RESULT_ROOT, sequence,
                                         'interactive' + str(n_interaction),
                                         'turn' + str(inter_turn))):
                        os.makedirs(
                            os.path.join(cfg.RESULT_ROOT, sequence,
                                         'interactive' + str(n_interaction),
                                         'turn' + str(inter_turn)))
                    im_.save(
                        os.path.join(cfg.RESULT_ROOT, sequence,
                                     'interactive' + str(n_interaction),
                                     'turn' + str(inter_turn),
                                     'inter_' + ref_img_name + '.png'))

                scribble_label = scribble_label

                #######
                if first_scribble:

                    prev_label = None
                    prev_label_storage = paddle.zeros([104, h, w])
                    prev_label_storage = prev_label_storage
                else:
                    prev_label = prev_label_storage[start_annotated_frame]
                    prev_label = prev_label.unsqueeze(0).unsqueeze(0)
                if not first_scribble and paddle.unique(
                        scribble_label).shape[0] == 1:
                    final_masks = prev_label_storage[:
                                                     seq_imgnum_dict_[sequence]]
                    sess.submit_masks(final_masks.numpy())

                else:  ###inteaction segmentation head
                    print('inteaction segmentation head')
                    tmp_dic, local_map_dics = model.int_seghead(
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
                        if not os.path.exists(
                                os.path.join(cfg.RESULT_ROOT, sequence,
                                             'interactive' + str(n_interaction),
                                             'turn' + str(inter_turn))):
                            os.makedirs(
                                os.path.join(cfg.RESULT_ROOT, sequence,
                                             'interactive' + str(n_interaction),
                                             'turn' + str(inter_turn)))
                        im.save(
                            os.path.join(cfg.RESULT_ROOT, sequence,
                                         'interactive' + str(n_interaction),
                                         'turn' + str(inter_turn),
                                         imgname + '.png'))
                    #######################################
                    if first_scribble:
                        scribble_label = rough_ROI(scribble_label)

                    ##############################
                    ref_prev_label = pred_label.unsqueeze(0)
                    prev_label = pred_label.unsqueeze(0)
                    prev_embedding = ref_frame_embedding
                    #### Propagation ->
                    for ii in range(start_annotated_frame + 1, total_frame_num):
                        current_embedding = embedding_memory[ii]
                        current_embedding = current_embedding.unsqueeze(0)
                        prev_label = prev_label
                        tmp_dic, eval_global_map_tmp_dic, local_map_dics = model.prop_seghead(
                            ref_frame_embedding,
                            prev_embedding,
                            current_embedding,
                            scribble_label,
                            prev_label,
                            normalize_nearest_neighbor_distances=True,
                            use_local_map=True,
                            seq_names=[sequence],
                            gt_ids=paddle.to_tensor([obj_nums]),
                            k_nearest_neighbors=cfg.KNNS,
                            global_map_tmp_dic=eval_global_map_tmp_dic,
                            local_map_dics=local_map_dics,
                            interaction_num=n_interaction,
                            start_annotated_frame=start_annotated_frame,
                            frame_num=[ii],
                            dynamic_seghead=model.dynamic_seghead)
                        pred_label = tmp_dic[sequence]

                        pred_label = nn.functional.interpolate(
                            pred_label,
                            size=(h, w),
                            mode='bilinear',
                            align_corners=True)

                        pred_label = paddle.argmax(pred_label, axis=1)
                        pred_masks.append(float_(pred_label))
                        prev_label = pred_label.unsqueeze(0)
                        prev_embedding = current_embedding
                        prev_label_storage[ii] = float_(pred_label[0])
                        ####
                        if is_save_image:
                            pred_label_to_save = pred_label.squeeze(0).numpy()
                            im = Image.fromarray(
                                pred_label_to_save.astype('uint8')).convert(
                                    'P', )
                            im.putpalette(_palette)
                            imgname = str(ii)
                            while len(imgname) < 5:
                                imgname = '0' + imgname
                            if not os.path.exists(
                                    os.path.join(
                                        cfg.RESULT_ROOT, sequence,
                                        'interactive' + str(n_interaction),
                                        'turn' + str(inter_turn))):
                                os.makedirs(
                                    os.path.join(
                                        cfg.RESULT_ROOT, sequence,
                                        'interactive' + str(n_interaction),
                                        'turn' + str(inter_turn)))
                            im.save(
                                os.path.join(cfg.RESULT_ROOT, sequence,
                                             'interactive' + str(n_interaction),
                                             'turn' + str(inter_turn),
                                             imgname + '.png'))
                    #######################################
                    prev_label = ref_prev_label
                    prev_embedding = ref_frame_embedding
                    #######
                    # Propagation <-
                    for ii in range(start_annotated_frame):
                        current_frame_num = start_annotated_frame - 1 - ii
                        current_embedding = embedding_memory[current_frame_num]
                        current_embedding = current_embedding.unsqueeze(0)
                        prev_label = prev_label
                        tmp_dic, eval_global_map_tmp_dic, local_map_dics = model.prop_seghead(
                            ref_frame_embedding,
                            prev_embedding,
                            current_embedding,
                            scribble_label,
                            prev_label,
                            normalize_nearest_neighbor_distances=True,
                            use_local_map=True,
                            seq_names=[sequence],
                            gt_ids=paddle.to_tensor([obj_nums]),
                            k_nearest_neighbors=cfg.KNNS,
                            global_map_tmp_dic=eval_global_map_tmp_dic,
                            local_map_dics=local_map_dics,
                            interaction_num=n_interaction,
                            start_annotated_frame=start_annotated_frame,
                            frame_num=[current_frame_num],
                            dynamic_seghead=model.dynamic_seghead)
                        pred_label = tmp_dic[sequence]
                        pred_label = nn.functional.interpolate(
                            pred_label,
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
                                pred_label_to_save.astype('uint8')).convert(
                                    'P', )
                            im.putpalette(_palette)
                            imgname = str(current_frame_num)
                            while len(imgname) < 5:
                                imgname = '0' + imgname
                            if not os.path.exists(
                                    os.path.join(
                                        cfg.RESULT_ROOT, sequence,
                                        'interactive' + str(n_interaction),
                                        'turn' + str(inter_turn))):
                                os.makedirs(
                                    os.path.join(
                                        cfg.RESULT_ROOT, sequence,
                                        'interactive' + str(n_interaction),
                                        'turn' + str(inter_turn)))
                            im.save(
                                os.path.join(cfg.RESULT_ROOT, sequence,
                                             'interactive' + str(n_interaction),
                                             'turn' + str(inter_turn),
                                             imgname + '.png'))
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
            print('Total time for single interaction: ' + str(t_end - t_total))
        report = sess.get_report()
        summary = sess.get_global_summary(
            save_file=os.path.join(report_save_dir, 'summary.json'))
    inter_file.close()


def rough_ROI(ref_scribble_labels):
    dist = 20
    b, _, h, w = ref_scribble_labels.shape
    filter_ = paddle.zeros_like(ref_scribble_labels)
    to_fill = paddle.zeros_like(ref_scribble_labels)
    for i in range(b):
        no_background = (ref_scribble_labels[i] != -1)
        no_background = no_background.squeeze(0)

        no_b = no_background.nonzero()
        (h_min, w_min) = paddle.min(no_b, 0)
        (h_max, w_max) = paddle.max(no_b, 0)

        filter_[i, 0,
                max(h_min - dist, 0):min(h_max + dist, h - 1),
                max(w_min - dist, 0):min(w_max + dist, w - 1)] = 1

    final_scribble_labels = paddle.where(byte_(filter_), ref_scribble_labels,
                                         to_fill)
    return final_scribble_labels


def load_network(net, pretrained_dict):
    model_dict = net.state_dict()
    # 1. filter out unnecessary keys
    f_pretrained_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict:
            f_pretrained_dict[k] = v
        else:
            print(k)

    print(len(model_dict.keys()), len(pretrained_dict.keys()))

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    net.set_state_dict(model_dict)


_palette = [
    0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128,
    128, 128, 128, 128, 64, 0, 0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0,
    128, 191, 0, 128, 64, 128, 128, 191, 128, 128, 0, 64, 0, 128, 64, 0, 0, 191,
    0, 128, 191, 0, 0, 64, 128, 128, 64, 128, 22, 22, 22, 23, 23, 23, 24, 24,
    24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30,
    31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35, 36, 36, 36, 37,
    37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43,
    43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49,
    50, 50, 50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 54, 54, 54, 55, 55, 55, 56,
    56, 56, 57, 57, 57, 58, 58, 58, 59, 59, 59, 60, 60, 60, 61, 61, 61, 62, 62,
    62, 63, 63, 63, 64, 64, 64, 65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68, 68,
    69, 69, 69, 70, 70, 70, 71, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 75,
    75, 75, 76, 76, 76, 77, 77, 77, 78, 78, 78, 79, 79, 79, 80, 80, 80, 81, 81,
    81, 82, 82, 82, 83, 83, 83, 84, 84, 84, 85, 85, 85, 86, 86, 86, 87, 87, 87,
    88, 88, 88, 89, 89, 89, 90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 94,
    94, 94, 95, 95, 95, 96, 96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100,
    100, 100, 101, 101, 101, 102, 102, 102, 103, 103, 103, 104, 104, 104, 105,
    105, 105, 106, 106, 106, 107, 107, 107, 108, 108, 108, 109, 109, 109, 110,
    110, 110, 111, 111, 111, 112, 112, 112, 113, 113, 113, 114, 114, 114, 115,
    115, 115, 116, 116, 116, 117, 117, 117, 118, 118, 118, 119, 119, 119, 120,
    120, 120, 121, 121, 121, 122, 122, 122, 123, 123, 123, 124, 124, 124, 125,
    125, 125, 126, 126, 126, 127, 127, 127, 128, 128, 128, 129, 129, 129, 130,
    130, 130, 131, 131, 131, 132, 132, 132, 133, 133, 133, 134, 134, 134, 135,
    135, 135, 136, 136, 136, 137, 137, 137, 138, 138, 138, 139, 139, 139, 140,
    140, 140, 141, 141, 141, 142, 142, 142, 143, 143, 143, 144, 144, 144, 145,
    145, 145, 146, 146, 146, 147, 147, 147, 148, 148, 148, 149, 149, 149, 150,
    150, 150, 151, 151, 151, 152, 152, 152, 153, 153, 153, 154, 154, 154, 155,
    155, 155, 156, 156, 156, 157, 157, 157, 158, 158, 158, 159, 159, 159, 160,
    160, 160, 161, 161, 161, 162, 162, 162, 163, 163, 163, 164, 164, 164, 165,
    165, 165, 166, 166, 166, 167, 167, 167, 168, 168, 168, 169, 169, 169, 170,
    170, 170, 171, 171, 171, 172, 172, 172, 173, 173, 173, 174, 174, 174, 175,
    175, 175, 176, 176, 176, 177, 177, 177, 178, 178, 178, 179, 179, 179, 180,
    180, 180, 181, 181, 181, 182, 182, 182, 183, 183, 183, 184, 184, 184, 185,
    185, 185, 186, 186, 186, 187, 187, 187, 188, 188, 188, 189, 189, 189, 190,
    190, 190, 191, 191, 191, 192, 192, 192, 193, 193, 193, 194, 194, 194, 195,
    195, 195, 196, 196, 196, 197, 197, 197, 198, 198, 198, 199, 199, 199, 200,
    200, 200, 201, 201, 201, 202, 202, 202, 203, 203, 203, 204, 204, 204, 205,
    205, 205, 206, 206, 206, 207, 207, 207, 208, 208, 208, 209, 209, 209, 210,
    210, 210, 211, 211, 211, 212, 212, 212, 213, 213, 213, 214, 214, 214, 215,
    215, 215, 216, 216, 216, 217, 217, 217, 218, 218, 218, 219, 219, 219, 220,
    220, 220, 221, 221, 221, 222, 222, 222, 223, 223, 223, 224, 224, 224, 225,
    225, 225, 226, 226, 226, 227, 227, 227, 228, 228, 228, 229, 229, 229, 230,
    230, 230, 231, 231, 231, 232, 232, 232, 233, 233, 233, 234, 234, 234, 235,
    235, 235, 236, 236, 236, 237, 237, 237, 238, 238, 238, 239, 239, 239, 240,
    240, 240, 241, 241, 241, 242, 242, 242, 243, 243, 243, 244, 244, 244, 245,
    245, 245, 246, 246, 246, 247, 247, 247, 248, 248, 248, 249, 249, 249, 250,
    250, 250, 251, 251, 251, 252, 252, 252, 253, 253, 253, 254, 254, 254, 255,
    255, 255
]

if __name__ == '__main__':
    main()

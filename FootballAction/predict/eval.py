"""
get instance for lstm
根据gts计算每个proposal_bmn的iou、ioa、label等信息
"""
import os
import sys
import json
import random
import pickle
import numpy as np

import logging
logger = logging.getLogger(__name__)

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

dataset = "/home/work/datasets/EuroCup2016"
label_index_file = 'configs/index_label_football_8.json'
label_files = {
    'train': 'label_cls8_train.json',
    'validation': 'label_cls8_val.json'
}

global fps, mode
label_index = json.load(open(label_index_file, 'rb'))


def load_gts():
    global fps
    gts_data = {'fps': 0, 'gts': {}}
    for item, value in label_files.items():
        label_file = os.path.join(dataset, value)
        gts = json.load(open(label_file, 'rb'))
        gts_data['fps'] = gts['fps']
        fps = gts['fps']
        for gt in gts['gts']:
            gt['mode'] = item
            basename = os.path.basename(gt['url']).split('.')[0]
            gts_data['gts'][basename] = gt
    return gts_data['gts']


def computeIoU(e1, e2):
    """
    clc iou and ioa
    """
    if not (e1['label'] == e2['label'] and e1['basename'] == e2['basename']):
        return 0.
    area1 = e1["end"] - e1["start"]
    area2 = e2["end"] - e2["start"]
    x1 = np.maximum(e1["start"], e2["start"])
    x2 = np.minimum(e1["end"], e2["end"])
    inter = np.maximum(0.0, x2 - x1)
    if mode == 'proposal':
        iou = 0.0 if (area1 + area2 -
                      inter) == 0 else inter * 1.0 / (area1 + area2 - inter)
    else:
        iou = 0.0 if area2 == 0 else inter * 1.0 / area2
    return iou


def convert_proposal(boxes, basename, score_threshold=0.01):
    boxes = sorted(boxes, key=lambda x: float(x['score']), reverse=True)
    res = []
    for box in boxes:
        if not float(box['score']) >= score_threshold:
            continue
        res.append({
            'basename': basename,
            'start': int(float(box['start']) / fps),
            'end': int(float(box['end']) / fps),
            'label': 0
        })
    return res


def convert_classify(boxes, basename, iou_threshold=0.3, score_threshold=0.3):
    boxes = sorted(boxes,
                   key=lambda x:
                   (float(x['classify_score']), float(x['iou_score'])),
                   reverse=True)

    def convert_time_to_frame(time_type):
        h, m, s = time_type.split(':')
        return int(h) * 3600 + int(m) * 60 + int(s)

    res = []
    for box in boxes:
        if not (box['iou_score'] >= iou_threshold
                and box['classify_score'] >= score_threshold):
            continue
        res.append({
            'basename': basename,
            'start': convert_time_to_frame(box['start_time']),
            'end': convert_time_to_frame(box['end_time']),
            'label': box['label_id']
        })
    return res


def convert_groundtruth(boxes, basename, phase=None):
    res = []
    for box in boxes:
        for item in box['label_ids']:
            label = 0 if phase == 'proposal' else item
            res.append({
                'basename': basename,
                'start': box['start_id'],
                'end': box['end_id'],
                'label': label
            })
    return res


def print_head(iou):
    logger.info("\nioa = {:.1f}".format(iou))
    res_str = ''
    for item in ['label_name']:
        res_str += '{:<12s}'.format(item)
    for item in [
            'label_id', 'precision', 'recall', 'hit_prop', 'num_prop',
            'hit_gts', 'num_gts'
    ]:
        res_str += '{:<10s}'.format(item)
    #logger.info(res_str)
    print(res_str)


def print_result(res_dict, label='avg'):
    if label == 'avg':
        res_str = '{:<22s}'.format(str(label))
    else:
        res_str = '{0:{2}<6s}{1:<10s}'.format(label_index[str(label)],
                                              str(label), chr(12288))

    for item in ['prec', 'recall']:
        res_str += '{:<10.4f}'.format(res_dict[item])
    for item in ['hit_prop', 'num_prop', 'hit_gts', 'num_gts']:
        res_str += '{:<10d}'.format(res_dict[item])
    #logger.info(res_str)
    print(res_str)


def evaluation(res_boxes, gts_boxes, label_range, iou_range):
    iou_map = [computeIoU(resId, gtsId) for resId in res_boxes \
                                        for gtsId in gts_boxes]
    iou_map = np.array(iou_map).reshape((len(res_boxes), len(gts_boxes)))
    hit_map_prop_total = np.max(iou_map, axis=1)
    hit_map_index_total = np.argmax(iou_map, axis=1)

    res_dict = ['hit_prop', 'num_prop', 'hit_gts', 'num_gts']

    for iou_threshold in iou_range:
        print_head(iou_threshold)

        iou_prop = np.array([k >= iou_threshold for k in hit_map_prop_total])
        average_results = {}
        for label_id in label_range:
            sub_results = {}
            label_prop = np.array([k['label'] == label_id for k in res_boxes])
            label_gts = np.array([k['label'] == label_id for k in gts_boxes])
            sub_results['num_prop'] = sum(label_prop)
            sub_results['num_gts'] = sum(label_gts)
            hit_prop_index = label_prop & iou_prop
            sub_results['hit_prop'] = sum(hit_prop_index)
            sub_results['hit_gts'] = len(
                set(hit_map_index_total[hit_prop_index]))

            sub_results['prec'] = 0.0 if sub_results['num_prop'] == 0 \
                                      else sub_results['hit_prop'] * 1.0 / sub_results['num_prop']
            sub_results['recall'] = 0.0 if sub_results['num_gts'] == 0 \
                                        else sub_results['hit_gts'] * 1.0 / sub_results['num_gts']
            print_result(sub_results, label=label_id)
            for item in res_dict:
                if not item in average_results:
                    average_results[item] = 0
                average_results[item] += sub_results[item]
        if len(label_range) == 1:  # proposal 不需要输出average值
            continue
        average_results['prec'] = 0.0 if average_results['num_prop'] == 0 \
                                      else average_results['hit_prop'] * 1.0 / average_results['num_prop']
        average_results['recall'] = 0.0 if average_results['num_gts'] == 0 \
                                        else average_results['hit_gts'] * 1.0 /  average_results['num_gts']
        print_result(average_results)


def get_eval_results(predicts, gts_data, phase):
    global mode
    mode = phase
    res_boxes = []
    gts_boxes = []
    for ped_data in predicts:
        basename = ped_data['video_name']
        gts = gts_data[basename]['actions']
        if phase == 'proposal':
            res_boxes.extend(convert_proposal(ped_data['bmn_results'],
                                              basename))
            gts_boxes.extend(
                convert_groundtruth(gts, basename, phase='proposal'))
            label_range = [0]
            iou_range = np.arange(0.1, 1, 0.1)
        else:
            res_boxes.extend(
                convert_classify(ped_data['action_results'], basename))
            gts_boxes.extend(convert_groundtruth(gts, basename))
            label_range = range(1, 8)
            iou_range = np.arange(0.5, 0.6, 0.1)

    evaluation(res_boxes, gts_boxes, label_range, iou_range)


if __name__ == "__main__":
    result_file = sys.argv[1]
    predicts = json.load(open(result_file, 'r', encoding='utf-8'))
    gts_data = load_gts()

    #get_eval_results(predicts, gts_data, 'proposal')
    get_eval_results(predicts, gts_data, 'actions')

"""
get instance for bmn
使用winds=8的滑窗，将所有子窗口的长度之和小于winds的进行合并
合并后，父窗口代表bmn训练数据，子窗口代表tsn训练数据
"""
import os
import sys
import json
import random
import pickle
import numpy as np
import math

# for table tennis
bmn_window = 8
dataset = "/home/aistudio/work/BMN/"
feat_dir = dataset + '/Features_example'
out_dir = dataset + '/Input_for_bmn'
label_files = {
    'train': 'label_cls14_small_train.json',
    'validation': 'label_cls14_small_test.json'
}

global fps


def gen_gts_for_bmn(gts_data):
    """
    @param, gts_data, original gts for action detection
    @return, gts_bmn, output gts dict for bmn
    """
    fps = gts_data['fps']
    gts_bmn = {'fps': fps, 'gts': []}
    for sub_item in gts_data['gts']:
        url = sub_item['url']

        max_length = sub_item['total_frames']

        gts_bmn['gts'].append({
            'url': url,
            'total_frames': max_length,
            'root_actions': []
        })
        sub_actions = sub_item['actions']
        # 跳过没有动作的片段
        if len(sub_actions) == 0:
            continue
        # duration > bmn_window， 动作持续时间大于bmn_windows，直接删除
        for idx, sub_action in enumerate(sub_actions):
            if sub_action['end_id'] - sub_action['start_id'] > bmn_window:
                sub_actions.pop(idx)

        # 【滑动窗口，把每一个视频里的动作片段提取出来】
        root_actions = [sub_actions[0]]
        # before_id, 前一动作的最后一帧
        # after_id, 后一动作的第一帧
        before_id = 0
        for idx in range(1, len(sub_actions)):
            cur_action = sub_actions[idx]
            duration = (cur_action['end_id'] - root_actions[0]['start_id'])
            if duration > bmn_window:  # windows只能包住一个动作就包，包不住就包多个
                after_id = cur_action['start_id']
                gts_bmn['gts'][-1]['root_actions'].append({
                    'before_id':
                    before_id,
                    'after_id':
                    after_id,
                    'actions':
                    root_actions
                })
                before_id = root_actions[-1]['end_id']  #更新滑窗
                root_actions = [cur_action]
            else:
                root_actions.append(cur_action)
            if idx == len(sub_actions) - 1:
                after_id = max_length
                gts_bmn['gts'][-1]['root_actions'].append({
                    'before_id':
                    before_id,
                    'after_id':
                    after_id,
                    'actions':
                    root_actions
                })

    return gts_bmn


def combile_gts(gts_bmn, gts_process, mode):
    """
    1、bmn_window 范围内只有一个动作，只取一个目标框
    2、bmn_window 范围内有多个动作，取三个目标框(第一个动作、最后一个动作、所有动作)
    """
    global fps
    fps = gts_process['fps']
    duration_second = bmn_window * 1.0
    duration_frame = bmn_window * fps
    feature_frame = duration_frame
    for item in gts_process['gts']:
        url = item['url']
        basename = os.path.basename(url).split('.')[0]
        root_actions = item['root_actions']
        # 把每一个视频里的动作片段提取出来
        for root_action in root_actions:
            segments = []
            # all actions
            segments.append({
                'actions': root_action['actions'],
                'before_id': root_action['before_id'],
                'after_id': root_action['after_id']
            })
            if len(root_action['actions']) > 1:  #如果有多个动作，则第一个动作和最后一个动作，额外添加一次
                # first action
                segments.append({
                    'actions': [root_action['actions'][0]],
                    'before_id':
                    root_action['before_id'],
                    'after_id':
                    root_action['actions'][1]['start_id']
                })
                # last action
                segments.append({
                    'actions': [root_action['actions'][-1]],
                    'before_id':
                    root_action['actions'][-2]['end_id'],
                    'after_id':
                    root_action['after_id']
                })

            # 把动作片段处理成window size大小，以适配BMN输入
            for segment in segments:
                before_id = segment['before_id']
                after_id = segment['after_id']
                actions = segment['actions']
                # before_id到after_id太长了，从里面取window_size帧，要先确定一个起始点，然后动作都要包住
                box0 = max(actions[-1]['end_id'] - bmn_window,
                           before_id)  #确定起始点
                box1 = min(actions[0]['start_id'],
                           after_id - bmn_window)  #确实起始点
                if box0 <= box1:  # 一次检查
                    if int(box0) - int(box1) == 0:
                        cur_start = box0
                    else:
                        box0 = math.ceil(box0)
                        box1 = int(box1)
                        cur_start = random.randint(box0, box1)
                    cur_end = cur_start + bmn_window
                    cur_start = round(cur_start, 2)
                    cur_end = round(cur_end, 2)
                    name = '{}_{}_{}'.format(basename, cur_start, cur_end)
                    annotations = []
                    for action in actions:
                        label = str(1.0 * action['label_ids'][0])
                        label_name = action['label_names'][0]
                        seg0 = 1.0 * round((action['start_id'] - cur_start),
                                           2)  #存储的是到开始位置(时间: s)的距离
                        seg1 = 1.0 * round((action['end_id'] - cur_start), 2)
                        annotations.append({
                            'segment': [seg0, seg1],
                            'label': label,
                            'label_name': label_name
                        })
                    gts_bmn[name] = {
                        'duration_second': duration_second,
                        'duration_frame': duration_frame,
                        'feature_frame': feature_frame,
                        'subset': mode,
                        'annotations': annotations
                    }

    return gts_bmn


def save_feature_to_numpy(gts_bmn, folder):
    global fps
    print('save feature for bmn ...')
    if not os.path.exists(folder):
        os.mkdir(folder)
    process_gts_bmn = {}
    miss = 0
    for item, value in gts_bmn.items():
        # split to rsplit 针对文件命名修改
        basename, start_id, end_id = item.rsplit('_', 2)
        if not basename in process_gts_bmn:
            process_gts_bmn[basename] = []
        process_gts_bmn[basename].append({
            'name': item,
            'start': float(start_id),
            'end': float(end_id)
        })
    for item, values in process_gts_bmn.items():
        feat_path = os.path.join(feat_dir, item + '.pkl')
        feature_video = pickle.load(open(feat_path, 'rb'))['image_feature']
        for value in values:
            save_cut_name = os.path.join(folder, value['name'])
            a, b, c = save_cut_name.rsplit('_', 2)
            if float(b) > 360:
                print(b)
            start_frame = round(value['start'] * fps)
            end_frame = round(value['end'] * fps)
            if end_frame > len(feature_video):
                miss += 1
                continue
            feature_cut = [
                feature_video[i] for i in range(start_frame, end_frame)
            ]
            np_feature_cut = np.array(feature_cut, dtype=np.float32)
            np.save(save_cut_name, np_feature_cut)

    print('miss number (broken sample):', miss)


if __name__ == "__main__":
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    gts_bmn = {}
    for item, value in label_files.items():
        label_file = os.path.join(dataset, value)
        gts_data = json.load(open(label_file, 'rb'))
        gts_process = gen_gts_for_bmn(gts_data)
        gts_bmn = combile_gts(gts_bmn, gts_process, item)

    with open(out_dir + '/label.json', 'w', encoding='utf-8') as f:
        data = json.dumps(gts_bmn, indent=4, ensure_ascii=False)
        f.write(data)

    save_feature_to_numpy(gts_bmn, out_dir + '/feature')

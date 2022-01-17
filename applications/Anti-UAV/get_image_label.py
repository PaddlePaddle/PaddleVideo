import cv2
import os
import json

# please change it to your path
path = '/workspace/wangqingzhong/Anti_UAV'
annotation_path = 'annotations'
train_img_path = 'train_imgs'
val_img_path = 'val_imgs'
if not os.path.exists(annotation_path):
    os.makedirs(annotation_path)
if not os.path.exists(train_img_path):
    os.makedirs(train_img_path)
if not os.path.exists(val_img_path):
    os.makedirs(val_img_path)

train_info = {
    'images': [],
    'type':
    'instances',
    'annotations': [],
    'categories': [{
        "supercategory": "none",
        "id": 1,
        "name": "drone"
    }, {
        "supercategory": "none",
        "id": 2,
        "name": "noise"
    }]
}
val_info = {
    'images': [],
    'type':
    'instances',
    'annotations': [],
    'categories': [{
        "supercategory": "none",
        "id": 1,
        "name": "drone"
    }, {
        "supercategory": "none",
        "id": 2,
        "name": "noise"
    }]
}

# you can change it
interval = 5
dirs = os.listdir(path)
train_img_id = 0
val_img_id = 0
for d in dirs:
    if 'new' in d:
        video_file = os.path.join(path, d, 'IR.mp4')
        label_file = os.path.join(path, d, 'IR_label.json')
        labels = json.load(open(label_file, 'r'))
        exits = labels['exist']
        gt_bbox = labels['gt_rect']
        assert len(exits) == len(gt_bbox)
        videocap = cv2.VideoCapture(video_file)
        i = 0
        while True:
            success, frame = videocap.read()
            if success:
                if i % interval == 0:
                    img_name = d + '_' + str(i) + '.jpg'
                    cv2.imwrite(os.path.join(val_img_path, img_name), frame)
                    height, width, depth = frame.shape
                    x, y, w, h = gt_bbox[i]
                    isexist = exits[i]
                    if isexist:
                        category_id = 1
                    else:
                        category_id = 2
                    draw_frame = cv2.rectangle(frame, (x, y), (x + w, y + h),
                                               (0, 255, 0), 2)
                    img_name_draw = d + '_' + str(i) + 'draw.jpg'
                    cv2.imwrite(os.path.join(val_img_path, img_name_draw),
                                draw_frame)

                    img_info = {
                        'file_name': img_name,
                        'height': float(height),
                        'width': float(width),
                        'id': val_img_id
                    }
                    ann_info = {
                        'area': float(w) * float(h),
                        'iscrowd': 0,
                        'bbox': [float(x),
                                 float(y),
                                 float(w),
                                 float(h)],
                        'category_id': category_id,
                        'ignore': 0,
                        'image_id': val_img_id,
                        'id': val_img_id + 1
                    }
                    val_info['images'].append(img_info)
                    val_info['annotations'].append(ann_info)
                    val_img_id += 1
                i += 1
            else:
                print('finish {}'.format(d))
                break
    else:
        video_file = os.path.join(path, d, 'IR.mp4')
        label_file = os.path.join(path, d, 'IR_label.json')
        labels = json.load(open(label_file, 'r'))
        exits = labels['exist']
        gt_bbox = labels['gt_rect']
        assert len(exits) == len(gt_bbox)
        videocap = cv2.VideoCapture(video_file)
        i = 0
        while True:
            success, frame = videocap.read()
            if success:
                if i % interval == 0:
                    img_name = d + '_' + str(i) + '.jpg'
                    cv2.imwrite(os.path.join(train_img_path, img_name), frame)
                    height, width, depth = frame.shape
                    x, y, w, h = gt_bbox[i]
                    isexist = exits[i]
                    if isexist:
                        category_id = 1
                    else:
                        category_id = 2
                    draw_frame = cv2.rectangle(frame, (x, y), (x + w, y + h),
                                               (0, 255, 0), 2)
                    img_name_draw = d + '_' + str(i) + 'draw.jpg'
                    cv2.imwrite(os.path.join(train_img_path, img_name_draw),
                                draw_frame)

                    img_info = {
                        'file_name': img_name,
                        'height': height,
                        'width': width,
                        'id': train_img_id
                    }
                    ann_info = {
                        'area': float(w) * float(h),
                        'iscrowd': 0,
                        'bbox': [float(x),
                                 float(y),
                                 float(w),
                                 float(h)],
                        'category_id': category_id,
                        'ignore': 0,
                        'image_id': train_img_id,
                        'id': train_img_id + 1
                    }
                    train_info['images'].append(img_info)
                    train_info['annotations'].append(ann_info)
                    train_img_id += 1
                i += 1
            else:
                print('finish {}'.format(d))
                break

with open('annotations/train.json', 'w') as f:
    json.dump(train_info, f)
with open('annotations/val.json', 'w') as f:
    json.dump(val_info, f)

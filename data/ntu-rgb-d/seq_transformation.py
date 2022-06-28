# ref: https://github.com/Uason-Chen/CTR-GCN/blob/main/data/ntu/seq_transformation.py

import os
import os.path as osp
import numpy as np
import pickle
import logging
from sklearn.model_selection import train_test_split

root_path = './'
stat_path = osp.join(root_path, 'statistics')
setup_file = osp.join(stat_path, 'setup.txt')
camera_file = osp.join(stat_path, 'camera.txt')
performer_file = osp.join(stat_path, 'performer.txt')
replication_file = osp.join(stat_path, 'replication.txt')
label_file = osp.join(stat_path, 'label.txt')
skes_name_file = osp.join(stat_path, 'skes_available_name.txt')

denoised_path = osp.join(root_path, 'denoised_data')
raw_skes_joints_pkl = osp.join(denoised_path, 'raw_denoised_joints.pkl')
frames_file = osp.join(denoised_path, 'frames_cnt.txt')

save_path = './'

if not osp.exists(save_path):
    os.mkdir(save_path)


def remove_nan_frames(ske_name, ske_joints, nan_logger):
    num_frames = ske_joints.shape[0]
    valid_frames = []

    for f in range(num_frames):
        if not np.any(np.isnan(ske_joints[f])):
            valid_frames.append(f)
        else:
            nan_indices = np.where(np.isnan(ske_joints[f]))[0]
            nan_logger.info('{}\t{:^5}\t{}'.format(ske_name, f + 1,
                                                   nan_indices))

    return ske_joints[valid_frames]


def seq_translation(skes_joints):
    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2
        if num_bodies == 2:
            missing_frames_1 = np.where(ske_joints[:, :75].sum(axis=1) == 0)[0]
            missing_frames_2 = np.where(ske_joints[:, 75:].sum(axis=1) == 0)[0]
            cnt1 = len(missing_frames_1)
            cnt2 = len(missing_frames_2)

        i = 0  # get the "real" first frame of actor1
        while i < num_frames:
            if np.any(ske_joints[i, :75] != 0):
                break
            i += 1

        origin = np.copy(ske_joints[i, 3:6])  # new origin: joint-2

        for f in range(num_frames):
            if num_bodies == 1:
                ske_joints[f] -= np.tile(origin, 25)
            else:  # for 2 actors
                ske_joints[f] -= np.tile(origin, 50)

        if (num_bodies == 2) and (cnt1 > 0):
            ske_joints[missing_frames_1, :75] = np.zeros((cnt1, 75),
                                                         dtype=np.float32)

        if (num_bodies == 2) and (cnt2 > 0):
            ske_joints[missing_frames_2, 75:] = np.zeros((cnt2, 75),
                                                         dtype=np.float32)

        skes_joints[idx] = ske_joints  # Update

    return skes_joints


def frame_translation(skes_joints, skes_name, frames_cnt):
    nan_logger = logging.getLogger('nan_skes')
    nan_logger.setLevel(logging.INFO)
    nan_logger.addHandler(logging.FileHandler("./nan_frames.log"))
    nan_logger.info('{}\t{}\t{}'.format('Skeleton', 'Frame', 'Joints'))

    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        # Calculate the distance between spine base (joint-1) and spine (joint-21)
        j1 = ske_joints[:, 0:3]
        j21 = ske_joints[:, 60:63]
        dist = np.sqrt(((j1 - j21)**2).sum(axis=1))

        for f in range(num_frames):
            origin = ske_joints[f, 3:
                                6]  # new origin: middle of the spine (joint-2)
            if (ske_joints[f, 75:] == 0).all():
                ske_joints[f, :75] = (ske_joints[f, :75] - np.tile(origin, 25)) / \
                                      dist[f] + np.tile(origin, 25)
            else:
                ske_joints[f] = (ske_joints[f] - np.tile(origin, 50)) / \
                                 dist[f] + np.tile(origin, 50)

        ske_name = skes_name[idx]
        ske_joints = remove_nan_frames(ske_name, ske_joints, nan_logger)
        frames_cnt[idx] = num_frames  # update valid number of frames
        skes_joints[idx] = ske_joints

    return skes_joints, frames_cnt


def align_frames(skes_joints, frames_cnt):
    """
    Align all sequences with the same frame length.

    """
    num_skes = len(skes_joints)
    max_num_frames = frames_cnt.max()  # 300
    aligned_skes_joints = np.zeros((num_skes, max_num_frames, 150),
                                   dtype=np.float32)

    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2
        if num_bodies == 1:
            aligned_skes_joints[idx, :num_frames] = np.hstack(
                (ske_joints, np.zeros_like(ske_joints)))
        else:
            aligned_skes_joints[idx, :num_frames] = ske_joints

    return aligned_skes_joints


def one_hot_vector(labels):
    num_skes = len(labels)
    labels_vector = np.zeros((num_skes, 60))
    for idx, l in enumerate(labels):
        labels_vector[idx, l] = 1

    return labels_vector


def split_train_val(train_indices, method='sklearn', ratio=0.05):
    """
    Get validation set by splitting data randomly from training set with two methods.
    In fact, I thought these two methods are equal as they got the same performance.

    """
    if method == 'sklearn':
        return train_test_split(train_indices,
                                test_size=ratio,
                                random_state=10000)
    else:
        np.random.seed(10000)
        np.random.shuffle(train_indices)
        val_num_skes = int(np.ceil(0.05 * len(train_indices)))
        val_indices = train_indices[:val_num_skes]
        train_indices = train_indices[val_num_skes:]
        return train_indices, val_indices


def split_dataset(skes_name, skes_joints, label, performer, camera, evaluation,
                  save_path):
    train_indices, test_indices = get_indices(performer, camera, evaluation)
    m = 'sklearn'  # 'sklearn' or 'numpy'
    # Select validation set from training set
    # train_indices, val_indices = split_train_val(train_indices, m)

    # Save labels and num_frames for each sequence of each data set
    train_labels = label[train_indices]
    test_labels = label[test_indices]

    train_x = skes_joints[train_indices]
    # train_y = one_hot_vector(train_labels)
    test_x = skes_joints[test_indices]
    # test_y = one_hot_vector(test_labels)

    evaluation_path = osp.join(save_path, evaluation)
    isExists = osp.exists(evaluation_path)
    if not isExists:
        os.makedirs(evaluation_path)

    train_data_save_path = osp.join(evaluation_path, 'train_data.npy')
    train_label_save_path = osp.join(evaluation_path, 'train_label.pkl')
    val_data_save_path = osp.join(evaluation_path, 'val_data.npy')
    val_label_save_path = osp.join(evaluation_path, 'val_label.pkl')

    # reshape data
    N, T, VC = train_x.shape
    train_x = np.reshape(train_x, (N, T, 2, 25, 3))
    train_x = np.transpose(train_x, (0, 4, 1, 3, 2))

    N, T, VC = test_x.shape
    test_x = np.reshape(test_x, (N, T, 2, 25, 3))
    test_x = np.transpose(test_x, (0, 4, 1, 3, 2))
    # save train
    np.save(train_data_save_path, train_x)
    out = [skes_name[train_indices], train_labels]
    with open(train_label_save_path, 'wb') as f:
        pickle.dump(out, f)
    # save test
    np.save(val_data_save_path, test_x)
    out = [skes_name[test_indices], test_labels]
    with open(val_label_save_path, 'wb') as f:
        pickle.dump(out, f)


def get_indices(performer, camera, evaluation='xsub'):
    test_indices = np.empty(0)
    train_indices = np.empty(0)

    if evaluation == 'xsub':  # Cross Subject (Subject IDs)
        train_ids = [
            1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34,
            35, 38
        ]
        test_ids = [
            3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37,
            39, 40
        ]

        # Get indices of test data
        for idx in test_ids:
            temp = np.where(performer == idx)[0]  # 0-based index
            test_indices = np.hstack((test_indices, temp)).astype(np.int)

        # Get indices of training data
        for train_id in train_ids:
            temp = np.where(performer == train_id)[0]  # 0-based index
            train_indices = np.hstack((train_indices, temp)).astype(np.int)
    else:  # Cross View (Camera IDs)
        train_ids = [2, 3]
        test_ids = 1
        # Get indices of test data
        temp = np.where(camera == test_ids)[0]  # 0-based index
        test_indices = np.hstack((test_indices, temp)).astype(np.int)

        # Get indices of training data
        for train_id in train_ids:
            temp = np.where(camera == train_id)[0]  # 0-based index
            train_indices = np.hstack((train_indices, temp)).astype(np.int)

    return train_indices, test_indices


if __name__ == '__main__':
    camera = np.loadtxt(camera_file, dtype=np.int)  # camera id: 1, 2, 3
    performer = np.loadtxt(performer_file, dtype=np.int)  # subject id: 1~40
    label = np.loadtxt(label_file, dtype=np.int) - 1  # action label: 0~59

    frames_cnt = np.loadtxt(frames_file, dtype=np.int)  # frames_cnt
    skes_name = np.loadtxt(skes_name_file, dtype=np.string_)

    with open(raw_skes_joints_pkl, 'rb') as fr:
        skes_joints = pickle.load(fr)  # a list

    skes_joints = seq_translation(skes_joints)

    skes_joints = align_frames(skes_joints,
                               frames_cnt)  # aligned to the same frame length

    evaluations = ['xview', 'xsub']
    for evaluation in evaluations:
        split_dataset(skes_name, skes_joints, label, performer, camera,
                      evaluation, save_path)
    print('Done!')

import os
import pickle
import logging
from tqdm import tqdm
import numpy as np
import math



# ntu_reader.py
class NTU_Reader():
    def __init__(self, args, root_folder, transform, ntu60_path, ntu120_path, **kwargs):
        self.max_channel = 3
        self.max_frame = 300
        self.max_joint = 25
        self.max_person = 4
        self.select_person_num = 2
        self.dataset = args.dataset
        self.progress_bar = not args.no_progress_bar
        self.transform = transform

        # Set paths
        ntu_ignored = '{}/ignore.txt'.format(os.path.dirname(os.path.realpath(__file__)))
        if self.transform:
            self.out_path = '{}/transformed/{}'.format(root_folder, self.dataset)
        else:
            self.out_path = '{}/original/{}'.format(root_folder, self.dataset)
        create_folder(self.out_path)

        # Divide train and eval samples
        training_samples = dict()
        training_samples['ntu-xsub'] = [
            1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
        ]
        training_samples['ntu-xview'] = [2, 3]
        training_samples['ntu-xsub120'] = [
            1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
            38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
            80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
        ]
        training_samples['ntu-xset120'] = set(range(2, 33, 2))
        self.training_sample = training_samples[self.dataset]

        # Get ignore samples
        try:
            with open(ntu_ignored, 'r') as f:
                self.ignored_samples = [line.strip() + '.skeleton' for line in f.readlines()]
        except:
            logging.info('')
            logging.error('Error: Wrong in loading ignored sample file {}'.format(ntu_ignored))
            raise ValueError()

        # Get skeleton file list
        self.file_list = []
        for folder in [ntu60_path, ntu120_path]:
            for filename in os.listdir(folder):
                self.file_list.append((folder, filename))
            if '120' not in self.dataset:  # for NTU 60, only one folder
                break

    def read_file(self, file_path):
        skeleton = np.zeros((self.max_person, self.max_frame, self.max_joint, self.max_channel), dtype=np.float32)
        with open(file_path, 'r') as fr:
            frame_num = int(fr.readline())
            for frame in range(frame_num):
                person_num = int(fr.readline())
                for person in range(person_num):
                    person_info = fr.readline().strip().split()
                    joint_num = int(fr.readline())
                    for joint in range(joint_num):
                        joint_info = fr.readline().strip().split()
                        skeleton[person,frame,joint,:] = np.array(joint_info[:self.max_channel], dtype=np.float32)
        return skeleton[:,:frame_num,:,:], frame_num

    def get_nonzero_std(self, s):  # (T,V,C)
        index = s.sum(-1).sum(-1) != 0  # select valid frames
        s = s[index]
        if len(s) != 0:
            s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
        else:
            s = 0
        return s

    def gendata(self, phase):
        sample_data = []
        sample_label = []
        sample_path = []
        sample_length = []
        iterizer = tqdm(sorted(self.file_list), dynamic_ncols=True) if self.progress_bar else sorted(self.file_list)
        for folder, filename in iterizer:
            if filename in self.ignored_samples:
                continue
            # print(filename)

            # Get sample information
            file_path = os.path.join(folder, filename)
            
            setup_loc = filename.find('S')
            camera_loc = filename.find('C')
            subject_loc = filename.find('P')
            action_loc = filename.find('A')
            # print(filename[(setup_loc+1):(setup_loc+4)])
            setup_id = int(filename[(setup_loc+1):(setup_loc+4)])
            camera_id = int(filename[(camera_loc+1):(camera_loc+4)])
            subject_id = int(filename[(subject_loc+1):(subject_loc+4)])
            action_class = int(filename[(action_loc+1):(action_loc+4)])

            # Distinguish train or eval sample
            if self.dataset == 'ntu-xview':
                is_training_sample = (camera_id in self.training_sample)
            elif self.dataset == 'ntu-xsub' or self.dataset == 'ntu-xsub120':
                is_training_sample = (subject_id in self.training_sample)
            elif self.dataset == 'ntu-xset120':
                is_training_sample = (setup_id in self.training_sample)
            else:
                logging.info('')
                logging.error('Error: Do NOT exist this dataset {}'.format(self.dataset))
                raise ValueError()
            if (phase == 'train' and not is_training_sample) or (phase == 'eval' and is_training_sample):
                continue

            # Read one sample
            data = np.zeros((self.max_channel, self.max_frame, self.max_joint, self.select_person_num), dtype=np.float32)
            skeleton, frame_num = self.read_file(file_path)

            # Select person by max energy
            energy = np.array([self.get_nonzero_std(skeleton[m]) for m in range(self.max_person)])
            index = energy.argsort()[::-1][:self.select_person_num]
            skeleton = skeleton[index]
            data[:,:frame_num,:,:] = skeleton.transpose(3, 1, 2, 0)

            sample_data.append(data)
            sample_path.append(file_path)
            sample_label.append(action_class - 1)  # to 0-indexed
            sample_length.append(frame_num)

        # Save label
        with open('{}/{}_label.pkl'.format(self.out_path, phase), 'wb') as f:
            pickle.dump((sample_path, list(sample_label), list(sample_length)), f)

        # Transform data
        sample_data = np.array(sample_data)
        if self.transform:
            sample_data = pre_normalization(sample_data, progress_bar=self.progress_bar)

        # Save data
        np.save('{}/{}_data.npy'.format(self.out_path, phase), sample_data)

    def start(self):
        for phase in ['train', 'eval']:
            logging.info('Phase: {}'.format(phase))
            self.gendata(phase)

# __init__.py
__generator = {
    'ntu': NTU_Reader,
}
def create(args):
    dataset = args.dataset.split('-')[0]
    dataset_args = args.dataset_args[dataset]
    if dataset not in __generator.keys():
        logging.info('')
        logging.error('Error: Do NOT exist this dataset: {}!'.format(dataset))
        raise ValueError()
    return __generator[dataset](args, **dataset_args)

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)






# graphs.py
# Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
class Graph():
    def __init__(self, dataset, max_hop=10, dilation=1):
        self.dataset = dataset.split('-')[0]
        self.max_hop = max_hop
        self.dilation = dilation

        # get edges
        self.num_node, self.edge, self.connect_joint, self.parts = self._get_edge()

        # get adjacency matrix
        self.A = self._get_adjacency()

    def __str__(self):
        return self.A

    def _get_edge(self):
        if self.dataset == 'kinetics':
            num_node = 18
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14), (8, 11)]
            connect_joint = np.array([1,1,1,2,3,1,5,6,2,8,9,5,11,12,0,0,14,15])
            parts = [
                np.array([5, 6, 7]),              # left_arm
                np.array([2, 3, 4]),              # right_arm
                np.array([11, 12, 13]),           # left_leg
                np.array([8, 9, 10]),             # right_leg
                np.array([0, 1, 14, 15, 16, 17])  # torso
            ]
        elif self.dataset == 'ntu':
            num_node = 25
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            connect_joint = np.array([2,2,21,3,21,5,6,7,21,9,10,11,1,13,14,15,1,17,18,19,2,23,8,25,12]) - 1
            parts = [
                np.array([5, 6, 7, 8, 22, 23]) - 1,     # left_arm
                np.array([9, 10, 11, 12, 24, 25]) - 1,  # right_arm
                np.array([13, 14, 15, 16]) - 1,         # left_leg
                np.array([17, 18, 19, 20]) - 1,         # right_leg
                np.array([1, 2, 3, 4, 21]) - 1          # torso
            ]
        elif self.dataset == 'sysu':
            num_node = 20
            neighbor_1base = [(1, 2), (2, 3), (3, 4), (3, 5), (5, 6),
                              (6, 7), (7, 8), (3, 9), (9, 10), (10, 11),
                              (11, 12), (1, 13), (13, 14), (14, 15), (15, 16),
                              (1, 17), (17, 18), (18, 19), (19, 20)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            connect_joint = np.array([2,2,2,3,3,5,6,7,3,9,10,11,1,13,14,15,1,17,18,19]) - 1
            parts = [
                np.array([5, 6, 7, 8]) - 1,      # left_arm
                np.array([9, 10, 11, 12]) - 1,   # right_arm
                np.array([13, 14, 15, 16]) - 1,  # left_leg
                np.array([17, 18, 19, 20]) - 1,  # right_leg
                np.array([1, 2, 3, 4]) - 1       # torso
            ]
        elif self.dataset == 'ucla':
            num_node = 20
            neighbor_1base = [(1, 2), (2, 3), (3, 4), (3, 5), (5, 6),
                              (6, 7), (7, 8), (3, 9), (9, 10), (10, 11),
                              (11, 12), (1, 13), (13, 14), (14, 15), (15, 16),
                              (1, 17), (17, 18), (18, 19), (19, 20)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            connect_joint = np.array([2,2,2,3,3,5,6,7,3,9,10,11,1,13,14,15,1,17,18,19]) - 1
            parts = [
                np.array([5, 6, 7, 8]) - 1,      # left_arm
                np.array([9, 10, 11, 12]) - 1,   # right_arm
                np.array([13, 14, 15, 16]) - 1,  # left_leg
                np.array([17, 18, 19, 20]) - 1,  # right_leg
                np.array([1, 2, 3, 4]) - 1       # torso
            ]
        elif self.dataset == 'cmu':
            num_node = 26
            neighbor_1base = [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7),
                              (7, 8), (1, 9), (5, 9), (9, 10), (10, 11),
                              (11, 12), (12, 13), (13, 14), (12, 15), (15, 16),
                              (16, 17), (17, 18), (18, 19), (17, 20), (12, 21),
                              (21, 22), (22, 23), (23, 24), (24, 25), (23, 26)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            connect_joint = np.array([9,1,2,3,9,5,6,7,10,10,10,11,12,13,12,15,16,17,18,17,12,21,22,23,24,23]) - 1
            parts = [
                np.array([15, 16, 17, 18, 19, 20]) - 1,  # left_arm
                np.array([21, 22, 23, 24, 25, 26]) - 1,  # right_arm
                np.array([1, 2, 3, 4]) - 1,              # left_leg
                np.array([5, 6, 7, 8]) - 1,              # right_leg
                np.array([9, 10, 11, 12, 13, 14]) - 1    # torso
            ]
        elif self.dataset == 'h36m':
            num_node = 20
            neighbor_1base = [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7),
                              (7, 8), (1, 9), (5, 9), (9, 10), (10, 11),
                              (11, 12), (10, 13), (13, 14), (14, 15), (15, 16),
                              (10, 17), (17, 18), (18, 19), (19, 20)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            connect_joint = np.array([9,1,2,3,9,5,6,7,9,9,10,11,10,13,14,15,10,17,18,19]) - 1
            parts = [
                np.array([13, 14, 15, 16]) - 1,  # left_arm
                np.array([17, 18, 19, 20]) - 1,  # right_arm
                np.array([1, 2, 3, 4]) - 1,      # left_leg
                np.array([5, 6, 7, 8]) - 1,      # right_leg
                np.array([9, 10, 11, 12]) - 1    # torso
            ]
        else:
            logging.info('')
            logging.error('Error: Do NOT exist this dataset: {}!'.format(self.dataset))
            raise ValueError()
        self_link = [(i, i) for i in range(num_node)]
        edge = self_link + neighbor_link
        return num_node, edge, connect_joint, parts

    def _get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_adjacency(self):
        hop_dis = self._get_hop_distance()
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[hop_dis == hop] = 1
        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
        return A

    def _normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD



# transformer.py

def rotation_matrix(axis, theta):
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3)
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def angle_between(v1, v2):
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    v1_u = v1 / np.linalg.norm(v1)  # unit_vector
    v2_u = v2 / np.linalg.norm(v2)  # unit_vector
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def pre_normalization(data, progress_bar=True, zaxis=[0, 1], xaxis=[8, 4]):
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C

    logging.info('Pad the null frames with the previous frames')
    items = tqdm(s, dynamic_ncols=True) if progress_bar else s
    for i_s, skeleton in enumerate(items):  # pad
        if skeleton.sum() == 0:
            logging.info('Sample {:d} has no skeleton'.format(i_s))
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            if person[0].sum() == 0:
                index = (person.sum(-1).sum(-1) != 0)
                tmp = person[index].copy()
                person *= 0
                person[:len(tmp)] = tmp
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    if person[i_f:].sum() == 0:
                        rest = len(person) - i_f
                        num = int(np.ceil(rest / i_f))
                        pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]
                        s[i_s, i_p, i_f:] = pad
                        break

    logging.info('Sub the center joint #1 (spine joint in ntu and neck joint in kinetics)')
    items = tqdm(s, dynamic_ncols=True) if progress_bar else s
    for i_s, skeleton in enumerate(items):
        if skeleton.sum() == 0:
            continue
        main_body_center = skeleton[0][:, 1:2, :].copy()
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            mask = (person.sum(-1) != 0).reshape(T, V, 1)
            s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask

    logging.info('Parallel the bone between hip(jpt 0) and spine(jpt 1) of the first person to the z axis')
    items = tqdm(s, dynamic_ncols=True) if progress_bar else s
    for i_s, skeleton in enumerate(items):
        if skeleton.sum() == 0:
            continue
        joint_bottom = skeleton[0, 0, zaxis[0]]
        joint_top = skeleton[0, 0, zaxis[1]]
        axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
        angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
        matrix_z = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_z, joint)

    logging.info('Parallel the bone between right shoulder(jpt 8) and left shoulder(jpt 4) of the first person to the x axis')
    items = tqdm(s, dynamic_ncols=True) if progress_bar else s
    for i_s, skeleton in enumerate(items):
        if skeleton.sum() == 0:
            continue
        joint_rshoulder = skeleton[0, 0, xaxis[0]]
        joint_lshoulder = skeleton[0, 0, xaxis[1]]
        axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        matrix_x = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_x, joint)

    data = np.transpose(s, [0, 4, 2, 3, 1])
    return data


# ntu_location_feeder.py
class NTU_Location_Feeder():
    def __init__(self, data_shape):
        _, _, self.T, self.V, self.M = data_shape

    def load(self, names):
        location = np.zeros((len(names), 2, self.T, self.V, self.M))
        for i, name in enumerate(names):
            with open(name, 'r') as fr:
                frame_num = int(fr.readline())
                for frame in range(frame_num):
                    if frame >= self.T:
                        break
                    person_num = int(fr.readline())
                    for person in range(person_num):
                        fr.readline()
                        joint_num = int(fr.readline())
                        for joint in range(joint_num):
                            v = fr.readline().split(' ')
                            if joint < self.V and person < self.M:
                                location[i,0,frame,joint,person] = float(v[5])
                                location[i,1,frame,joint,person] = float(v[6])
        return location



# muti_fea_gen.py
def multi_input(data,conn):
        C, T, V, M = data.shape
        joint = np.zeros((C*2, T, V, M))
        velocity = np.zeros((C*2, T, V, M))
        bone = np.zeros((C*2, T, V, M))
        joint[:C,:,:,:] = data
        for i in range(V):
            joint[C:,:,i,:] = data[:,:,i,:] - data[:,:,1,:]
        for i in range(T-2):
            velocity[:C,i,:,:] = data[:,i+1,:,:] - data[:,i,:,:]
            velocity[C:,i,:,:] = data[:,i+2,:,:] - data[:,i,:,:]
        for i in range(len(conn)):
            bone[:C,:,i,:] = data[:,:,i,:] - data[:,:,conn[i],:]
        bone_length = 0
        for i in range(C):
            bone_length += bone[i,:,:,:] ** 2
        bone_length = np.sqrt(bone_length) + 0.0001
        for i in range(C):
            bone[C+i,:,:,:] = np.arccos(bone[i,:,:,:] / bone_length)
        return joint, velocity, bone
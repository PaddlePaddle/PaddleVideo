import os.path as osp
import copy
import random
import numpy as np
import pickle

from ..registry import DATASETS
from .base import BaseDataset
from ...utils import get_logger

logger = get_logger("paddlevideo")


class Graph():
    def __init__(self, dataset, max_hop=10, dilation=1):
        self.dataset = dataset.split('-')[0]
        self.max_hop = max_hop
        self.dilation = dilation

        # get edges
        self.num_node, self.edge, self.connect_joint, self.parts = self._get_edge(
        )

        # get adjacency matrix
        self.A = self._get_adjacency()

    def __str__(self):
        return self.A

    def _get_edge(self):
        if self.dataset == 'ntu':
            num_node = 25
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                              (7, 6), (8, 7), (9, 21), (10, 9), (11, 10),
                              (12, 11), (13, 1), (14, 13), (15, 14), (16, 15),
                              (17, 1), (18, 17), (19, 18), (20, 19), (22, 23),
                              (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            connect_joint = np.array([
                2, 2, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17,
                18, 19, 2, 23, 8, 25, 12
            ]) - 1
            parts = [
                np.array([5, 6, 7, 8, 22, 23]) - 1,  # left_arm
                np.array([9, 10, 11, 12, 24, 25]) - 1,  # right_arm
                np.array([13, 14, 15, 16]) - 1,  # left_leg
                np.array([17, 18, 19, 20]) - 1,  # right_leg
                np.array([1, 2, 3, 4, 21]) - 1  # torso
            ]
        else:
            logger.info('')
            logger.error('Error: Do NOT exist this dataset: {}!'.format(
                self.dataset))
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
        transfer_mat = [
            np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)
        ]
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


@DATASETS.register()
class NTUDataset(BaseDataset):
    def __init__(self,
                 file_path,
                 pipeline,
                 label_path=None,
                 test_mode=False,
                 T=300,
                 dataset="ntu",
                 inputs="JVB"):
        self.T = T
        graph = Graph(dataset)
        self.inputs = inputs
        self.conn = graph.connect_joint
        self.label_path = label_path
        super().__init__(file_path, pipeline, test_mode=test_mode)

    def load_file(self):
        logger.info("Loading data, it will take some moment...")
        data_path = self.file_path
        self.data = None
        try:
            self.data = np.load(data_path, mmap_mode='r')
            with open(self.label_path, 'rb') as f:
                self.name, self.label = pickle.load(f, encoding='latin1')[0:2]
        except:
            logger.info('')
            logger.error('Error: Wrong in loading data files: {} or {}!'.format(
                data_path, self.label_path))
            logger.info('Please generate data first!')
            raise ValueError()
        logger.info("Data Loaded!")
        return self.data

    def prepare_train(self, idx):
        data = np.array(self.data[idx])
        label = self.label[idx]
        name = self.name[idx]

        joint, velocity, bone = self.multi_input(data[:, :self.T, :, :])
        data_new = []
        if 'J' in self.inputs:
            data_new.append(joint)
        if 'V' in self.inputs:
            data_new.append(velocity)
        if 'B' in self.inputs:
            data_new.append(bone)
        data_new = np.stack(data_new, axis=0)
        results = dict()
        results['data'] = copy.deepcopy(data_new)
        results['label'] = copy.deepcopy(label)
        results = self.pipeline(results)
        return results['data'], results['label']  # , name

    def prepare_test(self, idx):
        data = np.array(self.data[idx])
        label = self.label[idx]
        name = self.name[idx]

        joint, velocity, bone = self.multi_input(data[:, :self.T, :, :])
        data_new = []
        if 'J' in self.inputs:
            data_new.append(joint)
        if 'V' in self.inputs:
            data_new.append(velocity)
        if 'B' in self.inputs:
            data_new.append(bone)
        data_new = np.stack(data_new, axis=0)
        results = dict()
        results['data'] = copy.deepcopy(data_new)
        results['label'] = copy.deepcopy(label)
        results = self.pipeline(results)
        return results['data'], results['label']  # , name

    def multi_input(self, data):
        C, T, V, M = data.shape
        joint = np.zeros((C * 2, T, V, M))
        velocity = np.zeros((C * 2, T, V, M))
        bone = np.zeros((C * 2, T, V, M))
        joint[:C, :, :, :] = data
        for i in range(V):
            joint[C:, :, i, :] = data[:, :, i, :] - data[:, :, 1, :]
        for i in range(T - 2):
            velocity[:C, i, :, :] = data[:, i + 1, :, :] - data[:, i, :, :]
            velocity[C:, i, :, :] = data[:, i + 2, :, :] - data[:, i, :, :]
        for i in range(len(self.conn)):
            bone[:C, :, i, :] = data[:, :, i, :] - data[:, :, self.conn[i], :]
        bone_length = 0
        for i in range(C):
            bone_length += bone[i, :, :, :]**2
        bone_length = np.sqrt(bone_length) + 0.0001
        for i in range(C):
            bone[C + i, :, :, :] = np.arccos(bone[i, :, :, :] / bone_length)
        return joint, velocity, bone

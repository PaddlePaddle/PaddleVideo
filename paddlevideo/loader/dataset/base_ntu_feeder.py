import pickle, logging
import numpy as np

import paddle.io as io

from ..pipelines import multi_input

class NTU_Feeder(io.Dataset):
    def __init__(self, phase, dataset_path, inputs, num_frame, connect_joint, debug, **kwargs):
        self.T = num_frame
        self.inputs = inputs
        self.conn = connect_joint
        data_path = '{}/{}_data.npy'.format(dataset_path, phase)
        label_path = '{}/{}_label.pkl'.format(dataset_path, phase)

        try:
            self.data = np.load(data_path, mmap_mode='r')
            with open(label_path, 'rb') as f:
                self.name, self.label, self.seq_len = pickle.load(f, encoding='latin1')
        except:
            logging.info('')
            logging.error('Error: Wrong in loading data files: {} or {}!'.format(data_path, label_path))
            logging.info('Please generate data first!')
            raise ValueError()
        if debug:
            self.data = self.data[:300]
            self.label = self.label[:300]
            self.name = self.name[:300]
            self.seq_len = self.seq_len[:300]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = np.array(self.data[idx])
        label = self.label[idx]
        name = self.name[idx]

        # (C, max_frame, V, M) -> (I, C*2, T, V, M)
        joint, velocity, bone = multi_input(data[:,:self.T,:,:],self.conn)
        data_new = []
        if 'J' in self.inputs:
            data_new.append(joint)
        if 'V' in self.inputs:
            data_new.append(velocity)
        if 'B' in self.inputs:
            data_new.append(bone)
        data_new = np.stack(data_new, axis=0)

        return data_new, label, name

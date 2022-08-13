import logging

from .graphs import Graph
from .ntu_feeder import NTU_Feeder, NTU_Location_Feeder


__data_args = {
    'ntu-xsub': {'class': 60, 'shape': [3, 6, 300, 25, 2], 'feeder': NTU_Feeder},
    'ntu-xview': {'class': 60, 'shape': [3, 6, 300, 25, 2], 'feeder': NTU_Feeder},
    'ntu-xsub120': {'class': 120, 'shape': [3, 6, 300, 25, 2], 'feeder': NTU_Feeder},
    'ntu-xset120': {'class': 120, 'shape': [3, 6, 300, 25, 2], 'feeder': NTU_Feeder},
}

def create(dataset, root_folder, transform, num_frame, inputs, **kwargs):
    graph = Graph(dataset)
    try:
        data_args = __data_args[dataset]
        data_args['shape'][0] = len(inputs)
        data_args['shape'][2] = num_frame
    except:
        logging.info('')
        logging.error('Error: Do NOT exist this dataset: {}!'.foramt(dataset))
        raise ValueError()
    if transform:
        dataset_path = '{}/transformed/{}'.format(root_folder, dataset)
    else:
        dataset_path = '{}/original/{}'.format(root_folder, dataset)
    kwargs.update({
        'dataset_path': dataset_path,
        'inputs': inputs,
        'num_frame': num_frame,
        'connect_joint': graph.connect_joint,
    })
    feeders = {
        'train': data_args['feeder']('train', **kwargs),
        'eval' : data_args['feeder']('eval', **kwargs),
    }
    if 'ntu' in dataset:
        feeders.update({'location': NTU_Location_Feeder(data_args['shape'])})
    return feeders, data_args['shape'], data_args['class'], graph.A, graph.parts

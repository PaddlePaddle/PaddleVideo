   
import os.path
import os
import argparse
import numpy as np
import pickle
import random

def get_args(add_help=True):
    """
    parse args
    """
    parser = argparse.ArgumentParser(
        description="gen sample data", add_help=add_help)
    parser.add_argument(
        '--source_path',
        type=str,
        default='./data/npy_dataset',
        help='save path of result data')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./data/ntu/tiny_dataset',
        help='save path of result data')
    parser.add_argument(
        '--data_num',
        type=int,
        default=16*5,
        help='data num of result data')
    parser.add_argument(
        '--transform',
        default=False, action='store_true') 
    parser.add_argument(
        '--dataset',
        type=str,
        default='ntu-xsub',
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        help='train/eval')        
    args = parser.parse_args()
    return args

def gen_tiny_data(source_path, save_dir, data_num, use_mmap=True, mode='train'):
    if mode == 'train':
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        train_data_path = os.path.join(source_path, 'train_data.npy')
        train_data_label = os.path.join(source_path, 'train_label.pkl')
        eval_data_path = os.path.join(source_path, 'eval_data.npy')
        eval_data_label = os.path.join(source_path, 'eval_label.pkl')
        if use_mmap:
            train_data = np.load(train_data_path, mmap_mode='r')
            eval_data = np.load(eval_data_path, mmap_mode='r')
        else:
            train_data = np.load(train_data_path)
            eval_data = np.load(eval_data_path)
        try:
            with open(train_data_label) as f:
                train_sample_name, train_label, train_seqlen = pickle.load(f)
            with open(eval_data_label) as f:
                eval_sample_name, eval_label, eval_seqlen = pickle.load(f)
        except:
            # for pickle file from python2
            with open(train_data_label, 'rb') as f:
                train_sample_name, train_label, train_seqlen = pickle.load(f, encoding='latin1')
            with open(eval_data_label, 'rb') as f:
                eval_sample_name, eval_label, eval_seqlen = pickle.load(f, encoding='latin1')
            """with open(label_path, 'rb') as f:
                sample_name, label, seqlen = pickle.load(f, encoding='latin1')"""
        start = random.randint(0, len(train_label) - data_num)
        train_label = train_label[start: start + data_num]
        train_data = train_data[start: start + data_num]
        train_seqlen = train_seqlen[start: start + data_num]
        train_sample_name = train_sample_name[start: start + data_num]
        with open(os.path.join(save_dir, f"train_label.pkl"), 'wb') as f:  
            pickle.dump((train_sample_name, list(train_label),train_seqlen), f)
        np.save(os.path.join(save_dir, f"train_data"), train_data)
        start = random.randint(0, len(eval_label) - data_num)
        eval_label = eval_label[start: start + data_num]
        eval_data = eval_data[start: start + data_num]
        eval_seqlen = eval_seqlen[start: start + data_num]
        eval_sample_name = eval_sample_name[start: start + data_num]
        with open(os.path.join(save_dir, f"eval_label.pkl"), 'wb') as f:  
            pickle.dump((eval_sample_name, list(eval_label),eval_seqlen), f)
        np.save(os.path.join(save_dir, f"eval_data"), eval_data)
        print("Successfully generate tiny dataset")
    elif mode == 'eval':
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        eval_data_path = os.path.join(source_path, 'eval_data.npy')
        eval_data_label = os.path.join(source_path, 'eval_label.pkl')
        if use_mmap:
            eval_data = np.load(eval_data_path, mmap_mode='r')
        else:
            eval_data = np.load(eval_data_path)
        try:
            with open(eval_data_label) as f:
                eval_sample_name, eval_label, eval_seqlen = pickle.load(f)
        except:
            # for pickle file from python2
            with open(eval_data_label, 'rb') as f:
                eval_sample_name, eval_label, eval_seqlen = pickle.load(f, encoding='latin1')
            """with open(label_path, 'rb') as f:
                sample_name, label, seqlen = pickle.load(f, encoding='latin1')"""
        start = random.randint(0, len(eval_label) - data_num)

        eval_label = eval_label[start: start + data_num]
        eval_data = eval_data[start: start + data_num]
        eval_seqlen = eval_seqlen[start: start + data_num]
        eval_sample_name = eval_sample_name[start: start + data_num]
        with open(os.path.join(save_dir, f"eval_label.pkl"), 'wb') as f:  
            pickle.dump((eval_sample_name, list(eval_label),eval_seqlen), f)
        np.save(os.path.join(save_dir, f"eval_data"), eval_data)
        print("Successfully generate tiny dataset")
    else:
        raise NotImplementedError
    


if __name__ == "__main__":
    args = get_args()
    gen_tiny_data(os.path.join(args.source_path, 'transform' if args.transform else 'original', args.dataset), 
            os.path.join(args.save_dir, 'transform' if args.transform else 'original', args.dataset), 
            data_num=args.data_num, mode=args.mode)

import os, argparse

from regex import R
os.chdir(os.getcwd())


parser = argparse.ArgumentParser(description='Modify Configs')
parser.add_argument('--root_folder', '-r', type=str, default='', help='Your path to save numpy dataset')
parser.add_argument('--ntu60_path', '-n60p', type=str, default='', help='Your path to save original NTU 60 dataset (S001 to S017)')
parser.add_argument('--ntu120_path', '-n120p', type=str, default='', help='Your path to save original NTU 120 dataset (S018 to S032)')
parser.add_argument('--pretrained_path', '-pp', type=str, default='', help='Your path to save pretrained models')
parser.add_argument('--work_dir', '-wd', type=str, default='', help='Your path to save checkpoints and log files')
parser.add_argument('--gpus', '-g', type=str, default='', help='Your gpu device numbers')
args, _ = parser.parse_known_args()

for file in os.listdir('./PaddleVideo-develop/configs/segmentation/EfficientGCN/'):
    with open(os.path.join('./PaddleVideo-develop/configs/segmentation/EfficientGCN/',file),'r') as f:
        lines = f.readlines()
    fr = open(os.path.join('./PaddleVideo-develop/configs/segmentation/EfficientGCN/', file), 'w')
    for line in lines:
        if 'root_folder:' in line and args.root_folder != '':
            new_line = '    root_folder: '+ args.root_folder + '\n'
        elif 'ntu60_path:' in line and args.ntu60_path != '':
            new_line = '    ntu60_path: ' + args.ntu60_path + '\n'
        elif 'ntu120_path:' in line and args.ntu120_path != '':
            new_line = '    ntu120_path: ' + args.ntu120_path +'\n'
        elif 'pretrained_path:' in line and args.pretrained_path != '':
            new_line = 'pretrained_path: ' + args.pretrained_path + '\n'
        elif 'work_dir:' in line and args.work_dir != '':
            new_line = 'work_dir: ' + args.work_dir + '\n'
        elif 'gpus:' in line and args.work_dir != '':
            new_line = 'gpus: '+ args.gpus + '\n'
        else:
            new_line = line
        fr.write(new_line)
    fr.close()

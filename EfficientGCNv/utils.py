import os, sys, shutil, logging, json
from time import time, strftime, localtime


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def get_time(total_time):
    s = int(total_time % 60)
    m = int(total_time / 60) % 60
    h = int(total_time / 60 / 60) % 24
    d = int(total_time / 60 / 60 / 24)
    return '{:0>2d}d-{:0>2d}h-{:0>2d}m-{:0>2d}s'.format(d, h, m, s)


def get_current_timestamp():
    ct = time()
    ms = int((ct - int(ct)) * 1000)
    return '[ {},{:0>3d} ] '.format(strftime('%Y-%m-%d %H:%M:%S', localtime(ct)), ms)


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def set_logging(args):
    if args.debug or args.evaluate or args.extract or args.visualize or args.generate_data:
        save_dir = '{}/temp'.format(args.work_dir)
    else:
        ct = strftime('%Y-%m-%d %H-%M-%S')
        save_dir = '{}/{}_{}_{}/{}'.format(args.work_dir, args.config, args.model_type, args.dataset, ct)
    create_folder(save_dir)
    log_format = '[ %(asctime)s ] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
    handler = logging.FileHandler('{}/log.txt'.format(save_dir), mode='w', encoding='UTF-8')
    handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(handler)
    return save_dir


    
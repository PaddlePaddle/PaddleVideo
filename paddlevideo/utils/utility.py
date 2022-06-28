import os, sys, shutil, logging, json
from time import time, strftime, localtime
import paddle

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


def load_checkpoint(work_dir, model_name='resume'):
    if model_name == 'resume':
        file_name = '{}/checkpoint.pth.tar'.format(work_dir)
    elif model_name == 'debug':
        file_name = '{}/temp/debug.pth.tar'.format(work_dir)
    else:
        dirs, accs = {}, {}
        work_dir = '{}/{}'.format(work_dir, model_name)
        # logging.info('work_dir is ::: ', work_dir)
        if os.path.exists(work_dir):
            for i, dir_time in enumerate(os.listdir(work_dir)):
                if os.path.isdir('{}/{}'.format(work_dir, dir_time)):
                    state_file = '{}/{}/reco_results.json'.format(work_dir, dir_time)
                    if os.path.exists(state_file):
                        print(state_file)
                        with open(state_file, 'r') as f:
                            best_state = json.load(f)
                        accs[str(i+1)] = best_state['acc_top1']
                        dirs[str(i+1)] = dir_time
        if len(dirs) == 0:
            logging.warning('Warning: Do NOT exists any model in workdir!')
            logging.info('Evaluating initial or pretrained model.')
            return None
        logging.info('Please choose the evaluating model from the following models.')
        logging.info('Default is the initial or pretrained model.')
        for key in dirs.keys():
            logging.info('({}) accuracy: {:.2%} | training time: {}'.format(key, accs[key], dirs[key]))
        logging.info('Your choice (number of the model, q for quit): ')
        while True:
            # idx = input(get_current_timestamp())
            idx = '1'

            if idx == '':
                logging.info('Evaluating initial or pretrained model.')
                return None
            elif idx in dirs.keys():
                break
            elif idx == 'q':
                logging.info('Quit!')
                sys.exit(1)
            else:
                logging.info('Wrong choice!')
        file_name = '{}/{}/{}.pth.tar'.format(work_dir, dirs[idx], model_name)
    try:
        checkpoint = paddle.load(file_name)
    except:
        logging.info('')
        logging.error('Error: Wrong in loading this checkpoint: {}!'.format(file_name))
        raise ValueError()
    return checkpoint


def save_checkpoint(model, optimizer, scheduler, epoch, best_state, is_best, work_dir, save_dir, model_name):
    for key in model.keys():
        model[key] = model[key].cpu()
    checkpoint = {
        'model':model, 'optimizer':optimizer, 'scheduler':scheduler,
        'best_state':best_state, 'epoch':epoch,
    }
    cp_name = '{}/checkpoint.pth.tar'.format(work_dir)

    paddle.save(checkpoint, cp_name)
    
    if is_best:
        shutil.copy(cp_name, '{}/{}.pth.tar'.format(save_dir, model_name))
        with open('{}/reco_results.json'.format(save_dir), 'w') as f:
            del best_state['cm']
            json.dump(best_state, f)


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

class CrossEntropyLabelSmooth(paddle.nn.Layer):
	def __init__(self, num_classes, epsilon):
		super(CrossEntropyLabelSmooth, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.logsoftmax = paddle.nn.LogSoftmax(axis=1)

	def forward(self, inputs, targets):
		log_probs = self.logsoftmax(inputs)

		targets = paddle.zeros_like(log_probs)
		targets = paddle.scatter(paddle.unsqueeze(targets,axis=1), 1, 1)

		targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
		loss = (-targets * log_probs).mean(0).sum()
		return loss

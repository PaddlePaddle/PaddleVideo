
import os, yaml, warnings, logging, pynvml, numpy as np,seaborn as sns, matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import paddle
from tqdm import tqdm
from time import time
import sys

sys.path.insert(0,'/home/aistudio/PaddleVideo-develop/paddlevideo')
from metrics import metrics_train,metrics_eval
from utils import utility as U
import solver
from loader import dataset
from modeling import framework
from loader import pipelines

class Generator():
    def __init__(self, args):
        U.set_logging(args)
        self.dataset = args.dataset
        self.generator = pipelines.create(args)

    def start(self):
        logging.info('')
        logging.info('Starting generating ...')
        logging.info('Dataset: {}'.format(self.dataset))
        self.generator.start()
        logging.info('Finish generating!')

class Initializer():
    def __init__(self, args):
        self.args = args                            # 读取参数
        self.init_save_dir()                        # 设置logging
        logging.info('')
        logging.info('Starting preparing ...')
        self.init_environment()
        self.init_device()
        self.init_dataloader()

        self.init_model()
        self.init_lr_scheduler()
        self.init_optimizer()
        self.init_loss_func()
        logging.info('Successful!')
        logging.info('')

    def init_save_dir(self):
        self.save_dir = U.set_logging(self.args)
        with open('{}/config.yaml'.format(self.save_dir), 'w') as f:
            yaml.dump(vars(self.args), f)
        logging.info('Saving folder path: {}'.format(self.save_dir))

    def init_environment(self):
        np.random.seed(self.args.seed)

        paddle.seed(self.args.seed)

        self.global_step = 0
        if self.args.debug:
            self.no_progress_bar = True
            self.model_name = 'debug'
            self.scalar_writer = None
        elif self.args.evaluate or self.args.extract:

            self.no_progress_bar = self.args.no_progress_bar
            self.model_name = '{}_{}_{}'.format(self.args.config, self.args.model_type, self.args.dataset)
            self.scalar_writer = None
            warnings.filterwarnings('ignore')
        else:
            self.no_progress_bar = self.args.no_progress_bar
            self.model_name = '{}_{}_{}'.format(self.args.config, self.args.model_type, self.args.dataset)
            self.scalar_writer = SummaryWriter(logdir=self.save_dir)
            warnings.filterwarnings('ignore')
        logging.info('Saving model name: {}'.format(self.model_name))

    def init_device(self):
        if type(self.args.gpus) is int:
            self.args.gpus = [self.args.gpus]
        
        if len(self.args.gpus) > 0 :
            pynvml.nvmlInit()
            for i in self.args.gpus:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memused = meminfo.used / 1024 / 1024
                logging.info('GPU-{} used: {}MB'.format(i, memused))
                if memused > 3000:
                    pynvml.nvmlShutdown()
                    logging.info('')
                    logging.error('GPU-{} is occupied!'.format(i))
                    # raise ValueError()
            pynvml.nvmlShutdown()
            self.output_device = self.args.gpus[0]

            self.device = 'gpu:{}'.format(self.output_device)
            paddle.device.set_device(self.device)

        else:
            logging.info('Using CPU!')
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            self.device =  paddle.device.set_device('cpu')

    def init_dataloader(self):
        dataset_name = self.args.dataset.split('-')[0]
        dataset_args = self.args.dataset_args[dataset_name]
        dataset_args['debug'] = self.args.debug
        self.train_batch_size = dataset_args['train_batch_size']
        self.eval_batch_size = dataset_args['eval_batch_size']
        self.feeders, self.data_shape, self.num_class, self.A, self.parts = dataset.create(
            self.args.dataset, **dataset_args
        )
        self.train_loader = paddle.io.DataLoader(self.feeders['train'],
            batch_size=self.train_batch_size, 
            shuffle=True, drop_last=True
        )
        self.eval_loader = paddle.io.DataLoader(self.feeders['eval'],
            batch_size=self.eval_batch_size, 
            shuffle=False, drop_last=False
        )
        self.location_loader = self.feeders['location'] if dataset_name == 'ntu' else None
        logging.info('Dataset: {}'.format(self.args.dataset))
        logging.info('Batch size: train-{}, eval-{}'.format(self.train_batch_size, self.eval_batch_size))
        logging.info('Data shape (branch, channel, frame, joint, person): {}'.format(self.data_shape))
        logging.info('Number of action classes: {}'.format(self.num_class))

    def init_model(self):
        kwargs = {
            'data_shape': self.data_shape,
            'num_class': self.num_class,
            'A': paddle.to_tensor(self.A),
            'parts': self.parts,
        }
        self.model = framework.create(self.args.model_type, **(self.args.model_args), **kwargs)
        logging.info('Model: {} {}'.format(self.args.model_type, self.args.model_args))
        with open('{}/model.txt'.format(self.save_dir), 'w') as f:
            print(self.model, file=f)

        pretrained_model = '{}/{}.pdparams.tar'.format(self.args.pretrained_path, self.model_name)
        if os.path.exists(pretrained_model):
            checkpoint = paddle.load(pretrained_model)
            self.model.load_dict(checkpoint['model'])
            self.cm = checkpoint['best_state']['cm']
            logging.info('Pretrained model: {}'.format(pretrained_model))
        elif self.args.pretrained_path:
            logging.warning('Warning: Do NOT exist this pretrained model: {}!'.format(pretrained_model))
            logging.info('Create model randomly.')

    def init_optimizer(self):
        try:
            if self.args.optimizer == 'SGD':
                optimizer = U.import_class('paddle.optimizer.Momentum')
            else:
                optimizer = U.import_class('paddle.optimizer.{}'.format(self.args.optimizer))

        except:
            logging.warning('Warning: Do NOT exist this optimizer: {}!'.format(self.args.optimizer))
            logging.info('Try to use SGD optimizer.')
            self.args.optimizer = 'SGD'

            optimizer = U.import_class('paddle.optimizer.Momentum')

        optimizer_args = self.args.optimizer_args[self.args.optimizer]
        if self.args.optimizer =='SGD':
            optimizer_args['learning_rate'] = self.scheduler
            optimizer_args['use_nesterov'] = optimizer_args['nesterov']
            del optimizer_args['nesterov']
            del optimizer_args['lr']
        elif self.args.optimizer == 'Adam':
            optimizer_args['learning_rate'] = self.scheduler
            optimizer_args['beta1'] = optimizer_args['betas'][0]
            optimizer_args['beta2'] = optimizer_args['betas'][1]
            del optimizer_args['lr']
            del optimizer_args['betas']
        self.optimizer = optimizer(parameters=self.model.parameters(), **optimizer_args)
            
        logging.info('Optimizer: {} {}'.format(self.args.optimizer, optimizer_args))

    def init_lr_scheduler(self):
        scheduler_args = self.args.scheduler_args[self.args.lr_scheduler]
        self.max_epoch = scheduler_args['max_epoch']
        lr_scheduler = solver.create(self.args.lr_scheduler, len(self.train_loader), **scheduler_args)
        self.eval_interval, lr_lambda = lr_scheduler.get_lambda()

        self.scheduler = paddle.optimizer.lr.LambdaDecay(\
            self.args.optimizer_args[self.args.optimizer]['lr'], lr_lambda=lr_lambda)

        logging.info('LR_Scheduler: {} {}'.format(self.args.lr_scheduler, scheduler_args))

    def init_loss_func(self):

        self.loss_func = paddle.nn.CrossEntropyLoss()
        logging.info('Loss function: {}'.format(self.loss_func.__class__.__name__))


class Processor(Initializer):

    def train(self, epoch):
        paddle.device.set_device('gpu:0')
        self.model.train()
        start_train_time = time()
        num_top1, num_sample = 0, 0
        train_iter = self.train_loader if self.no_progress_bar else tqdm(self.train_loader, dynamic_ncols=True)

        for num, (x, y, _) in enumerate(train_iter):
            self.optimizer.clear_grad()

            # Using GPU
            x = paddle.cast(x, 'float32')
            y = paddle.cast(y, 'int64')

            out, _ = self.model(x)
            
            # Updating Weights
            loss = self.loss_func(out, y)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            # Calculating Recognition Accuracies         
            num_top1, num_sample = metrics_train(x, out, y, num_sample, num_top1)
            
            # Showing Progress
            lr = self.optimizer.get_lr()
            
            if self.scalar_writer:
                self.scalar_writer.add_scalar('learning_rate', lr, self.global_step)
                self.scalar_writer.add_scalar('train_loss', loss.item(), self.global_step)
            
            if self.no_progress_bar:
                logging.info('Epoch: {}/{}, Batch: {}/{}, Loss: {:.4f}, LR: {:.4f}'.format(
                    epoch+1, self.max_epoch, num+1, len(self.train_loader), loss.item(), lr
                ))
            else:
                train_iter.set_description('Loss: {:.4f}, LR: {:.4f}'.format(loss.item(), lr))
        train_acc = num_top1 / num_sample
        if self.scalar_writer:
            self.scalar_writer.add_scalar('train_acc', train_acc, self.global_step)
        
        logging.info('Epoch: {}/{}, Training accuracy: {:d}/{:d}({:.2%}), Training time: {:.2f}s'.format(
            epoch+1, self.max_epoch, int(num_top1), num_sample, train_acc, time()-start_train_time
        ))
        logging.info('')

    def eval(self):

        self.model.eval()
        start_eval_time = time()
        with paddle.no_grad():
            
            num_top1, num_top5 = 0, 0
            num_sample, eval_loss = 0, []
            cm = np.zeros((self.num_class, self.num_class))
            eval_iter = self.eval_loader if self.no_progress_bar else tqdm(self.eval_loader, dynamic_ncols=True)
            for num, (x, y, _) in enumerate(eval_iter):

                # Using GPU
                x = paddle.cast(x, 'float32')
                y = paddle.cast(y, 'int64')
                # Calculating Output
                out, _ = self.model(x)
                # Getting Loss
                loss = self.loss_func(out, y)
                eval_loss.append(loss.item())
                # Calculating Recognition Accuracies
                num_top1, num_top5, num_sample, cm = metrics_eval(x, out, y, num_sample, num_top1, num_top5, cm)
                # Showing Progress
                if self.no_progress_bar and self.args.evaluate:
                    logging.info('Batch: {}/{}'.format(num+1, len(self.eval_loader)))

        acc_top1 = num_top1 / num_sample
        acc_top5 = num_top5 / num_sample
        eval_loss = sum(eval_loss) / len(eval_loss)
        eval_time = time() - start_eval_time
        eval_speed = len(self.eval_loader) * self.eval_batch_size / eval_time / len(self.args.gpus)
        logging.info('Top-1 accuracy: {:d}/{:d}({:.2%}), Top-5 accuracy: {:d}/{:d}({:.2%}), Mean loss:{:.4f}'.format(
            int(num_top1), num_sample, acc_top1, num_top5, num_sample, acc_top5, eval_loss
        ))
        logging.info('Evaluating time: {:.2f}s, Speed: {:.2f} sequnces/(second*GPU)'.format(
            eval_time, eval_speed
        ))
        logging.info('')
        if self.scalar_writer:
            self.scalar_writer.add_scalar('eval_acc', acc_top1, self.global_step)
            self.scalar_writer.add_scalar('eval_loss', eval_loss, self.global_step)

        paddle.device.cuda.empty_cache()

        return acc_top1, acc_top5, cm

    def start(self):
        start_time = time()
        self.losslist = []
        if self.args.evaluate:
            if self.args.debug:
                logging.warning('Warning: Using debug setting now!')
                logging.info('')

            # Loading Evaluating Model
            logging.info('Loading evaluating model ...')

            checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name)
            
            if checkpoint:
                self.model.set_state_dict(checkpoint['model'])

            
            logging.info('Successful!')
            logging.info('')

            # Evaluating
            logging.info('Starting evaluating ...')
            self.eval()
            logging.info('Finish evaluating!')

        else:
            # Resuming
            start_epoch = 0
            best_state = {'acc_top1':0, 'acc_top5':0, 'cm':0}
            if self.args.resume:
                logging.info('Loading checkpoint ...')
                checkpoint = U.load_checkpoint(self.args.work_dir)
                self.model.set_state_dict(checkpoint['model'])
                self.optimizer.set_state_dict(checkpoint['optimizer'])
                self.scheduler.set_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch']
                best_state.update(checkpoint['best_state'])
                self.global_step = start_epoch * len(self.train_loader)
                logging.info('Start epoch: {}'.format(start_epoch+1))
                logging.info('Best accuracy: {:.2%}'.format(best_state['acc_top1']))
                logging.info('Successful!')
                logging.info('')

            # Training
            self.max_epoch = 80
            logging.info('Starting training ...')
            for epoch in range(start_epoch, self.max_epoch):
                self.train(epoch)

                # Evaluating
                is_best = False
                if (epoch+1) % self.eval_interval(epoch) == 0:
                    logging.info('Evaluating for epoch {}/{} ...'.format(epoch+1, self.max_epoch))
                    acc_top1, acc_top5, cm = self.eval()
                    if acc_top1 > best_state['acc_top1']:
                        is_best = True
                        best_state.update({'acc_top1':acc_top1, 'acc_top5':acc_top5, 'cm':cm})

                # Saving Model
                logging.info('Saving model for epoch {}/{} ...'.format(epoch+1, self.max_epoch))
                U.save_checkpoint(
                    self.model.state_dict(), self.optimizer.state_dict(), self.scheduler.state_dict(),
                    epoch+1, best_state, is_best, self.args.work_dir, self.save_dir, self.model_name
                )

                logging.info('Best top-1 accuracy: {:.2%}, Total time: {}'.format(
                    best_state['acc_top1'], U.get_time(time()-start_time)
                ))
                logging.info('')
            logging.info('Finish training!')
            logging.info('')


    def extract(self):
        logging.info('Starting extracting ...')
        if self.args.debug:
            logging.warning('Warning: Using debug setting now!')
            logging.info('')

        # Loading Model
        logging.info('Loading evaluating model ...')
        checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name)
        if checkpoint:
            self.cm = checkpoint['best_state']['cm']
            self.model.load_dict(checkpoint['model'])
        logging.info('Successful!')
        logging.info('')

        # Loading Data
        x, y, names = iter(self.eval_loader).next()
        location = self.location_loader.load(names) if self.location_loader else []

        # Calculating Output
        self.model.eval()
        paddle.device.set_device(self.device)
        out, feature = self.model(x.float())

        # Processing Data
        data, label = x.numpy(), y.numpy()
        out = paddle.nn.functional.softmax(out, axis=1).detach().cpu().numpy()
        weight = self.model.module.classifier.fc.weight.squeeze().detach().cpu().numpy()


        feature = feature.detach().cpu().numpy()

        # Saving Data
        if not self.args.debug:
            U.create_folder('./visualization')
            np.savez('./visualization/extraction_{}.npz'.format(self.args.config),
                data=data, label=label, name=names, out=out, cm=self.cm,
                feature=feature, weight=weight, location=location
            )
        logging.info('Finish extracting!')
        logging.info('')

class Visualizer():
    def __init__(self, args):
        self.args = args
        U.set_logging(args)
        logging.info('')
        logging.info('Starting visualizing ...')

        self.action_names = {}
        self.action_names['ntu'] = [
            'drink water 1', 'eat meal/snack 2', 'brushing teeth 3', 'brushing hair 4', 'drop 5', 'pickup 6',
            'throw 7', 'sitting down 8', 'standing up 9', 'clapping 10', 'reading 11', 'writing 12',
            'tear up paper 13', 'wear jacket 14', 'take off jacket 15', 'wear a shoe 16', 'take off a shoe 17',
            'wear on glasses 18','take off glasses 19', 'put on a hat/cap 20', 'take off a hat/cap 21', 'cheer up 22',
            'hand waving 23', 'kicking something 24', 'put/take out sth 25', 'hopping 26', 'jump up 27',
            'make a phone call 28', 'playing with a phone 29', 'typing on a keyboard 30',
            'pointing to sth with finger 31', 'taking a selfie 32', 'check time (from watch) 33',
            'rub two hands together 34', 'nod head/bow 35', 'shake head 36', 'wipe face 37', 'salute 38',
            'put the palms together 39', 'cross hands in front 40', 'sneeze/cough 41', 'staggering 42', 'falling 43',
            'touch head 44', 'touch chest 45', 'touch back 46', 'touch neck 47', 'nausea or vomiting condition 48',
            'use a fan 49', 'punching 50', 'kicking other person 51', 'pushing other person 52',
            'pat on back of other person 53', 'point finger at the other person 54', 'hugging other person 55',
            'giving sth to other person 56', 'touch other person pocket 57', 'handshaking 58',
            'walking towards each other 59', 'walking apart from each other 60'
        ]
        self.action_names['cmu'] = [
            'walking 1', 'running 2', 'directing_traffic 3', 'soccer 4',
            'basketball 5', 'washwindow 6', 'jumping 7', 'basketball_signal 8'
        ]

        self.font_sizes = {
            'ntu': 6,
            'cmu': 14,
        }


    def start(self):
        self.read_data()
        logging.info('Please select visualization function from follows: ')
        logging.info('1) wrong sample (ws), 2) important joints (ij), 3) NTU skeleton (ns),')
        logging.info('4) confusion matrix (cm), 5) action accuracy (ac)')
        while True:
            logging.info('Please input the number (or name) of function, q for quit: ')
            cmd = input(U.get_current_timestamp())
            if cmd in ['q', 'quit', 'exit', 'stop']:
                break
            elif cmd == '1' or cmd == 'ws' or cmd == 'wrong sample':
                self.show_wrong_sample()
            elif cmd == '2' or cmd == 'ij' or cmd == 'important joints':
                self.show_important_joints()
            elif cmd == '3' or cmd == 'ns' or cmd == 'NTU skeleton':
                self.show_NTU_skeleton()
            elif cmd == '4' or cmd == 'cm' or cmd == 'confusion matrix':
                self.show_confusion_matrix()
            elif cmd == '5' or cmd == 'ac' or cmd == 'action accuracy':
                self.show_action_accuracy()
            else:
                logging.info('Can not find this function!')
                logging.info('')


    def read_data(self):
        logging.info('Reading data ...')
        logging.info('')
        data_file = './visualization/extraction_{}.npz'.format(self.args.config)
        try:
            data = np.load(data_file)
        except:
            data = None
            logging.info('')
            logging.error('Error: Wrong in loading this extraction file: {}!'.format(data_file))
            logging.info('Please extract the data first!')
            raise ValueError()
        logging.info('*********************Video Name************************')
        logging.info(data['name'][self.args.visualization_sample])
        logging.info('')

        feature = data['feature'][self.args.visualization_sample]
        self.location = data['location']
        if len(self.location) > 0:
            self.location = self.location[self.args.visualization_sample]
        self.data = data['data'][self.args.visualization_sample]
        self.label = data['label']
        weight = data['weight']
        out = data['out']
        cm = data['cm']
        self.cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]

        dataset = self.args.dataset.split('-')[0]
        self.names = self.action_names[dataset]
        self.font_size = self.font_sizes[dataset]

        self.pred = np.argmax(out, 1)
        self.pred_class = self.pred[self.args.visualization_sample] + 1
        self.actural_class = self.label[self.args.visualization_sample] + 1
        if self.args.visualization_class == 0:
            self.args.visualization_class = self.actural_class
        self.probablity = out[self.args.visualization_sample, self.args.visualization_class-1]
        self.result = np.einsum('kc,ctvm->ktvm', weight, feature)   # CAM method
        self.result = self.result[self.args.visualization_class-1]


    def show_action_accuracy(self):
        cm = self.cm.round(4)

        logging.info('Accuracy of each class:')
        accuracy = cm.diagonal()
        for i in range(len(accuracy)):
            logging.info('{}: {}'.format(self.names[i], accuracy[i]))
        logging.info('')

        plt.figure()
        plt.bar(self.names, accuracy, align='center')
        plt.xticks(fontsize=10, rotation=90)
        plt.yticks(fontsize=10)
        plt.show()


    def show_confusion_matrix(self):
        cm = self.cm.round(2)
        show_name_x = range(1,len(self.names)+1)
        show_name_y = self.names

        plt.figure()
        font_size = self.font_size
        sns.heatmap(cm, cmap=plt.cm.Blues, annot=True, annot_kws={'fontsize':font_size-2}, cbar=False,
                    square=True, linewidths=0.1, linecolor='black', xticklabels=show_name_x, yticklabels=show_name_y)
        plt.xticks(fontsize=font_size, rotation=0)
        plt.yticks(fontsize=font_size)
        plt.xlabel('Index of Predict Classes', fontsize=font_size)
        plt.ylabel('Index of True Classes', fontsize=font_size)
        plt.show()


    def show_NTU_skeleton(self):
        if len(self.location) == 0:
            logging.info('This function is only for NTU dataset!')
            logging.info('')
            return

        C, T, V, M = self.location.shape
        connecting_joint = np.array([2,1,21,3,21,5,6,7,21,9,10,11,1,13,14,15,1,17,18,19,2,23,8,25,12])
        result = np.maximum(self.result, 0)
        result = result/np.max(result)

        if len(self.args.visualization_frames) > 0:
            pause, frames = 10, self.args.visualization_frames
        else:
            pause, frames = 0.1, range(self.location.shape[1])

        plt.figure()
        plt.ion()
        for t in frames:
            if np.sum(self.location[:,t,:,:]) == 0:
                break

            plt.cla()
            plt.xlim(-50, 2000)
            plt.ylim(-50, 1100)
            plt.axis('off')
            plt.title('sample:{}, class:{}, frame:{}\n probablity:{:2.2f}%, pred_class:{}, actural_class:{}'.format(
                self.args.visualization_sample, self.names[self.args.visualization_class-1],
                t, self.probablity*100, self.pred_class, self.actural_class
            ))

            for m in range(M):
                x = self.location[0,t,:,m]
                y = 1080 - self.location[1,t,:,m]

                c = []
                for v in range(V):
                    r = result[t//4,v,m]
                    g = 0
                    b = 1 - r
                    c.append([r,g,b])
                    k = connecting_joint[v] - 1
                    plt.plot([x[v],x[k]], [y[v],y[k]], '-o', c=np.array([0.1,0.1,0.1]), linewidth=0.5, markersize=0)
                plt.scatter(x, y, marker='o', c=c, s=16)
            plt.pause(pause)
        plt.ioff()
        plt.show()


    def show_wrong_sample(self):
        wrong_sample = []
        for i in range(len(self.pred)):
            if not self.pred[i] == self.label[i]:
                wrong_sample.append(i)
        logging.info('*********************Wrong Sample**********************')
        logging.info(wrong_sample)
        logging.info('')


    def show_important_joints(self):
        first_sum = np.sum(self.result[:,:,0], axis=0)
        first_index = np.argsort(-first_sum) + 1
        logging.info('*********************First Person**********************')
        logging.info('Weights of all joints:')
        logging.info(first_sum)
        logging.info('')
        logging.info('Most important joints:')
        logging.info(first_index)
        logging.info('')

        if self.result.shape[-1] > 1:
            second_sum = np.sum(self.result[:,:,1], axis=0)
            second_index = np.argsort(-second_sum) + 1
            logging.info('*********************Second Person*********************')
            logging.info('Weights of all joints:')
            logging.info(second_sum)
            logging.info('')
            logging.info('Most important joints:')
            logging.info(second_index)
            logging.info('')
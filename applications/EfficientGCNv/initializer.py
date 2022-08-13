import os, yaml, warnings, logging, paddle, numpy as np
from copy import deepcopy
from paddle.optimizer.lr import LambdaDecay
from paddle.io import DataLoader

import utils as U
import dataset
import model
import scheduler
class Initializer():
    def __init__(self, args):
        self.args = args
        self.init_save_dir()
        logging.info('')
        logging.info('Starting preparing ...')
        self.init_save_dir()
        self.init_environment()
        self.init_dataloader()
        self.init_model()
        self.init_lr_scheduler()
        self.init_optimizer()
        self.init_loss_func()
        logging.info('Successful!')
        logging.info('')
    def init_environment(self):
        paddle.seed = self.args.seed
        self.global_step = 0
        if self.args.debug:
            self.no_progress_bar = True
            self.model_name = 'debug'
        elif self.args.evaluate or self.args.extract:
            self.no_progress_bar = self.args.no_progress_bar
            self.model_name = self.args.pretrained_path
            warnings.filterwarnings('ignore')
        else:
            self.no_progress_bar = self.args.no_progress_bar
            self.model_name = self.args.pretrained_path
            warnings.filterwarnings('ignore')
        logging.info('Saving model name: {}'.format(self.model_name))
    def init_save_dir(self):
        self.save_dir = U.set_logging(self.args)
        with open('{}/config.yaml'.format(self.save_dir), 'w') as f:
            yaml.dump(vars(self.args), f)
        logging.info('Saving folder path: {}'.format(self.save_dir))

    def init_dataloader(self):
        dataset_name = self.args.dataset.split('-')[0]
        dataset_args = self.args.dataset_args[dataset_name]
        dataset_args['debug'] = self.args.debug
        self.train_batch_size = dataset_args['train_batch_size']
        self.eval_batch_size = dataset_args['eval_batch_size']
        self.feeders, self.data_shape, self.num_class, self.A, self.parts = dataset.create(
            self.args.dataset, **dataset_args
        )
        self.train_loader = DataLoader(
            self.feeders['train'],
            batch_size=self.train_batch_size,
            # num_workers=4 * len(self.args.gpus),
            shuffle=True,
            drop_last=True
        )
        self.eval_loader = DataLoader(
            self.feeders['eval'],
            batch_size=self.eval_batch_size,
            # num_workers=4 * len(self.args.gpus),
            shuffle=True,
            drop_last=True
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
        self.model = model.create(self.args.model_type, **(self.args.model_args), **kwargs)
        logging.info('Model: {} {}'.format(self.args.model_type, self.args.model_args))
        with open('{}/model.txt'.format(self.save_dir), 'w') as f:
            print(self.model, file=f)


    def init_optimizer(self):
        try:
             optimizer = U.import_class('paddle.optimizer.{}'.format(self.args.optimizer))
        except:
            logging.warning('Warning: Do NOT exist this optimizer: {}!'.format(self.args.optimizer))
            logging.info('Try to use SGD optimizer.')
            self.args.optimizer = 'SGD'
            optimizer = U.import_class('paddle.optimizer.SGD')
        optimizer_args = self.args.optimizer_args[self.args.optimizer]
        optimizer_args["learning_rate"] = self.scheduler
        self.optimizer = optimizer(parameters=self.model.parameters(), **optimizer_args)
        logging.info('Optimizer: {} {}'.format(self.args.optimizer, optimizer_args))

    def init_lr_scheduler(self):
        scheduler_args = self.args.scheduler_args[self.args.lr_scheduler]
        self.max_epoch = scheduler_args['max_epoch']
        lr_scheduler = scheduler.create(self.args.lr_scheduler, len(self.train_loader), **scheduler_args)
        self.eval_interval, lr_lambda = lr_scheduler.get_lambda()
        self.scheduler = LambdaDecay(learning_rate=self.args.optimizer_args[self.args.optimizer]["learning_rate"], lr_lambda=lr_lambda)
        logging.info('LR_Scheduler: {} {}'.format(self.args.lr_scheduler, scheduler_args))
    def init_loss_func(self):
        self.loss_func = paddle.nn.CrossEntropyLoss()
        logging.info('Loss function: {}'.format(self.loss_func.__class__.__name__))










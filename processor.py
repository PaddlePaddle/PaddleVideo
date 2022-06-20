import logging, paddle, numpy as np
from tqdm import tqdm
from time import time
import os

import utils as U
from initializer import Initializer

class Proccessor(Initializer):

    def train(self, epoch):
        self.model.train()
        start_train_time = time()
        num_top1, num_sample = 0, 0
        train_iter = self.train_loader if self.no_progress_bar else tqdm(self.train_loader, dynamic_ncols=True)
        for num, (x, y, _) in enumerate(train_iter):
            self.optimizer.clear_grad()

            x = paddle.cast(x, dtype='float32')
            out, _ = self.model(x)
            loss = self.loss_func(out, y)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            num_sample += x.shape[0]
            reco_top1 = out.argmax(axis=1)
            num_top1 += reco_top1.equal(y).sum().item()

            lr = self.optimizer.get_lr()
            if self.no_progress_bar:
                logging.info('Epoch: {}/{}, Batch: {}/{}, Loss: {:.4f}, LR: {:.4f}'.format(
                    epoch+1, self.max_epoch, num+1, len(self.train_loader), loss.item(), lr
                ))
            else:
                train_iter.set_description('Loss: {:.4f}, LR: {:.4f}'.format(loss.item(), lr))
        
        # Showing Train Results
        train_acc = num_top1 / num_sample
        logging.info('Epoch: {}/{}, Training accuracy: {:d}/{:d}({:.2%}), Training time: {:.2f}s'.format(
            epoch+1, self.max_epoch, num_top1, num_sample, train_acc, time()-start_train_time
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
                x = paddle.cast(x, dtype='float32')
               
                out, _ = self.model(x)
                
                loss = self.loss_func(out, y)
                eval_loss.append(loss.item())

                num_sample += x.shape[0]
                reco_top1 = out.argmax(axis=1)
                num_top1 += reco_top1.equal(y).sum().item()
                values, reco_top5 = paddle.topk(out, 5, 1) 
                num_top5 += sum([y[n] in reco_top5[n,:] for n in range(x.shape[0])])
                for i in range(x.shape[0]):
                    cm[y[i], reco_top1[i]] += 1
                if self.no_progress_bar and self.args.evaluate:
                    logging.info('Batch: {}/{}'.format(num+1, len(self.eval_loader)))
        
        acc_top1 = num_top1 / num_sample
        acc_top5 = num_top5 / num_sample
        eval_loss = sum(eval_loss) / len(eval_loss)

        eval_time = time() - start_eval_time
        eval_speed = len(self.eval_loader) * self.eval_batch_size / eval_time / len(self.args.gpus)
        logging.info('Top-1 accuracy: {:d}/{:d}({:.2%}), Top-5 accuracy: {:d}/{:d}({:.2%}), Mean loss:{:.4f}'.format(
            num_top1, num_sample, acc_top1, num_top5, num_sample, acc_top5, eval_loss
        ))
        logging.info('Evaluating time: {:.2f}s, Speed: {:.2f} sequnces/(second*GPU)'.format(
            eval_time, eval_speed
        ))
        logging.info('')

        # torch.cuda.empty_cache()
        return acc_top1, acc_top5, cm
    
    def start(self):
        start_time = time()
        if self.args.evaluate:
            if self.args.debug:
                logging.warning('Warning: Using debug setting now!')
                logging.info('')

            # Loading Evaluating Model
            logging.info('Loading evaluating model ...')
            
            checkpoint = paddle.load(self.model_name)
            if checkpoint:
                self.model.set_state_dict(checkpoint['model'])
            logging.info('Successful!')
            logging.info('')
            
            # Evaluating
            logging.info('Starting evaluating ...')
            self.eval()
            logging.info('Finish evaluating!')

        else:
            start_epoch = 0
            best_state = {'acc_top1':0, 'acc_top5':0, 'cm':0}
            if self.args.resume:
                # Loading Evaluating Model
                logging.info('Loading pretrain model ...')
                
                checkpoint = paddle.load(self.model_name)
                if checkpoint:
                    self.model.set_state_dict(checkpoint['model'])
                    self.scheduler.set_state_dict(checkpoint['scheduler'])
                    self.optimizer.set_state_dict(checkpoint['opt'])
                    start_epoch = checkpoint['epoch']
                    best_state.update(checkpoint['best_state'])
                logging.info('Successful!')
                logging.info('')

            # Training
            logging.info('Starting training ...')
            for epoch in range(start_epoch, self.max_epoch):

                self.train(epoch)

                is_best = False
                if (epoch+1) % self.eval_interval(epoch) == 0:
                    logging.info('Evaluating for epoch {}/{} ...'.format(epoch+1, self.max_epoch))
                    acc_top1, acc_top5, cm = self.eval()
                    if acc_top1 > best_state['acc_top1']:
                        is_best = True
                        best_state.update({'acc_top1':acc_top1, 'acc_top5':acc_top5, 'cm':cm})
                logging.info('Saving model for epoch {}/{} ...'.format(epoch+1, self.max_epoch))  
                obj = {'model': self.model.state_dict(), 'opt': self.optimizer.state_dict(), 'epoch': epoch, 'best_state': best_state, 'scheduler': self.scheduler.state_dict()}
                if epoch != self.max_epoch - 1:
                    paddle.save(obj, '{}/{}/{}.pdparams'.format(self.args.work_dir, epoch+1, self.max_epoch))
                else:
                    paddle.save(obj, '{}/model.pdparams'.format(self.args.work_dir))
                logging.info('Best top-1 accuracy: {:.2%}, Total time: {}'.format(
                    best_state['acc_top1'], U.get_time(time()-start_time)
                ))
                logging.info('')


            logging.info('Finish training!')
            logging.info('')    

    def extract(self):
        if self.args.debug:
            logging.warning('Warning: Using debug setting now!')
            logging.info('')

        # Loading Evaluating Model
        logging.info('Loading model ...')
        checkpoint = paddle.load(self.model_name)
        if checkpoint:
            self.model.set_state_dict(checkpoint['model'])          
        self.data_shape.insert(0, self.eval_batch_size)
        self.model.eval()
        new_net = paddle.jit.to_static(self.model, input_spec=[paddle.static.InputSpec(shape=self.data_shape, dtype='float32')])
        save_path = self.model_name.replace('.pdparams', "")
        paddle.jit.save(new_net, save_path)
        logging.info('Successful!')
        logging.info('')

    


        

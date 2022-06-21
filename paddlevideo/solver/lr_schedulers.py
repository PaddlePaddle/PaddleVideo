import numpy as np


class Step_Scheduler():
    def __init__(self, num_sample, warm_up, step_lr, **kwargs):
        warm_up_num = warm_up * num_sample
        self.eval_interval = lambda epoch: 1 if (epoch+1) > step_lr[-1] else 5
        self.lr_lambda = lambda num: num / warm_up_num \
                                     if num < warm_up_num else \
                                     0.1 ** np.sum(np.array(step_lr) <= num // num_sample)

    def get_lambda(self):
        return self.eval_interval, self.lr_lambda


class Cosine_Scheduler():
    def __init__(self, num_sample, max_epoch, warm_up, **kwargs):
        warm_up_num = warm_up * num_sample
        max_num = max_epoch * num_sample
        self.eval_interval = lambda epoch: 1 if (epoch+1) > max_epoch - 10 else 5
        self.lr_lambda = lambda num: num / warm_up_num \
                                     if num < warm_up_num else \
                                     0.5 * (np.cos((num - warm_up_num) / (max_num - warm_up_num) * np.pi) + 1)

    def get_lambda(self):
        return self.eval_interval, self.lr_lambda

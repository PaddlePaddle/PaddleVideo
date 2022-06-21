import logging

from .lr_schedulers import *


__scheduler = {
    'step': Step_Scheduler,
    'cosine': Cosine_Scheduler,
}

def create(lr_scheduler, num_sample, **kwargs):
    if lr_scheduler not in __scheduler.keys():
        logging.info('')
        logging.error('Error: Do NOT exist this lr_scheduler: {}!'.format(lr_scheduler))
        raise ValueError()
    return __scheduler[lr_scheduler](num_sample, **kwargs)

import logging

import utils as U
import reader


class Generator():
    def __init__(self, args):
        U.set_logging(args)
        self.dataset = args.dataset
        self.generator = reader.create(args)

    def start(self):
        logging.info('')
        logging.info('Starting generating ...')
        logging.info('Dataset: {}'.format(self.dataset))
        self.generator.start()
        logging.info('Finish generating!')
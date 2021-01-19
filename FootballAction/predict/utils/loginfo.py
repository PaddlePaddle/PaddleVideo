"""
log for spider
"""
import os
import logging


class Logger(logging.Logger):
    """logger

    initialize log class

    Attributes:
        log_path: Log file path prefix.
        level: msg above the level will be displayed
    """
    def __init__(self, log_path="log/algo.log", level=logging.INFO):
        """
        Inits Logger
        """
        super(Logger, self).__init__(self)
        # msg format
        format = "%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s"
        datefmt = "%m-%d %H:%M:%S"

        # make the output log folder
        dir = os.path.dirname(log_path)
        if not os.path.isdir(dir):
            os.makedirs(dir)

        handle = logging.FileHandler(filename=log_path)
        handle.setLevel(level)
        formatter = logging.Formatter(format, datefmt)
        handle.setFormatter(formatter)
        self.addHandler(handle)

"""
config parser
"""

try:
    from configparser import ConfigParser
except BaseException:
    from ConfigParser import ConfigParser

from utils import AttrDict

import logging
logger = logging.getLogger(__name__)

CONFIG_SECS = [
    'train',
    'valid',
    'test',
    'infer',
]


def parse_config(cfg_file):
    """parse_config
    """
    parser = ConfigParser()
    cfg = AttrDict()
    parser.read(cfg_file)
    for sec in parser.sections():
        sec_dict = AttrDict()
        for k, v in parser.items(sec):
            try:
                v = eval(v)
            except BaseException:
                pass
            setattr(sec_dict, k, v)
        setattr(cfg, sec.upper(), sec_dict)

    return cfg


def merge_configs(cfg, sec, args_dict):
    """merge_configs
    """
    assert sec in CONFIG_SECS, "invalid config section {}".format(sec)
    sec_dict = getattr(cfg, sec.upper())
    for k, v in args_dict.items():
        if v is None:
            continue
        # try:
        #     if hasattr(sec_dict, k):
        #         setattr(sec_dict, k, v)
        # except BaseException:
        #     pass
        if k in sec_dict:
            setattr(sec_dict, k, v)
    return cfg

def print_configs(cfg, mode):
    """print_configs
    """
    logger.info("---------------- {:>5} Arguments ----------------".format(mode))
    for sec, sec_items in cfg.items():
        if isinstance(sec_items, dict) is True:
            logger.info("{}:".format(sec))
            for k, v in sec_items.items():
                logger.info("    {}:{}".format(k, v))
        else:
            logger.info("{}:{}".format(sec, sec_items))
            
    logger.info("-------------------------------------------------")

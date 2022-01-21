# Copyright 2021 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import paddle
import pprint
import logging
from typing import Dict
from pathlib import Path
from datetime import datetime
from operator import getitem
from functools import reduce

from mergedeep import Strategy, merge
from zsvision.zs_utils import set_nested_key_val
from typeguard import typechecked

from utils import read_json, write_json
from logger import setup_logging


class ConfigParser:
    def __init__(self, args, options='', timestamp=True, slave_mode=False):
        # slave_mode - when calling the config parser form an existing process, we
        # avoid reinitialising the logger and ignore sys.argv when argparsing.

        # parse default and custom cli options
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)

        if slave_mode:
            args = args.parse_args(args=[])
        else:
            args = args.parse_args()

        if args.resume and not slave_mode:
            self.resume = Path(args.resume)
        else:
            msg_no_cfg = "Config file must be specified"
            assert args.config is not None, msg_no_cfg
            self.resume = None
        self.cfg_fname = Path(args.config)

        config = self.load_config(self.cfg_fname)
        self._config = _update_config(config, options, args)

        if self._config.get("eval_config", False):
            # validate path to evaluation file
            eval_cfg_path = self._config.get("eval_config")
            msg = f"eval_config was specified, but `{eval_cfg_path}` does not exist"
            assert Path(self._config.get("eval_config")).exists(), msg

        # set save_dir where trained model and log will be saved.
        if "tester" in self.config:
            save_dir = Path(self.config['tester']['save_dir'])
        else:
            save_dir = Path(self.config['trainer']['save_dir'])
        timestamp = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S") if timestamp else ""

        if slave_mode:
            timestamp = f"{timestamp}-eval-worker"

        exper_name = self.set_exper_name(args, config=config)

        if getattr(args, "group_id", False):
            subdir = Path(args.group_id) / f"seed-{args.group_seed}" / timestamp
        else:
            subdir = timestamp

        self._save_dir = save_dir / 'models' / exper_name / subdir
        self._log_dir = save_dir / 'log' / exper_name / subdir
        self._exper_name = exper_name
        self._args = args

        # if set, remove all previous experiments with the current config
        if vars(args).get("purge_exp_dir", False):
            for dirpath in (self._save_dir, self._log_dir):
                config_dir = dirpath.parent
                existing = list(config_dir.glob("*"))
                print(f"purging {len(existing)} directories from config_dir...")
                tic = time.time()
                os.system(f"rm -rf {config_dir}")
                print(f"Finished purge in {time.time() - tic:.3f}s")

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / 'config.json')

        # configure logging module
        if not slave_mode:
            self.log_path = setup_logging(self.log_dir)

        self.log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

    def set_exper_name(self, args, config):
        # We assume that the config files are organised into directories such that
        # each directory has the name of the dataset.
        dataset_name = self.cfg_fname.parent.stem
        exper_name = f"{dataset_name}-{self.cfg_fname.stem}"
        if args.custom_args:
            key_val_lists = args.custom_args.split("+")
            for key_val_pair in key_val_lists:
                print(f"parsing key-val pair : {key_val_pair}")
                key, val = key_val_pair.split("@")
                set_nested_key_val(key, val, self._config)
                # remove periods from key names
                key_ = key.replace("_.", "--")
                # remove commas from value names
                val = val.replace(",", "--")
                custom_tag = "-".join(key_.split(".")[-2:])
                exper_name = f"{exper_name}-{custom_tag}-{val}"

        if getattr(args, "disable_workers", False):
            print("Disabling data loader workers....")
            config["data_loader"]["args"]["num_workers"] = 0

        if getattr(args, "train_single_epoch", False):
            print("Restricting training to a single epoch....")
            config["trainer"]["epochs"] = 1
            config["trainer"]["save_period"] = 1
            config["trainer"]["skip_first_n_saves"] = 0
            exper_name = f"{exper_name}-train-single-epoch"
        return exper_name

    @staticmethod
    @typechecked
    def load_config(cfg_fname: Path) -> Dict:
        config = read_json(cfg_fname)
        # apply inheritance through config hierarchy
        descendant, ancestors = config, []
        while "inherit_from" in descendant:
            parent_config = read_json(Path(descendant["inherit_from"]))
            ancestors.append(parent_config)
            descendant = parent_config
        for ancestor in ancestors:
            merge(ancestor, config, strategy=Strategy.REPLACE)
            config = ancestor
        return config

    def init(self, name, module, *args, **kwargs):
        """Finds a function handle with the name given as 'type' in config, and returns
        the instance initialized with corresponding keyword args given as 'args'.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        msg = (f"Fail for {module_name}\n"
               f"overwriting kwargs given in config file is not allowed\n"
               f"passed kwargs: {kwargs}\n"
               f"for module_args: {module_args})")
        assert all([k not in module_args for k in kwargs]), msg
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, name):
        return self.config[name]

    def __len__(self):
        # NOTE: This is used for boolean checking deep inside ray.tune, so we required it
        # to be defined.
        return len(self.config)

    def __setitem__(self, name, value):
        self.config[name] = value

    def __contains__(self, name):
        return name in self.config

    def get(self, name, default):
        return self.config.get(name, default)

    def keys(self):
        return self.config.keys()

    def get_logger(self, name, verbosity=2):
        msg_verbosity = "verbosity option {} is invalid. Valid options are {}."
        msg_verbosity = msg_verbosity.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

    def __repr__(self):
        return pprint.PrettyPrinter().pformat(self.__dict__)

    def items(self):
        return self._config.items()


# helper functions used to update config dict with custom cli options
def _update_config(config, options, args):
    for opt in options:
        value = getattr(args, _get_opt_name(opt.flags))
        if value is not None:
            _set_by_path(config, opt.target, value)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)

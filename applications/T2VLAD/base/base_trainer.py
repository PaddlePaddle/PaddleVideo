# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import re
import copy
import time
import paddle
import pickle
import numpy as np
from pathlib import Path
from abc import abstractmethod

class BaseTrainer:
    """ Base class for all trainers
    """
    def __init__(self, model, loss, metrics, optimizer, config, mini_train,
                 num_keep_ckpts, skip_tboard):
        self.config = config
        self.logger = config.get_logger(
            'trainer', config['trainer']['verbosity'])

        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.num_keep_ckpts = num_keep_ckpts
        self.skip_tboard = skip_tboard or mini_train

        # This property can be overriden in the subclass
        self.skip_first_n_saves = 0

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.save_only_best = cfg_trainer.get("save_only_best", True)
        self.val_freq = cfg_trainer['val_freq']
        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = np.inf if self.mnt_mode == 'min' else -np.inf
            self.early_stop = cfg_trainer.get('early_stop', np.inf)

        self.start_epoch = 1

        self.model_dir = config.save_dir

        self.include_optim_in_save_model = config["trainer"].get("include_optim_in_save_model", 1)
        if config.resume is not None:
            self._resume_model(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """Full training logic.  Responsible for iterating over epochs, early stopping,
        modeling and logging metrics.
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            result, cached_preds = self._train_epoch(epoch)

            if epoch % self.val_freq != 0:
                continue
            # save logged informations into log dict
            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__: value[i]
                                for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__: value[i]
                                for i, mtr in enumerate(self.metrics)})
                elif key == 'nested_val_metrics':
                    # NOTE: currently only supports two layers of nesting
                    for subkey, subval in value.items():
                        for subsubkey, subsubval in subval.items():
                            log[f"val_{subkey}_{subsubkey}"] = subsubval
                else:
                    log[key] = value

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # eval model according to configured metric, save best # ckpt as trained_model
            not_improved_count = 0
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether specified metric improved or not, according to
                    # specified metric(mnt_metric)
                    lower = log[self.mnt_metric] <= self.mnt_best
                    higher = log[self.mnt_metric] >= self.mnt_best
                    improved = (self.mnt_mode == 'min' and lower) or \
                               (self.mnt_mode == 'max' and higher)
                except KeyError:
                    msg = "Warning: Metric '{}' not found, perf monitoring is disabled."
                    self.logger.warning(msg.format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False
                    not_improved_count = 0
                    raise ValueError("Pick a metric that will save models!!!!!!!!")

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    # TODO(Samuel): refactor the code so that we don't move the model
                    # off the GPU or duplicate on the GPU (we should be able to safely
                    # copy the state dict directly to CPU)
                    copy_model = copy.deepcopy(self.model)
                    self.best_model = {"epoch": epoch, "model": copy_model}
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Val performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if self.save_only_best:
                if epoch == self.epochs:
                    best_model = self.best_model
                    self.model = best_model["model"]
                    print(f"saving the best model to disk (epoch {epoch})")
                    self._save_model(best_model["epoch"], save_best=True)
                continue

            # If modeling is done intermittently, still save models that outperform
            # the best metric
            # save_best = best and not self.mnt_metric == "epoch"
            save_best = True

            # Due to the fast runtime/slow HDD combination, modeling can dominate
            # the total training time, so we optionally skip models for some of
            # the first epochs
            if epoch < self.skip_first_n_saves and not self.save_only_best:
                msg = f"Skipping model save at epoch {epoch} <= {self.skip_first_n_saves}"
                self.logger.info(msg)
                continue

            if epoch % self.save_period == 0 and save_best:
                self._save_model(epoch, save_best=best)
                print("This epoch, the save best :{}".format(best))
                if best:
                    for key, cached in cached_preds.items():
                        log_dir = Path(self.config.log_dir)
                        prediction_path = log_dir / f"{key}_preds.txt"
                        prediction_logits_path = log_dir / f"{key}_preds_logits.npy"
                        np.save(prediction_logits_path, cached["preds"])
                        gt_logits_path = log_dir / f"{key}_gt_logits.npy"
                        np.save(gt_logits_path, cached["labels"].cpu().numpy())
                        vid_names = []
                        sort_predict = np.argsort(cached["preds"])[:, ::-1]
                        with open(str(prediction_path), 'w') as f:
                            for kk in range(cached["preds"].shape[0]):
                                pred_classes = [str(v) for v in sort_predict[kk, :]]
                                vid_name = cached["vid_name"][kk]
                                if key == "test":
                                    vid_name = vid_name[kk].split('/')[-1] + '.mp4'
                                row = f"{vid_name} {' '.join(pred_classes)}"
                                print(row, file=f)
                                vid_names.append(vid_name)
                        save_name_path = log_dir / f"{key}_vid_name.pkl"
                        with open(save_name_path, 'wb') as f:
                            pickle.dump(vid_names, f)
                        self.logger.info(f"All {key} preds saved")
                        self.logger.info(f"Wrote result to: {str(prediction_path)}")

            if epoch > self.num_keep_ckpts:
                self.purge_stale_models()

    def purge_stale_models(self):
        """Remove models that are no longer neededself.

        NOTE: This function assumes that the `best` model has already been renamed
        to have a format that differs from `model-epoch<num>.pth`
        """
        all_ckpts = list(self.model_dir.glob("*.pdparams"))
        found_epoch_ckpts = list(self.model_dir.glob("model-epoch*.pdparams"))
        if len(all_ckpts) <= self.num_keep_ckpts:
            return

        msg = "Expected at the best model to have been renamed to a different format"
        if not len(all_ckpts) > len(found_epoch_ckpts):
            print("Warning, purging model, but the best epoch was not saved!")
        # assert len(all_ckpts) > len(found_epoch_ckpts), msg

        # purge the oldest models
        regex = r".*model-epoch(\d+)[.pdparams$"
        epochs = [int(re.search(regex, str(x)).groups()[0]) for x in found_epoch_ckpts]
        sorted_ckpts = sorted(list(zip(epochs, found_epoch_ckpts)), key=lambda x: -x[0])

        for epoch, stale_ckpt in sorted_ckpts[self.num_keep_ckpts:]:
            tic = time.time()
            stale_ckpt.unlink()
            msg = f"removing stale model [epoch {epoch}] [took {time.time() - tic:.2f}s]"
            self.logger.info(msg)

    def _save_model(self, epoch, save_best=False):
        """Saving models

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved model to 'trained_model.pdparams'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if self.include_optim_in_save_model:
            state["optimizer"] = self.optimizer.state_dict()

        filename = str(self.model_dir /
                       'model-epoch{}.pdparams'.format(epoch))
        tic = time.time()
        self.logger.info("Saving model: {} ...".format(filename))
        paddle.save(state, filename)
        self.logger.info(f"Done in {time.time() - tic:.3f}s")
        if save_best:
            self.logger.info("Updating 'best' model: {} ...".format(filename))
            best_path = str(self.model_dir / 'trained_model.pdparams')
            paddle.save(state, best_path)
            self.logger.info(f"Done in {time.time() - tic:.3f}s")

    def _resume_model(self, resume_path):
        """ Resume from saved models

        :param resume_path: model path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading model: {} ...".format(resume_path))
        model = paddle.load(resume_path)
        self.model.load_dict(model)
        self.logger.info(f"model loaded. Resume training from epoch {self.start_epoch}")

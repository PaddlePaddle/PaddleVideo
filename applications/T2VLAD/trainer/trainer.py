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

import paddle
import numpy as np

from base import BaseTrainer
from utils import memory_summary
from contextlib import contextmanager


def verbose(epoch, metrics, mode, name="TEST"):
    r1, r5, r10, r50 = metrics["R1"], metrics["R5"], metrics["R10"], metrics["R50"]
    msg = f"[{mode}]{name:s} epoch {epoch}, R@1: {r1:.1f}"
    msg += f", R@5: {r5:.1f}, R@10 {r10:.1f}, R@50 {r50:.1f}"
    msg += f"MedR: {metrics['MedR']:g}, MeanR: {metrics['MeanR']:.1f}"
    print(msg)


@contextmanager
def ctxt_mgr(samples):
    """Provide a context for managing temporary, cloned copies of retrieval
    sample tensors.

    The rationale here is that to use nan-checking in the model (to validate the
    positions of missing experts), we need to modify the underlying tensors. This
    function lets the evaluation code run (and modify) temporary copies, without
    modifying the originals.
    """

    exp_dict = samples["experts"].items()
    experts = {key: val.clone() for key, val in exp_dict}
    samples_ = {
        "experts": experts,
        "ind": samples["ind"],
        "text": samples["text"],
        "cap_id": samples["cap_id"],
        "att_mask": samples["att_mask"],
    }
    if "text_token_mask" in samples:
        samples_["text_token_mask"] = samples["text_token_mask"]
    try:
        yield samples_
    finally:
        del samples_


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config, data_loaders,
                 lr_scheduler, visualizer, skip_first_n_saves,
                 include_optim_in_save_model, force_cpu_val, cache_targets=set(),
                 num_keep_ckpts=3, mini_train=False, val_freq=1, skip_tboard=False):
        super().__init__(model, loss, metrics, optimizer, config, mini_train=mini_train,
                         skip_tboard=skip_tboard, num_keep_ckpts=num_keep_ckpts)
        self.config = config
        self.cache_targets = cache_targets
        self.data_loaders = data_loaders
        self.lr_scheduler = lr_scheduler
        self.mini_train = mini_train
        self.len_epoch = len(self.data_loaders["train"])
        self.log_step = int(np.sqrt(data_loaders["train"].batch_size))
        self.visualizer = visualizer
        self.force_cpu_val = force_cpu_val
        self.val_freq = val_freq
        self.skip_first_n_saves = skip_first_n_saves
        self.include_optim_in_save_model = include_optim_in_save_model
        self.seen = {"train": 0, "val": 0}

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        total_loss = 0
        self.model.train()
        memory_summary()

        for batch_idx, minibatch in enumerate(self.data_loaders["train"]):
            output = self.model(**minibatch)
            if "retrieval" in self.data_loaders.dataloaders:
                loss = self.loss(output["cross_view_conf_matrix"])
            else:
                loss = self.loss(x=output["class_preds"], target=labels)
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.clear_grad()

            sample_key = list(minibatch["experts"].keys())[0]
            batch_size = minibatch["experts"][sample_key].shape[0]
            self.seen["train"] += batch_size

            total_loss += loss.item()

            if batch_idx % self.log_step == 0:
                prog = self._progress(batch_idx)
                self.logger.info(f"Train Epoch: {epoch} {prog} Loss: {loss.item():.6f}")

            if batch_idx == self.len_epoch or (self.mini_train and batch_idx > 3):
                break

        log = {'loss': total_loss / self.len_epoch}
        if epoch % self.val_freq == 0:
            nested_log, cached_preds = self._valid_epoch(epoch)
            log.update(nested_log)
        else:
            nested_log, cached_preds = {}, None
            self.logger.info(f"skipping val for epoch: {epoch}")

        self.lr_scheduler.step()

        self.logger.info(f"LR {self.lr_scheduler.get_lr()}")
        return log, cached_preds

    def _valid_epoch(self, epoch):
        """Validate model after an epoch of training and store results to disk.

        Args:
            epoch (int): the current epoch

        Returns:
            A log that contains information about validation

        NOTE: The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        cached_preds = {key: {"vid_name": [], "preds": [], "labels": []}
                        for key in self.cache_targets}

        with paddle.no_grad():
            if "retrieval" in self.data_loaders.dataloaders:
                samples, meta = self.data_loaders["retrieval"]
                sample_key = list(samples["experts"].keys())[0]
                batch_size = samples["experts"][sample_key].shape[0]
                self.seen["val"] += batch_size
                num_queries = samples["text"].shape[0] * samples["text"].shape[1]
                safe_queries = 1
                text_keys = ['text', 'cap_id', 'att_mask', 'text_token_mask']
                if num_queries > safe_queries:
                    chk = 50
                    tck = 50
                    if samples['text'].shape[0] % chk == 0:
                        vid_batch = samples['text'].shape[0] // chk
                    else:
                        vid_batch = samples['text'].shape[0] // chk + 1
                    if samples['text'].shape[0] % tck == 0:
                        text_batch  =  samples['text'].shape[0] // tck
                    else:
                        text_batch  =  samples['text'].shape[0] // tck + 1

                    sub_sims = []
                    for idx in range(text_batch):
                        if idx % 5 == 0:
                            print(idx,'/',text_batch)
                        sub_samples = {}
                        for key in text_keys:
                            sub_samples.update({key: samples[key][idx*tck:idx*tck+tck]})
                        subsub_sims = []
                        for vid in range(vid_batch):
                            sub_samples['experts'] = {}
                            sub_samples['ind'] = {} 
                            for expert in samples['experts'].keys():
                                sub_samples['experts'][expert] = samples['experts'][expert][vid*chk:vid*chk+chk]
                                sub_samples['ind'][expert] = samples['ind'][expert][vid*chk:vid*chk+chk]
                            with ctxt_mgr(sub_samples) as xx:
                                output = self.model(**xx)
                            subsub_sims.append(output["cross_view_conf_matrix"].cpu())
                    
                        subsub_sims = paddle.concat(subsub_sims, axis=1)
                        sub_sims.append(subsub_sims)

                    sims = paddle.concat(sub_sims, axis=0)
                    sims = paddle.to_tensor(sims, dtype='float32').cpu().numpy()
                else:
                    with ctxt_mgr(samples) as xx:
                        output = self.model(**xx)
                    sims = paddle.to_tensor(output["cross_view_conf_matrix"], dtype='float32').cpu().numpy()

                # sample the loss (using only the first query for each video)
                queries_per_vid = meta["query_masks"].shape[1]
                sims_ = paddle.to_tensor(sims).reshape([-1, queries_per_vid, sims.shape[-1]])
                loss = self.loss(sims_[:, 0, :])
                dataset = self.data_loaders.dataset_name
                nested_metrics = {}
                for metric in self.metrics:
                    metric_name = metric.__name__
                    res = metric(sims, query_masks=meta["query_masks"])
                    if metric_name == "mean_average_precision":
                        print(f"Epoch: {epoch}, mean AP: {res['mAP']}")
                    else:
                        verbose(epoch=epoch, metrics=res, name=dataset, mode=metric_name)
                    nested_metrics[metric_name] = res

                # TODO(Samuel) disabled visualisation for now, simple to add in later
                num_test_caps = self.data_loaders.num_test_captions
                if num_test_caps == 1 and meta["raw_captions"] is not None:
                    if self.visualizer is not None:
                        self.visualizer.visualize_ranking(
                            sims=sims,
                            meta=meta,
                            epoch=epoch,
                            nested_metrics=nested_metrics,
                        )
                return {"nested_val_metrics": nested_metrics}, cached_preds

            elif "val" in self.data_loaders.dataloaders:
                metrics = [x() for x in self.metrics]
                for batch_idx, minibatch in enumerate(self.data_loaders["val"]):
                    labels = minibatch.pop("labels")
                    vid_name = minibatch.pop("vid_name")
                    output = self.model(**minibatch)
                    if "val" in self.cache_targets:
                        cached_preds["val"]["vid_name"].append(vid_name)
                        cached_preds["val"]["preds"].append(output["class_preds"])

                    for metric in metrics:
                        metric.add(output=output["class_preds"], target=labels)
                    if batch_idx % self.log_step == 0:
                        prog = self._progress(batch_idx)
                        self.logger.info(f"Val Epoch: {epoch} {prog}")
                
                nested_metrics = {}
                for metric in metrics:
                    if hasattr(metric, "topk"):
                        res = {f"top{key}": val for key, val in
                               zip(metric.topk, metric.value())}
                        nested_metrics["accuracy"] = res
                    else:
                        raise ValueError(f"unsupported mettric: {type(metric)}")
                nested = {"nested_val_metrics": nested_metrics}

                for target in self.cache_targets - {"val"}:
                    for batch_idx, minibatch in enumerate(self.data_loaders["tiny"]):
                        if "labels" in minibatch:
                            cached_preds[target]["labels"].append(minibatch.pop("labels"))
                        cached_preds[target]["vid_name"].append(minibatch.pop("vid_name"))
                        output = self.model(**minibatch)
                        cached_preds[target]["preds"].append(output["class_preds"])

                # aggregate all cached predictions
                for target in self.cache_targets:
                    for key, val in cached_preds[target].items():
                        cached_preds[key] = paddle.concat(val).cpu().numpy()
                return nested, cached_preds

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loaders, 'n_samples'):
            current = batch_idx * self.data_loaders.batch_size
            total = self.data_loaders.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

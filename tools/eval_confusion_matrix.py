# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standalone script to evaluate a trained recognizer on a validation set and
draw confusion_matrix.png + confusion_matrix_normalized.png.

Example:
    python tools/eval_confusion_matrix.py \
        -c configs/recognition/pptsm/pptsm_k400_videos_uniform.yaml \
        -w /home/quoctuan/Downloads/ppTSM/ppTSM_best.pdparams \
        --file_path "/path/to/Valid/val.txt" \
        --data_prefix "/path/to/Valid" \
        --output_dir /home/quoctuan/Downloads/ppTSM
"""
import argparse
import os
import os.path as osp
import sys

# make 'paddlevideo' importable when running this file directly from anywhere
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))

import numpy as np
import paddle

from paddlevideo.loader.builder import build_dataloader, build_dataset
from paddlevideo.modeling.builder import build_model
from paddlevideo.utils import (
    compute_confusion_matrix,
    get_config,
    load,
    plot_confusion_matrix,
)


def parse_args():
    parser = argparse.ArgumentParser("Draw confusion matrix on a valid set")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/recognition/pptsm/pptsm_k400_videos_uniform.yaml",
        help="config file path")
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        required=True,
        help="trained .pdparams weights path")
    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="validation list file (each line: 'relative/path.mp4 label')")
    parser.add_argument(
        "--data_prefix",
        type=str,
        default=None,
        help="root dir prepended to each path in file_path")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="where to save the confusion matrix figures")
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="number of classes; auto-detected from the checkpoint if omitted")
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument(
        "--class_names",
        type=str,
        default=None,
        help="comma-separated class names; if omitted, inferred from the "
        "sub-directories of data_prefix (sorted alphabetically)")
    return parser.parse_args()


def detect_num_classes(state_dict):
    """Infer the number of classes from the classification head weight."""
    for k, v in state_dict.items():
        if k.endswith("fc.weight") and len(v.shape) == 2:
            return int(v.shape[1])
    return None


def infer_class_names(data_prefix, num_classes):
    """Use the sorted sub-directory names of data_prefix as class names,
    which matches the label order produced by alphabetical folder indexing."""
    if data_prefix and osp.isdir(data_prefix):
        subdirs = sorted(
            d for d in os.listdir(data_prefix)
            if osp.isdir(osp.join(data_prefix, d)))
        if len(subdirs) == num_classes:
            return subdirs
    return [str(i) for i in range(num_classes)]


def main():
    args = parse_args()
    cfg = get_config(args.config, show=False)

    paddle.set_device("gpu" if args.use_gpu else "cpu")

    # 1. Load checkpoint and figure out the real number of classes.
    state_dict = load(args.weights)
    num_classes = args.num_classes or detect_num_classes(state_dict)
    assert num_classes is not None, \
        "Could not detect num_classes, please pass --num_classes explicitly."
    print(f"num_classes = {num_classes}")

    # 2. Build the model with the matching head and load weights.
    #    Skip the ImageNet pretrain since we load the full checkpoint anyway.
    if cfg.MODEL.get("backbone") is not None:
        cfg.MODEL.backbone.pretrained = None
    if cfg.MODEL.get("head") is not None:
        cfg.MODEL.head.num_classes = num_classes
    model = build_model(cfg.MODEL)
    model.set_state_dict(state_dict)
    model.eval()

    # 3. Build the validation dataloader from the given list/prefix.
    valid_cfg = cfg.DATASET.valid
    valid_cfg.file_path = args.file_path
    if args.data_prefix is not None:
        valid_cfg.data_prefix = args.data_prefix
    dataset = build_dataset((valid_cfg, cfg.PIPELINE.valid))
    dataloader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        places=paddle.get_device(),
        drop_last=False,
        shuffle=False)

    # 4. Run inference and collect predictions / labels.
    all_preds, all_labels = [], []
    total = len(dataloader)
    with paddle.no_grad():
        for i, data in enumerate(dataloader):
            cls_score = model(data, mode="test")
            preds = paddle.argmax(cls_score, axis=-1)
            all_preds.append(preds.numpy().reshape(-1))
            all_labels.append(np.asarray(data[1]).reshape(-1))
            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"  [{i + 1}/{total}] batches done")

    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)
    print(f"Evaluated {len(labels)} samples.")

    # 5. Compute and save the confusion matrices.
    class_names = (args.class_names.split(",")
                   if args.class_names else infer_class_names(
                       args.data_prefix or valid_cfg.get("data_prefix"),
                       num_classes))
    cm = compute_confusion_matrix(labels, preds, num_classes)

    os.makedirs(args.output_dir, exist_ok=True)
    plot_confusion_matrix(
        cm,
        osp.join(args.output_dir, "confusion_matrix.png"),
        class_names=class_names,
        normalize=False)
    plot_confusion_matrix(
        cm,
        osp.join(args.output_dir, "confusion_matrix_normalized.png"),
        class_names=class_names,
        normalize=True)

    # 6. Print a quick per-class summary (support / recall / precision).
    print("\nPer-class summary:")
    print(f"{'class':>16} {'support':>8} {'recall':>8} {'precision':>10}")
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    diag = np.diag(cm)
    for i in range(num_classes):
        recall = diag[i] / row_sum[i] if row_sum[i] else 0.0
        precision = diag[i] / col_sum[i] if col_sum[i] else 0.0
        print(f"{class_names[i]:>16} {int(row_sum[i]):>8} "
              f"{recall:>8.3f} {precision:>10.3f}")
    acc = diag.sum() / cm.sum() if cm.sum() else 0.0
    print(f"\nOverall top-1 accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()

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

import os.path as osp

import numpy as np

from .logger import get_logger

logger = get_logger("paddlevideo")

__all__ = [
    "plot_curve",
    "compute_confusion_matrix",
    "plot_confusion_matrix",
]


def _get_pyplot():
    """Lazily import matplotlib with a non-interactive backend so that
    plotting works on headless training machines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_curve(history, save_path):
    """Plot the training/validation loss and accuracy curves.

    Args:
        history (dict): collected metrics. Expected keys (any subset):
            'epoch', 'train_loss', 'train_top1',
            'val_epoch', 'val_loss', 'val_top1'.
        save_path (str): file path of the output figure (e.g. ``curve.png``).
    """
    try:
        plt = _get_pyplot()
    except Exception as e:
        logger.warning(f"Skip plotting curve, matplotlib is unavailable: {e}")
        return

    epoch = history.get("epoch", [])
    if not epoch:
        logger.warning("Skip plotting curve, no epoch metrics were recorded.")
        return

    fig, ax_loss = plt.subplots(figsize=(10, 6))

    # Left axis: loss
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    if history.get("train_loss"):
        ax_loss.plot(epoch, history["train_loss"], "-o", color="tab:blue",
                     markersize=3, label="train loss")
    if history.get("val_loss"):
        ax_loss.plot(history["val_epoch"], history["val_loss"], "-s",
                     color="tab:cyan", markersize=3, label="val loss")
    ax_loss.tick_params(axis="y")

    # Right axis: accuracy (top1)
    has_acc = history.get("train_top1") or history.get("val_top1")
    if has_acc:
        ax_acc = ax_loss.twinx()
        ax_acc.set_ylabel("top1 acc")
        if history.get("train_top1"):
            ax_acc.plot(epoch, history["train_top1"], "-o", color="tab:red",
                        markersize=3, label="train top1")
        if history.get("val_top1"):
            ax_acc.plot(history["val_epoch"], history["val_top1"], "-s",
                        color="tab:orange", markersize=3, label="val top1")
        ax_acc.set_ylim(0, 1)

    # Merge legends from both axes.
    lines, labels = ax_loss.get_legend_handles_labels()
    if has_acc:
        lines2, labels2 = ax_acc.get_legend_handles_labels()
        lines += lines2
        labels += labels2
    ax_loss.legend(lines, labels, loc="best")

    plt.title("Training Curve")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Training curve saved to {save_path}")


def compute_confusion_matrix(labels, preds, num_classes):
    """Compute a confusion matrix with numpy (no sklearn dependency).

    Args:
        labels (array-like): ground-truth class indices.
        preds (array-like): predicted class indices.
        num_classes (int): number of classes.

    Returns:
        np.ndarray: confusion matrix of shape ``[num_classes, num_classes]``
            where rows are the true label and columns the prediction.
    """
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    preds = np.asarray(preds).astype(np.int64).reshape(-1)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm, save_path, class_names=None, normalize=False):
    """Plot a confusion matrix as a heatmap.

    Args:
        cm (np.ndarray): confusion matrix, shape ``[C, C]``.
        save_path (str): output figure path.
        class_names (list[str], optional): tick labels. Defaults to indices.
        normalize (bool): if True, normalize each row to sum to 1.
    """
    try:
        plt = _get_pyplot()
    except Exception as e:
        logger.warning(
            f"Skip plotting confusion matrix, matplotlib is unavailable: {e}")
        return

    cm = np.asarray(cm)
    num_classes = cm.shape[0]
    if class_names is None or len(class_names) != num_classes:
        class_names = [str(i) for i in range(num_classes)]

    if normalize:
        # normalize per column (by predicted label) so each column sums to 1;
        # the diagonal then reads as the per-class precision.
        col_sum = cm.sum(axis=0, keepdims=True)
        # avoid division by zero for classes the model never predicts
        cm_display = np.divide(cm, col_sum,
                               out=np.zeros(cm.shape, dtype=np.float64),
                               where=col_sum != 0)
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        cm_display = cm
        fmt = "d"
        title = "Confusion Matrix"

    # Scale figure size with the number of classes for readability.
    side = max(6, num_classes * 0.6)
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel="True label",
           xlabel="Predicted label",
           title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Annotate cells when the matrix is small enough to stay legible.
    if num_classes <= 30:
        thresh = cm_display.max() / 2.0 if cm_display.size else 0.0
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, format(cm_display[i, j], fmt),
                        ha="center", va="center", fontsize=8,
                        color="white" if cm_display[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {save_path}")

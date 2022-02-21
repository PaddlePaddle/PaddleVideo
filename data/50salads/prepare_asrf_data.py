import argparse
import glob
import os
import sys
from typing import Dict
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

dataset_names = ["50salads", "breakfast", "gtea"]


def get_class2id_map(dataset: str,
                     dataset_dir: str = "./dataset") -> Dict[str, int]:
    """
    Args:
        dataset: 50salads, gtea, breakfast
        dataset_dir: the path to the datset directory
    """

    assert (dataset in dataset_names
            ), "You have to choose 50salads, gtea or breakfast as dataset."

    with open(os.path.join(dataset_dir, "{}/mapping.txt".format(dataset)),
              "r") as f:
        actions = f.read().split("\n")[:-1]

    class2id_map = dict()
    for a in actions:
        class2id_map[a.split()[1]] = int(a.split()[0])

    return class2id_map


def get_arguments() -> argparse.Namespace:
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="convert ground truth txt files to numpy array")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./dataset",
        help="path to a dataset directory (default: ./dataset)",
    )

    return parser.parse_args()


def main() -> None:
    args = get_arguments()

    datasets = ["50salads", "gtea", "breakfast", "baseball"]

    for dataset in datasets:
        # make directory for saving ground truth numpy arrays
        cls_save_dir = os.path.join(args.dataset_dir, dataset, "gt_arr")
        if not os.path.exists(cls_save_dir):
            os.mkdir(cls_save_dir)

        # make directory for saving ground truth numpy arrays
        boundary_save_dir = os.path.join(args.dataset_dir, dataset,
                                         "gt_boundary_arr")
        if not os.path.exists(boundary_save_dir):
            os.mkdir(boundary_save_dir)

        # class to index mapping
        class2id_map = get_class2id_map(dataset, dataset_dir=args.dataset_dir)

        gt_dir = os.path.join(args.dataset_dir, dataset, "groundTruth")
        gt_paths = glob.glob(os.path.join(gt_dir, "*.txt"))

        for gt_path in gt_paths:
            # the name of ground truth text file
            gt_name = os.path.relpath(gt_path, gt_dir)

            with open(gt_path, "r") as f:
                gt = f.read().split("\n")[:-1]

            gt_array = np.zeros(len(gt))
            for i in range(len(gt)):
                gt_array[i] = class2id_map[gt[i]]

            # save array
            np.save(os.path.join(cls_save_dir, gt_name[:-4] + ".npy"), gt_array)

            # the name of ground truth text file
            gt_name = os.path.relpath(gt_path, gt_dir)

            with open(gt_path, "r") as f:
                gt = f.read().split("\n")[:-1]

            # define the frame where new action starts as boundary frame
            boundary = np.zeros(len(gt))
            last = gt[0]
            boundary[0] = 1
            for i in range(1, len(gt)):
                if last != gt[i]:
                    boundary[i] = 1
                    last = gt[i]

            # save array
            np.save(os.path.join(boundary_save_dir, gt_name[:-4] + ".npy"),
                    boundary)

    print("Done")


if __name__ == "__main__":
    main()

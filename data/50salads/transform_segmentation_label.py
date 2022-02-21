import json
import numpy as np
import argparse
import os

from tqdm import tqdm


def generate_mapping_list_txt(action_dict, out_path):
    out_txt_file_path = os.path.join(out_path, "mapping.txt")
    f = open(out_txt_file_path, "w", encoding='utf-8')
    for key, action_name in action_dict.items():
        str_str = str(key) + " " + action_name + "\n"
        f.write(str_str)
    # add None
    str_str = str(len(action_dict)) + " None" + "\n"
    f.write(str_str)
    f.close()


def segmentation_convert_localization_label(prefix_data_path, out_path,
                                            action_dict, fps):
    label_path = os.path.join(prefix_data_path)
    label_txt_name_list = os.listdir(label_path)

    labels_dict = {}
    labels_dict["fps"] = fps
    labels_list = []
    for label_name in tqdm(label_txt_name_list, desc='label convert:'):
        label_dict = {}
        label_dict["url"] = label_name.split(".")[0] + ".mp4"
        label_txt_path = os.path.join(prefix_data_path, label_name)

        with open(label_txt_path, "r", encoding='utf-8') as f:
            gt = f.read().split("\n")[:-1]
        label_dict["total_frames"] = len(gt)

        boundary_index_list = [0]
        before_action_name = gt[0]
        for index in range(1, len(gt)):
            if before_action_name != gt[index]:
                boundary_index_list.append(index)
                before_action_name = gt[index]
        actions_list = []
        for index in range(len(boundary_index_list) - 1):
            if gt[boundary_index_list[index]] != "None":
                action_name = gt[boundary_index_list[index]]
                start_sec = float(boundary_index_list[index]) / float(fps)
                end_sec = float(boundary_index_list[index + 1] - 1) / float(fps)
                action_id = action_dict[action_name]
                label_action_dict = {}
                label_action_dict["label_names"] = action_name
                label_action_dict["start_id"] = start_sec
                label_action_dict["end_id"] = end_sec
                label_action_dict["label_ids"] = [action_id]
                actions_list.append(label_action_dict)

        label_dict["actions"] = actions_list
        labels_list.append(label_dict)
    labels_dict["gts"] = labels_list
    output_path = os.path.join(out_path, "output.json")
    f = open(output_path, "w", encoding='utf-8')
    f.write(json.dumps(labels_dict, indent=4))
    f.close()


def generate_action_dict(label):
    action_dict = {}
    for gt in label["gts"]:
        for action in gt["actions"]:
            label_id = action["label_ids"][0]
            label_name = action["label_names"][0]
            action_dict[label_id] = label_name

    return action_dict


def load_action_dict(data_path):
    mapping_txt_path = os.path.join(data_path, "mapping.txt")
    with open(mapping_txt_path, "r", encoding='utf-8') as f:
        actions = f.read().split("\n")[:-1]

    class2id_map = dict()
    for a in actions:
        class2id_map[a.split()[1]] = int(a.split()[0])

    return class2id_map


def localization_convert_segmentation_label(label, prefix_data_path, out_path):
    path = os.path.join(out_path, "groundTruth")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
    else:
        print(path + ' 目录已存在')

    fps = float(label["fps"])
    video_list = []
    for gt in tqdm(label["gts"], desc='label convert:'):
        video_name = gt["url"].split(".")[0]
        data_path = os.path.join(prefix_data_path, video_name + ".pkl")
        video_list.append(video_name + ".txt")
        feature = np.load(data_path, allow_pickle=True)["image_feature"]

        num_feture = feature.shape[0]
        seg_label = ["None"] * (num_feture)
        for action in gt["actions"]:
            start_id = action["start_id"]
            end_id = action["end_id"]

            label_name = action["label_names"]

            start_index = int(np.floor(start_id * fps))
            end_index = int(np.floor(end_id * fps)) + 1

            if end_index < num_feture - 1:
                seg_label[start_index:end_index] = label_name * (end_index -
                                                                 start_index)
            elif start_index < num_feture - 1:
                seg_label[start_index:] = label_name * (num_feture -
                                                        start_index)
            else:
                pass

        if len(seg_label) != num_feture:
            seg_label = seg_label[:num_feture]
        out_txt_file_path = os.path.join(out_path, "groundTruth",
                                         video_name + ".txt")
        str = '\n'
        f = open(out_txt_file_path, "w", encoding='utf-8')
        f.write(str.join(seg_label) + str)
        f.close()
    out_txt_file_path = os.path.join(out_path, "train_list.txt")
    str = '\n'
    f = open(out_txt_file_path, "w", encoding='utf-8')
    f.write(str.join(video_list) + str)
    f.close()


def main():
    args = get_arguments()

    if args.mode in ["segmentation", "localization"]:
        if args.mode == "segmentation":
            with open(args.label_path, 'r', encoding='utf-8') as json_file:
                label = json.load(json_file)
            action_dict = generate_action_dict(label)
            generate_mapping_list_txt(action_dict, args.out_path)
            localization_convert_segmentation_label(label, args.data_path,
                                                    args.out_path)

        elif args.mode == "localization":
            action_dict = load_action_dict(args.label_path)
            segmentation_convert_localization_label(args.data_path,
                                                    args.out_path,
                                                    action_dict,
                                                    fps=25.0)

    else:
        raise NotImplementedError


def get_arguments():
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="convert segmentation and localization label")
    parser.add_argument("label_path", type=str, help="path of a label file")
    parser.add_argument(
        "data_path",
        type=str,
        help="path of video feature or segmentation label txt.",
    )
    parser.add_argument(
        "out_path",
        type=str,
        help="path of output file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="segmentation",
        help="Convert segmentation label or localization label.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()

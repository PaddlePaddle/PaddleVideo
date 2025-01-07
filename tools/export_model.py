# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import os.path as osp
import sys
import yaml
import copy
import paddle
from paddle.jit import to_static
from paddle.static import InputSpec
from collections import OrderedDict

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../")))

from paddlevideo.modeling.builder import build_model
from paddlevideo.utils import get_config
from paddlevideo.utils import mkdir, get_logger
from paddlevideo.tasks.download import get_weights_path_from_url


def parse_args():
    parser = argparse.ArgumentParser("PaddleVideo export model script")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/example.yaml",
        help="config file path",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="config options to be overridden",
    )
    parser.add_argument(
        "-p",
        "--pretrained_params",
        default="./best.pdparams",
        type=str,
        help="params path",
    )
    parser.add_argument(
        "-o", "--output_path", type=str, default="./inference", help="output path"
    )

    parser.add_argument(
        "--save_name",
        type=str,
        default=None,
        help="specify the exported inference \
                             files(pdiparams and pdmodel) name,\
                             only used in TIPC",
    )

    return parser.parse_args()


def trim_config(cfg):
    """
    Reuse the trainging config will bring useless attributes, such as: backbone.pretrained model.
    and some build phase attributes should be overrided, such as: backbone.num_seg.
    Trim it here.
    """
    model_name = cfg.model_name
    if cfg.MODEL.get("backbone") and cfg.MODEL.backbone.get("pretrained"):
        cfg.MODEL.backbone.pretrained = ""  # not ued when inference

    # for distillation
    if cfg.MODEL.get("models"):
        if cfg.MODEL.models[0]["Teacher"]["backbone"].get("pretrained"):
            cfg.MODEL.models[0]["Teacher"]["backbone"]["pretrained"] = ""
        if cfg.MODEL.models[1]["Student"]["backbone"].get("pretrained"):
            cfg.MODEL.models[1]["Student"]["backbone"]["pretrained"] = ""

    return cfg, model_name


def get_input_spec(cfg, model_name):
    if model_name in ["ppTSM", "TSM", "MoViNet", "ppTSMv2"]:
        input_spec = [
            [
                InputSpec(
                    shape=[None, cfg.num_seg, 3, cfg.target_size, cfg.target_size],
                    dtype="float32",
                ),
            ]
        ]
    elif model_name in ["TokenShiftVisionTransformer"]:
        input_spec = [
            [
                InputSpec(
                    shape=[None, 3, cfg.num_seg * 3, cfg.target_size, cfg.target_size],
                    dtype="float32",
                ),
            ]
        ]
    elif model_name in ["TSN", "ppTSN"]:
        input_spec = [
            [
                InputSpec(
                    shape=[None, cfg.num_seg * 10, 3, cfg.target_size, cfg.target_size],
                    dtype="float32",
                ),
            ]
        ]
    elif model_name in ["BMN"]:
        input_spec = [
            [
                InputSpec(
                    shape=[None, cfg.feat_dim, cfg.tscale],
                    dtype="float32",
                    name="feat_input",
                ),
            ]
        ]
    elif model_name in ["TimeSformer", "ppTimeSformer"]:
        input_spec = [
            [
                InputSpec(
                    shape=[None, 3, cfg.num_seg * 3, cfg.target_size, cfg.target_size],
                    dtype="float32",
                ),
            ]
        ]
    elif model_name in ["VideoSwin"]:
        input_spec = [
            [
                InputSpec(
                    shape=[
                        None,
                        3,
                        cfg.num_seg * cfg.seg_len * 1,
                        cfg.target_size,
                        cfg.target_size,
                    ],
                    dtype="float32",
                ),
            ]
        ]
    elif model_name in ["VideoSwin_TableTennis"]:
        input_spec = [
            [
                InputSpec(
                    shape=[
                        None,
                        3,
                        cfg.num_seg * cfg.seg_len * 3,
                        cfg.target_size,
                        cfg.target_size,
                    ],
                    dtype="float32",
                ),
            ]
        ]
    elif model_name in ["AttentionLSTM"]:
        input_spec = [
            [
                InputSpec(
                    shape=[None, cfg.embedding_size, cfg.feature_dims[0]],
                    dtype="float32",
                ),  # for rgb_data
                InputSpec(
                    shape=[
                        None,
                    ],
                    dtype="int64",
                ),  # for rgb_len
                InputSpec(
                    shape=[None, cfg.embedding_size, cfg.feature_dims[0]],
                    dtype="float32",
                ),  # for rgb_mask
                InputSpec(
                    shape=[None, cfg.embedding_size, cfg.feature_dims[1]],
                    dtype="float32",
                ),  # for audio_data
                InputSpec(
                    shape=[
                        None,
                    ],
                    dtype="int64",
                ),  # for audio_len
                InputSpec(
                    shape=[None, cfg.embedding_size, cfg.feature_dims[1]],
                    dtype="float32",
                ),  # for audio_mask
            ]
        ]
    elif model_name in ["SlowFast"]:
        input_spec = [
            [
                InputSpec(
                    shape=[
                        None,
                        3,
                        cfg.num_frames // cfg.alpha,
                        cfg.target_size,
                        cfg.target_size,
                    ],
                    dtype="float32",
                    name="slow_input",
                ),
                InputSpec(
                    shape=[None, 3, cfg.num_frames, cfg.target_size, cfg.target_size],
                    dtype="float32",
                    name="fast_input",
                ),
            ]
        ]
    elif model_name in ["STGCN", "AGCN", "CTRGCN"]:
        input_spec = [
            [
                InputSpec(
                    shape=[
                        None,
                        cfg.num_channels,
                        cfg.window_size,
                        cfg.vertex_nums,
                        cfg.person_nums,
                    ],
                    dtype="float32",
                ),
            ]
        ]
    # 由于在模型运行过程中涉及到第一维乘human个数(N*M), 所以这里用1作为shape
    elif model_name in ["AGCN2s"]:
        input_spec = [
            [
                InputSpec(
                    shape=[
                        1,
                        cfg.num_channels,
                        cfg.window_size,
                        cfg.vertex_nums,
                        cfg.person_nums,
                    ],
                    dtype="float32",
                ),
            ]
        ]
    elif model_name in ["TransNetV2"]:
        input_spec = [
            [
                InputSpec(
                    shape=[
                        None,
                        cfg.num_frames,
                        cfg.height,
                        cfg.width,
                        cfg.num_channels,
                    ],
                    dtype="float32",
                ),
            ]
        ]
    elif model_name in ["MSTCN", "ASRF"]:
        input_spec = [
            [
                InputSpec(shape=[None, cfg.num_channels, None], dtype="float32"),
            ]
        ]
    elif model_name in ["ADDS"]:
        input_spec = [
            [
                InputSpec(
                    shape=[None, cfg.num_channels, cfg.height, cfg.width],
                    dtype="float32",
                ),
            ]
        ]
    elif model_name in ["AVA_SlowFast_FastRcnn"]:
        input_spec = [
            [
                InputSpec(
                    shape=[
                        None,
                        3,
                        cfg.num_frames // cfg.alpha,
                        cfg.target_size,
                        cfg.target_size,
                    ],
                    dtype="float32",
                    name="slow_input",
                ),
                InputSpec(
                    shape=[None, 3, cfg.num_frames, cfg.target_size, cfg.target_size],
                    dtype="float32",
                    name="fast_input",
                ),
                InputSpec(shape=[None, None, 4], dtype="float32", name="proposals"),
                InputSpec(shape=[None, 2], dtype="float32", name="img_shape"),
            ]
        ]
    elif model_name in ["PoseC3D"]:
        input_spec = [
            [
                InputSpec(shape=[None, 1, 17, 48, 56, 56], dtype="float32"),
            ]
        ]
    elif model_name in ["YOWO"]:
        input_spec = [
            [
                InputSpec(
                    shape=[1, 3, cfg.num_seg, cfg.target_size, cfg.target_size],
                    dtype="float32",
                ),
            ]
        ]
    return input_spec


def main():
    args = parse_args()
    cfg, model_name = trim_config(
        get_config(args.config, overrides=args.override, show=False)
    )
    logger = get_logger("paddlevideo")
    if cfg.get("Global") is not None:
        print(f"Building model({model_name})...")
    model = build_model(cfg.MODEL)
    if cfg.get("Global") is not None:
        weight = cfg.Global.pretrained_model
        if weight is not None:
            if weight.startswith(("http://", "https://")):
                weight = get_weights_path_from_url(weight)
            logger.info(f"Load pretrained model from {weight}")
            args.pretrained_params = weight
    assert osp.isfile(
        args.pretrained_params
    ), f"pretrained params ({args.pretrained_params} is not a file path.)"

    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    print(f"Loading params from ({args.pretrained_params})...")
    params = paddle.load(args.pretrained_params)
    model.set_dict(params)

    model.eval()

    if cfg.get("Global") is not None:

        export(cfg, model, args.output_path, uniform_output_enabled=True, logger=logger)
    else:
        # for rep nets
        for layer in model.sublayers():
            if hasattr(layer, "rep") and not getattr(layer, "is_repped"):
                layer.rep()

        input_spec = get_input_spec(cfg.INFERENCE, model_name)
        model = to_static(model, input_spec=input_spec)
        paddle.jit.save(
            model,
            osp.join(
                args.output_path,
                model_name if args.save_name is None else args.save_name,
            ),
        )
        print(f"model ({model_name}) has been already saved in ({args.output_path}).")


def export(
    cfg,
    model,
    save_path=None,
    uniform_output_enabled=False,
    ema_module=None,
    logger=None,
):
    output_dir = cfg.get("output_dir", f"./output")
    mkdir(output_dir)

    for layer in model.sublayers():
        if hasattr(layer, "rep") and not getattr(layer, "is_repped"):
            layer.rep()

    if not save_path:
        save_path = os.path.join(cfg.Global.save_inference_dir, "inference")
    else:
        save_path = os.path.join(save_path, "inference")
    model_name = cfg.model_name
    input_spec = get_input_spec(cfg.INFERENCE, model_name)
    model = to_static(model, input_spec=input_spec)

    paddle.jit.save(model, save_path)
    if cfg["Global"].get("export_for_fd", False) or uniform_output_enabled:
        dst_path = os.path.join(os.path.dirname(save_path), "inference.yml")
        target_size = cfg["INFERENCE"]["target_size"]
        infer_shape = [3, target_size, target_size]
        dump_infer_config(cfg, dst_path, infer_shape, logger)
    print(
        f'Export succeeded! The inference model exported has been saved in "{save_path}".'
    )


def represent_dictionary_order(self, dict_data):
    return self.represent_mapping("tag:yaml.org,2002:map", dict_data.items())


def setup_orderdict():
    yaml.add_representer(OrderedDict, represent_dictionary_order)


def dump_infer_config(inference_config, path, infer_shape, logger):
    setup_orderdict()
    infer_cfg = OrderedDict()
    config = inference_config

    if config["Global"].get("algorithm", None):
        infer_cfg["Global"] = {"model_name": config["Global"]["algorithm"]}

    if config["Infer"].get("transforms"):
        transforms = config["Infer"]["transforms"]
    else:
        logger.error("This config does not support dump transform config!")

    # Configuration required config for high-performance inference.
    if config["Global"].get("uniform_output_enabled"):
        infer_shape_with_batch = [
            [1] + infer_shape,
            [1] + infer_shape,
            [8] + infer_shape,
        ]

        dynamic_shapes = {"x": infer_shape_with_batch}

        backend_keys = ["paddle_infer", "tensorrt"]
        hpi_config = {
            "backend_configs": {
                key: {
                    (
                        "dynamic_shapes" if key == "tensorrt" else "trt_dynamic_shapes"
                    ): dynamic_shapes
                }
                for key in backend_keys
            }
        }

        infer_cfg["Hpi"] = hpi_config
    for transform in transforms:
        if "NormalizeImage" in transform:
            transform["NormalizeImage"]["channel_num"] = 3
            scale_str = transform["NormalizeImage"]["scale"]
            numerator, denominator = scale_str.split("/")
            numerator, denominator = float(numerator), float(denominator)
            transform["NormalizeImage"]["scale"] = float(numerator / denominator)
    infer_cfg["PreProcess"] = {
        "transform_ops": [
            infer_preprocess
            for infer_preprocess in transforms
            if "DecodeImage" not in infer_preprocess
        ]
    }
    if config.get("Infer"):
        if config["Infer"].get("PostProcess"):
            if config["Global"].get("algorithm") == "YOWO":
                infer_cfg["PostProcess"] = {
                     "transform_ops": [
                        post_op for post_op in config["Infer"].get("PostProcess")
                     ]
                }
                infer_cfg["label_list"] = config.get("label_list")

            else:
                postprocess_dict = copy.deepcopy(dict(config["Infer"]["PostProcess"]))
                with open(postprocess_dict["class_id_map_file"], "r", encoding="utf-8") as f:
                    label_id_maps = f.readlines()
                label_names = []
                for line in label_id_maps:
                    line = line.strip().split(" ", 1)
                    label_names.append(line[1:][0])

                postprocess_name = postprocess_dict.get("name", None)
                postprocess_dict.pop("class_id_map_file")
                postprocess_dict.pop("name")
                dic = OrderedDict()
                for item in postprocess_dict.items():
                    dic[item[0]] = item[1]
                dic["label_list"] = label_names

                if postprocess_name:
                    infer_cfg["PostProcess"] = {postprocess_name: dic}
                else:
                    raise ValueError("PostProcess name is not specified")
        else:
            infer_cfg["PostProcess"] = {"NormalizeFeatures": None}
    with open(path, "w") as f:
        yaml.dump(infer_cfg, f)
    logger.info("Export inference config file to {}".format(os.path.join(path)))


if __name__ == "__main__":
    main()

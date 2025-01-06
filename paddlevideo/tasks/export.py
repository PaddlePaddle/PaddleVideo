from paddle.static import InputSpec
from collections import OrderedDict

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
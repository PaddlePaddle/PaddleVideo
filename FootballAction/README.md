# 简介
该代码库用于体育动作检测+识别, 基于paddle1.8版本开发，结合PaddleVideo中的TSN,BMN, attentionLSTM的多个视频模型进行视频时空二阶段检测算法。
主要分为如下几步
 - 特征抽取
    - 图像特性，TSN
    - 音频特征，Vggsound
 - proposal提取，BMN
 - LSTM，动作分类 + 回归

# 基础镜像
```
docker pull tmtalgo/paddleaction:action-detection-v1
```

# 数据集
```
数据集来自欧洲杯2016，共49个足球视频，其中训练集44个，验证集5个
数据集地址: datasets/EuroCup2016/url.list
数据集label格式
{
    "0": "背景",
    "1": "进球",
    "2": "角球",
    "3": "任意球",
    "4": "黄牌",
    "5": "红牌",
    "6": "换人",
    "7": "界外球",
}
数据集标注文件:
datasets/EuroCup2016/label_cls8_train.json
datasets/EuroCup2016/label_cls8_val.json
```

# 代码结构
```
|-- root_dir
   |--  checkpoints                # 保存训练后的模型和log
   |--  datasets                   # 训练数据集和处理脚本
        |--  EuroCup2016           # xx数据集
            |--  feature_bmn       # bmn提取到的proposal
            |--  features          # tsn和audio特征, image fps=5, audio 每秒(1024)
            |--  input_for_bmn     # bmn训练的输入数据，widows=40
            |--  input_for_lstm    # lstm训练的输入数据
            |--  input_for_tsn     # tsn训练的数据数据
            |--  mp4               # 原始视频.mp4
            |--  frames            # 图像帧, fps=5, '.jpg'格式
            |--  pcm               # 音频pcm, 音频采样率16000，采用通道数1
            |--  url.list          # 视频列表
            |--  label_train.json  # 训练集原始gts
            |--  label_val.json    # 验证集原始gts
        |--  script                # 数据集处理脚本
    |--  predict                   # 模型预测代码
    |--  extractor                 # 特征提取脚本
    |--  train_lstm                # lstm训练代码
    |--  train_proposal            # tsn、bmn训练代码，基本保持paddle-release-v1.8版本，数据接口部分稍有改动，参考官网
        |--  configs               # tsn and bmn football config file
    |--  train_tsn.sh              # tsn训练启动脚本
    |--  train_bmn.sh              # bmn训练启动脚本
    |--  train_lstm.sh             # lstm训练启动脚本
```
# 训练与评估步骤
## step1, gts处理, 将原始标注数据处理成如下json格式
```
{
    'fps': 5,
    'gts': [
        {
            'url': 'xxx.mp4',
            'total_frames': 6341,
            'actions': [
                {
                    "label_ids": [7],
                    "label_names": ["界外球"],
                    "start_id": 395,
                    "end_id": 399
                },
                ...
            ]
        },
        ...
    ]
}
```

## step2 抽帧, 由mp4, 得到frames和pcm, 这里需要添加ffmpeg环境
```
cd datasets/script && python get_frames_pcm.py
```

## step3 TSN数据处理，由frames结合gts生成tsn训练所需要的正负样本
```
cd datasets/script && python get_instance_for_tsn.py
# 文件名按照如下格式
'{}_{}_{}_{}'.format(video_basename, start_id, end_id, label)
```

## step4 TSN训练
```
我们提供了足球数据训练的模型: checkpoints/models_tsn/TSN_epoch36.pdparams
如果需要在自己的数据上训练，可参考PaddleVideo https://github.com/PaddlePaddle/models/tree/release/2.0-beta/PaddleCV/video
config.yaml参考configs文件夹下tsn_football.yaml
```

## step 5 image and audio特征提取，保存到datasets features文件夹下
```
cd extractor && python extract_feat.py
# 特征维度, image(2048) + audio(1024)
# 特征保存格式如下，将如下dict保存在pkl格式，用于接下来的BMN训练
video_features = {'image_feature': np_image_features,
                  'audio_feature': np_audio_features}
```

## step 6 BMN特征处理，用于提取二分类的proposal，windows=40，根据gts和特征得到BMN训练所需要的数据集
```
cd datasets/script && python get_instance_for_bmn.py
# 数据格式
{
    "719b0a4bcb1f461eabb152298406b861_753_793": {
        "duration_second": 40.0,
        "duration_frame": 200,
        "feature_frame": 200,
        "subset": "train",
        "annotations": [
            {
                "segment": [
                    15.0,
                    22.0
                ],
                "label": "3.0",
                "label_name": "任意球"
            }
        ]
    },
    ...
}
```
## step 7 BMN训练
我们同样提供了足球数据训练的模型: checkpoints/models_bmn/BMN_epoch19.pdparams
如果要在自己的数据上训练，具体步骤参考step4 TSN 训练

## step 8 BMN预测，得到 start_id, end_id, score
```
cd extractor && python extract_bmn.py
# 数据格式
[
    {
        "video_name": "c9516c903de3416c97dae91a59e968d7",
        "num_proposal": 5534,
        "bmn_results": [
            {
                "start": 7850.0,
                "end": 7873.0,
                "score": 0.77194699622342
            },
            {
                "start": 4400.0,
                "end": 4443.0,
                "score": 0.7663803287641536
            },
            ...
        ]
    },
    ...
]
```

## step 9 LSTM数据处理，将BMN得到的proposal截断并处理成LSTM训练所需数据集
```
cd datasets/script && python get_instance_for_lstm.py
# 数据格式1，label_info
{
    "fps": 5,
    "results": [
        {
            "url": "https://xxx.mp4",
            "mode": "train",        # train or validation
            "total_frames": 6128,
            "num_gts": 93,
            "num_proposals": 5043,
            "proposal_actions": [
                {
                    "label": 6,
                    "norm_iou": 0.7575757575757576,
                    "norm_ioa": 0.7575757575757576,
                    "norm_start": -0.32,
                    "proposal": {
                        "start": 5011,
                        "end": 5036,
                        "score": 0.7723643666324231
                    },
                    "hit_gts": {
                        "label_ids": [
                            6
                        ],
                        "label_names": [
                            "换人"
                        ],
                        "start_id": 5003,
                        "end_id": 5036
                    }
                },
                ...
        },
        ...
}
# 数据格式2，LSTM训练所需要的feature
{
    'features': np.array(feature_hit, dtype=np.float32),    # TSN and audio 特征
    'feature_fps': 5,                                       # fps = 5
    'label_info': {'norm_iou': 0.5, 'label': 3, ...},       # 数据格式1中的'proposal_actions'
    'video_name': 'c9516c903de3416c97dae91a59e968d7'        # video_name
}
# 数据格式3，LSTM训练所需label.txt
'{} {}'.format(filename, label)
```

## step 10 LSTM训练
```
sh train_lstm.sh
```

## step 11 整个预测流程
```
cd predict && python predict.py
```

## step 12 结果评估
```
# 包括bmn proposal 评估和最终action评估
cd predict && python eval.py results.json
```

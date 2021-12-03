# 篮球动作检测模型

---
## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型推理](#模型推理)
- [模型评估](#模型评估)
- [代码结构](#代码结构)
- [参考论文](#参考论文)


## 模型简介
该代码库用于篮球动作检测+识别, 基于paddle2.0版本开发，结合PaddleVideo中的ppTSM, BMN, attentionLSTM的多个视频模型进行视频时空二阶段检测算法。
主要分为如下几步
 - 特征抽取
    - 图像特性，ppTSM
    - 音频特征，Vggsound
 - proposal提取，BMN
 - LSTM，动作分类 + 回归

<div align="center">
  <img src="BasketballAction.gif" width="480px"/><br>
</div>

### 基础镜像
```
docker pull tmtalgo/paddleaction:action-detection-v2
```


## 数据准备
```
数据集处理参考：https://github.com/PaddlePaddle/PaddleVideo/tree/application/FootballAction/datasets
数据集label格式
{
    "0": "背景",
    "1": "回放",
    "2": "进球-三分球",
    "3": "进球-两分球",
    "4": "进球-扣篮",
    "5": "罚球",
    "6": "跳球"
}
```


## 模型训练
代码参考足球动作检测：https://github.com/PaddlePaddle/PaddleVideo/tree/application/FootballAction
 - image 采样频率fps=5，如果有些动作时间较短，可以适当提高采样频率
 - BMN windows=200，即40s，所以测试自己的数据时，视频时长需大于40s

### step1, gts处理, 将原始标注数据处理成如下json格式
```
{
    'fps': 5,
    'gts': [
        {
            'url': 'xxx.mp4',
            'total_frames': 6341,
            'actions': [
                {
                    "label_ids": [6],
                    "label_names": ["跳球"],
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

### step2 抽帧, 由mp4, 得到frames和pcm, 这里需要添加ffmpeg环境
```
cd datasets/script && python get_frames_pcm.py
```

### step3 基础图像特征数据处理，由frames结合gts生成训练所需要的正负样本
```
cd datasets/script && python get_instance_for_tsn.py

# 文件名按照如下格式
'{}_{}_{}_{}'.format(video_basename, start_id, end_id, label)
```

### step4 ppTSM训练
```
我们提供了篮球数据训练的模型，参考checkpoints_basketball
如果需要在自己的数据上训练，可参考
https://github.com/PaddlePaddle/PaddleVideo/tree/release/2.0
config.yaml参考configs文件夹下pptsm_football_v2.0.yaml
```

### step 5 image and audio特征提取，保存到datasets features文件夹下
```
cd extractor && python extract_feat.py
# 特征维度, image(2048) + audio(1024) + pcm(640)
# 特征保存格式如下，将如下dict保存在pkl格式，用于接下来的BMN训练
video_features = {'image_feature': np_image_features,
                  'audio_feature': np_audio_features
                  'pcm_feature': np_pcm_features}
```

### step 6 BMN特征处理，用于提取二分类的proposal，windows=40，根据gts和特征得到BMN训练所需要的数据集
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
                "label": "6.0",
                "label_name": "跳球"
            }
        ]
    },
    ...
}
```

### step 7 BMN训练
我们同样提供了篮球数据训练的模型，参考checkpoints_basketball
如果要在自己的数据上训练，具体步骤参考step4 ppTSM 训练


### step 8 BMN预测，得到 start_id, end_id, score
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

### step 9 LSTM数据处理，将BMN得到的proposal截断并处理成LSTM训练所需数据集
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
                            "跳球"
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
    'features': np.array(feature_hit, dtype=np.float32),    # TSM audio and pcm 特征, 可根据需求选择组合
    'feature_fps': 5,                                       # fps = 5
    'label_info': {'norm_iou': 0.5, 'label': 3, ...},       # 数据格式1中的'proposal_actions'
    'video_name': 'c9516c903de3416c97dae91a59e968d7'        # video_name
}
# 数据格式3，LSTM训练所需label.txt
'{} {}'.format(filename, label)
```

### step 10 LSTM训练
```
sh run.sh       # LSTM 模块
```

## 模型推理
```
# 先将各模型export为inference model
# # export
# cd $home_dir/PaddleVideo-release-2.0
# python3 tools/export_model.py -c configs_basketball/configs_basketball.yaml \
#                               -p train_output/checkpoints/models_pptsm_pp/ppTSM_epoch26_00086.pdparams \
#                               -o inference/checkpoints_basketball/ppTSM

# 测试数据格式，可参考使用样例：
wget https://bj.bcebos.com/v1/acg-algo/PaddleVideo_application/basketball/datasets.tar.gz
# 测试模型，可使用我们提供的模型
wget https://bj.bcebos.com/v1/acg-algo/PaddleVideo_application/basketball/checkpoints_basketball.tar.gz

# 运行预测代码
cd predict
python predict.py
```


## 模型评估
```
# 代码: https://github.com/PaddlePaddle/PaddleVideo/blob/application/FootballAction/predict/eval.py
# 包括bmn proposal 评估和最终action评估
python eval.py results.json
```


## 代码结构
```
|-- root_dir
    |--  predict                    # 模型预测代码
        |--  datasets               # 数据
            |--  mp4                # 原始视频.mp4
            |--  frames             # 图像帧, fps=5, '.jpg'格式
            |--  pcm                # 音频pcm, 音频采样率16000，采用通道数1
            |--  mp4.list           # 视频列表
        |--  configs_basketball     # tsm, bmn and lstm basketball config file
        |--  checkpoints_basketball # tsm, bmn and lstm basketball model file
        |--  action_detect          # tsm, bmn and lstm inference

```

## 参考论文

- [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383.pdf), Ji Lin, Chuang Gan, Song Han
- [BMN: Boundary-Matching Network for Temporal Action Proposal Generation](https://arxiv.org/abs/1907.09702), Tianwei Lin, Xiao Liu, Xin Li, Errui Ding, Shilei Wen.
- [Attention Clusters: Purely Attention Based Local Feature Integration for Video Classification](https://arxiv.org/abs/1711.09550), Xiang Long, Chuang Gan, Gerard de Melo, Jiajun Wu, Xiao Liu, Shilei Wen
- [YouTube-8M: A Large-Scale Video Classification Benchmark](https://arxiv.org/abs/1609.08675), Sami Abu-El-Haija, Nisarg Kothari, Joonseok Lee, Paul Natsev, George Toderici, Balakrishnan Varadarajan, Sudheendra Vijayanarasimhan

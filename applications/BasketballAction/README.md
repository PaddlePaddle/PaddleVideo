# 篮球动作检测模型


## 内容
- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型评估](#模型评估)
- [模型推理](#模型推理)
- [模型优化](#模型优化)
- [模型部署](#模型部署)
- [参考论文](#参考论文)


## 模型简介
该代码库用于篮球动作检测+识别, 基于paddle2.0版本开发，结合PaddleVideo中的ppTSM, BMN, attentionLSTM的多个视频模型进行视频时空二阶段检测算法。
主要分为如下几步
 - 特征抽取
    - 图像特性，ppTSM
    - 音频特征，Vggsound
 - proposal提取，BMN
 - LSTM，动作分类 + 回归


## 数据准备
数据集处理代码
```
参考https://github.com/PaddlePaddle/PaddleVideo/tree/application/FootballAction/datasets
```

- 数据集label格式
```
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

- 数据集gts处理, 将原始标注数据处理成如下json格式
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

- 数据集抽帧, 由mp4, 得到frames和pcm, 这里需要添加ffmpeg环境
```
cd datasets/script && python get_frames_pcm.py
```

- 数据预处理后保存格式如下
```
   |--  datasets                   # 训练数据集和处理脚本
        |--  basketball            # xx数据集
            |--  mp4               # 原始视频.mp4
            |--  frames            # 图像帧, fps=5, '.jpg'格式
            |--  pcm               # 音频pcm, 音频采样率16000，采用通道数1
            |--  url.list          # 视频列表
            |--  label_train.json  # 训练集原始gts
            |--  label_val.json    # 验证集原始gts
```


## 模型训练
代码参考足球动作检测：https://github.com/PaddlePaddle/PaddleVideo/tree/application/FootballAction

将该代码库的文件夹 [datasets](https://github.com/PaddlePaddle/PaddleVideo/tree/application/FootballAction/datasets)，[extractor](https://github.com/PaddlePaddle/PaddleVideo/tree/application/FootballAction/extractor)，[train_lstm](https://github.com/PaddlePaddle/PaddleVideo/tree/application/FootballAction/train_lstm)， 拷贝到本代码库复用。

 - image 采样频率fps=5，如果有些动作时间较短，可以适当提高采样频率
 - BMN windows=200，即40s，所以测试自己的数据时，视频时长需大于40s

### 基础镜像
```
docker pull tmtalgo/paddleaction:action-detection-v2
```

### step1 ppTSM训练
我们提供了篮球数据训练的模型，参考checkpoints_basketball。如果使用提供的pptsm模型，可直接跳过下边的pptsm训练数据处理和训练步骤。
如果需要在自己的数据上训练，ppTSM训练代码为：https://github.com/PaddlePaddle/PaddleVideo/tree/release/2.0
ppTSM文档参考：https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/pp-tsm.md

#### step1.1  ppTSM 训练数据处理
由frames结合gts生成训练所需要的正负样本
```
cd ${BasketballAction}
cd datasets/script && python get_instance_for_tsn.py

# 文件名按照如下格式
'{}_{}_{}_{}'.format(video_basename, start_id, end_id, label)
```
完成该步骤后，数据存储位置
```
   |--  datasets                   # 训练数据集和处理脚本
        |--  basketball           # xx数据集
            |--  input_for_tsn     # tsn/tsm训练的数据
```

#### step1.2 ppTSM模型训练
```
# https://github.com/PaddlePaddle/PaddleVideo/tree/release/2.0
cd ${PaddleVideo}
# 修改config.yaml参数修改为 ${BasketballAcation}/configs_train/pptsm_basketball.yaml
python -B -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    --log_dir=$save_dir/logs \
    main.py  \
    --validate \
    -c {BasketballAcation}/configs_train/pptsm_basketball.yaml \
    -o output_dir=$save_dir
```

#### step1.3 ppTSM模型转为预测模式
```
# https://github.com/PaddlePaddle/PaddleVideo/tree/release/2.0
$cd {PaddleVideo}
python tools/export_model.py -c ${BasketballAcation}/configs_train/pptsm_basketball.yaml \
                               -p ${pptsm_train_dir}/checkpoints/models_pptsm/ppTSM_epoch_00057.pdparams \
                               -o {BasketballAcation}/checkpoints/ppTSM
```

####  step1.4 基于ppTSM视频特征提取
image and audio特征提取，保存到datasets features文件夹下
```
cd ${BasketballAcation}
cd extractor && python extract_feat.py
# 特征维度, image(2048) + audio(1024) + pcm(640)
# 特征保存格式如下，将如下dict保存在pkl格式，用于接下来的BMN训练
video_features = {'image_feature': np_image_features,
                  'audio_feature': np_audio_features
                  'pcm_feature': np_pcm_features}
```
完成该步骤后，数据存储位置
```
   |--  datasets                   # 训练数据集和处理脚本
        |--  basketball            # xx数据集
            |--  features          # 视频的图像+音频特征
```


### step2 BMN训练
BMN训练代码为：https://github.com/PaddlePaddle/PaddleVideo/tree/release/2.0
BMN文档参考：https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/localization/bmn.md

#### step2.1 BMN训练数据处理
用于提取二分类的proposal，windows=40，根据gts和特征得到BMN训练所需要的数据集
```
cd ${BasketballAcation}
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
完成该步骤后，数据存储位置
```
   |--  datasets                   # 训练数据集和处理脚本
        |--  basketball            # xx数据集
            |--  input_for_bmn     # bmn训练的proposal         
```

#### step2.2  BMN模型训练
```
# https://github.com/PaddlePaddle/PaddleVideo/tree/release/2.0
cd ${PaddleVideo}
# 修改config.yaml参数修为${BasketballAcation}/configs_train/bmn_basketball.yaml
python -B -m paddle.distributed.launch \
     --gpus="0,1" \
     --log_dir=$out_dir/logs \
     main.py  \
     --validate \
     -c ${BasketballAcation}/configs_train/bmn_basketball.yaml \
     -o output_dir=$out_dir
```

#### step2.3 BMN模型转为预测模式
```
# https://github.com/PaddlePaddle/PaddleVideo/tree/release/2.0
${PaddleVideo}
python tools/export_model.py -c $${BasketballAcation}/configs_train/bmn_basketball.yaml \
                               -p ${bmn_train_dir}/checkpoints/models_bmn/bmn_epoch16.pdparams \
                               -o {BasketballAcation}/checkpoints/BMN
```

#### step2.4  BMN模型预测
得到动作proposal信息： start_id, end_id, score
```
cd ${BasketballAcation}
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
完成该步骤后，数据存储位置
```
   |--  datasets                   # 训练数据集和处理脚本
        |--  basketball            # xx数据集
            |--  feature_bmn
                 |--  prop.json    # bmn 预测结果
```

### step3 LSTM训练
LSTM训练代码为：train_lstm

#### step3.1  LSTM训练数据处理
将BMN得到的proposal截断并处理成LSTM训练所需数据集
```
cd ${BasketballAcation}
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
完成该步骤后，数据存储位置
```
   |--  datasets                   # 训练数据集和处理脚本
        |--  basketball            # xx数据集
            |--  input_for_lstm    # LSTM训练数据集
```

#### step3.2  LSTM训练
```
#conf.yaml修改为 ${BasketballAcation}/configs_train/lstm_basketball.yaml
cd ${BasketballAcation}
python -u scenario_lib/train.py \
    --model_name=ActionNet \
    --config=${BasketballAcation}/configs_train/lstm_basketball.yaml \
    --save_dir=${out_dir}"/models_lstm/" \
    --log_interval=5 \
    --valid_interval=1
```

#### step3.3 LSTM模型转为预测模式
```
${BasketballAcation}
python tools/export_model.py -c ${BasketballAction}/train_lstm/conf/conf.yaml \
                               -p ${lstm_train_dir}/checkpoints/models_lstm/bmn_epoch29.pdparams \
                               -o {BasketballAcation}/checkpoints/LSTM
```


## 模型推理
测试数据格式，可参考使用样例
```
wget https://bj.bcebos.com/v1/acg-algo/PaddleVideo_application/basketball/datasets.tar.gz
```
测试模型，可使用我们提供的模型
```
wget https://bj.bcebos.com/v1/acg-algo/PaddleVideo_application/basketball/checkpoints_basketball.tar.gz
```
运行预测代码
```
cd ${BasketballAction}
cd predict
# 如果使用自己训练的模型，请将各训练过程中转换的inference模型放到predict库
# cp -rf ../checkpoints checkpoints_basketball
python predict.py
```
产出文件
```
${BasketballAction}/predict/results.json
```


## 模型评估
```
cd ${BasketballAction}
cd predict
python eval.py results.json
```


## 模型优化
在实际使用场景中可根据视频内容尝试优化策略
- 可根据动作运动速度，调整抽帧采样率，本代码默认为fps=5
- 统计动作的时间分布，调整bmn采样窗口
- 根据图像和音频的关联程度，调整图像和音频特征的融合方式：本代码将图像特征和音频在时间维度对齐，融合后再进入模型训练。也可尝试分别模型训练后，加权融合等
- 本代码的解决方案也可用于其他动作检测。变换场景后，图像特征重新训练效果更好。音频特征采用的VGGSound训练，如果使用场景仍为生活场景，可直接复用。


## 模型部署
本代码解决方案在动作的检测和召回指标F1-score=80.14%
<div align="center">
  <img src="images/BasketballAction_demo.gif" width="640px"/><br>
</div>


## 参考论文

- [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383.pdf), Ji Lin, Chuang Gan, Song Han
- [BMN: Boundary-Matching Network for Temporal Action Proposal Generation](https://arxiv.org/abs/1907.09702), Tianwei Lin, Xiao Liu, Xin Li, Errui Ding, Shilei Wen.
- [Attention Clusters: Purely Attention Based Local Feature Integration for Video Classification](https://arxiv.org/abs/1711.09550), Xiang Long, Chuang Gan, Gerard de Melo, Jiajun Wu, Xiao Liu, Shilei Wen
- [YouTube-8M: A Large-Scale Video Classification Benchmark](https://arxiv.org/abs/1609.08675), Sami Abu-El-Haija, Nisarg Kothari, Joonseok Lee, Paul Natsev, George Toderici, Balakrishnan Varadarajan, Sudheendra Vijayanarasimhan

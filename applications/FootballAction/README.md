# 足球动作检测模型


## 内容
- [1. 模型简介](#1-模型简介)
- [2. 环境准备](#2-环境准备)
- [3. 数据准备](#3-数据准备)
    - [3.1 数据集简介](#31-数据集简介)
    - [3.2 数据集下载](#32-数据集下载)
    - [3.3 数据预处理](#33-数据预处理)
- [4. 快速体验](#4-快速体验)
- [5. 进阶使用](#5-进阶使用)
    - [5.1 模型训练](#51-模型训练)
    - [5.2 模型推理](#52-模型推理)
    - [5.3 模型评估](#53-模型评估)
    - [5.4 模型优化](#54-模型优化)
    - [5.5 模型部署](#55-模型部署)
- [6. 参考论文](#6-参考论文)

<a name="模型简介"></a>
## 1. 模型简介

FootballAction是基于PaddleVideo实现的足球动作检测算法，用于从足球比赛视频中定位出精彩动作片段发生的起止时间和对应的动作类别。可以定位的足球动作类型包括8种，分别为：
```txt
背景、进球、角球、任意球、黄牌、红牌、换人、界外球
```

我们提出的方案结合PP-TSM、BMN和AttentionLSTM三个模型，图像和音频两种模态进行动作检测，算法整体流程共分为以下三步：
 - 特征抽取
    - 图像特性：PP-TSM
    - 音频特征：VGGish
 - proposal提取：BMN
 - 动作分类 + 回归：AttentionLSTM


AIStudio项目： [基于PP-TSM+BMN+AttentionLSTM实现足球精彩时刻剪辑](https://aistudio.baidu.com/aistudio/projectdetail/3473391?channelType=0&channel=0)

<a name="环境准备"></a>
## 2. 环境准备

- PaddleVideo模型库依赖安装请参考 [安装说明](../../docs/zh-CN/install.md)

<a name="数据准备"></a>
## 3. 数据准备

<a name="数据集简介"></a>
### 3.1 数据集简介

数据集来自欧洲杯2016，共49个足球视频，其中训练集44个，验证集5个。

- 数据集label格式
```
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
```

- 数据集标注文件:
```txt
datasets/EuroCup2016/label_cls8_train.json
datasets/EuroCup2016/label_cls8_val.json
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

<a name="数据集下载"></a>
### 3.2 数据集下载

数据集下载链接: [dataset_url.list](./datasets/EuroCup2016/dataset_url.list)

可使用如下脚本下载：
```
cd datasets/EuroCup2016 && sh download_dataset.sh
```

<a name="数据预处理"></a>
### 3.3 数据预处理

- 数据集抽帧, 由mp4, 得到frames和pcm, 这里需要添加ffmpeg环境
```
cd datasets/script && python get_frames_pcm.py
```


经过以上步骤，得到的代码结构如下所示：

```
|-- FootballAction
   |--  checkpoints                # 模型存放路径
   |--  datasets                   # 数据集和数据处理脚本
        |--  EuroCup2016           # 数据存放路径
            |--  feature_bmn       # bmn提取到的proposal
            |--  features          # image和audio特征, image fps=5, audio 每秒(1024)
            |--  input_for_bmn     # bmn训练的输入数据，widows=40
            |--  input_for_lstm    # lstm训练的输入数据
            |--  input_for_pptsm    # pptsm训练的数据数据
            |--  mp4               # 原始视频.mp4
            |--  frames            # 图像帧, fps=5, '.jpg'格式
            |--  pcm               # 音频pcm, 音频采样率16000，采用通道数1
            |--  url.list          # 视频列表
            |--  url_val.list          # 视频列表
            |--  label_cls8_train.json  # 训练集原始gts
            |--  label_cls8_val.json    # 验证集原始gts
            |--  label.json        # 动作label
        |--  script                # 数据集处理脚本
    |--  predict                   # 模型预测代码
    |--  extractor                 # 特征提取脚本
    |--  train_lstm                # lstm训练代码
    |--  train_proposal            # pptsm、bmn训练代码
        |--  configs               # pptsm、bmn配置文件
```

<a name="快速体验"></a>
## 4. 快速体验

首先，通过以下命令，下载训练好的模型文件：
```bash
cd checkpoints
sh  download.sh
```

运行预测代码：
```
cd ${FootballAction_root}/predict && python predict.py
```
产出文件：results.json


<a name="进阶使用"></a>
## 5. 进阶使用

<a name="模型训练"></a>
### 5.1 模型训练

采样方式：
- image 采样频率fps=5，如果有些动作时间较短，可以适当提高采样频率
- BMN windows=200，即40s，所以测试自己的数据时，视频时长需大于40s

请先参考[使用说明](../../docs/zh-CN/contribute/usage.md)了解PaddleVideo模型库的使用。

#### step1 PP-TSM训练

PP-TSM模型使用文档参考[PP-TSM](../../docs/zh-CN/model_zoo/recognition/pp-tsm.md)

##### step1.1  PP-TSM 训练数据处理

使用如下命令结合frames和gts生成训练所需要的正负样本:
```bash
cd datasets/script && python get_instance_for_pptsm.py
```

完成该步骤后，数据存储位置
```
   |--  datasets                   # 数据集和数据处理脚本
        |--  EuroCup2016           # 数据存放路径
            |--  input_for_pptsm   # pptsm训练的数据
```

文件按照如下格式命名：
```
'{}_{}_{}_{}'.format(video_basename, start_id, end_id, label)
```

##### step1.2 PP-TSM模型训练
训练启动命令如下：
```bash
cd ${FootballAction_root}
cd ../..  #进入PaddleVideo目录下

python -B -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    --log_dir=./football/logs_pptsm \
    main.py  \
    --validate \
    -c applications/FootballAction/train_proposal/configs/pptsm_football_v2.0.yaml  \
    -o output_dir=./football/pptsm
```

我们也提供了训练好的PP-TSM模型，下载链接已在快速体验章节中给出。

##### step1.3 导出PP-TSM推理模型
在转为预测模式前，需要修改 `PaddleVideo/paddlevideo/modeling/framework/recognizers/recognizer2d.py` 文件，将 init 和 infer_step 函数分别更新为如下代码：

```python
    def __init__(self, backbone=None, head=None):
        super().__init__(backbone=backbone, head=head)
        self.avgpool2d = paddle.nn.AdaptiveAvgPool2D((1, 1), data_format='NCHW')

    def infer_step(self, data_batch):
        """Define how the model is going to test, from input to output."""
        imgs = data_batch[0]
        imgs = paddle.reshape_(imgs, [-1] + list(imgs.shape[2:]))
        feature = self.backbone(imgs)
        feat = self.avgpool2d(feature)
        return feat
```
再执行如下命令：

```bash
cd ${PaddleVideo_root}
python tools/export_model.py -c applications/FootballAction/train_proposal/configs/pptsm_football_v2.0.yaml \
                             -p ./football/pptsm/ppTSM_best.pdparams \
                             -o ./football/inference_model
```

#####  step1.4  基于PP-TSM的视频特征提取

将 `PaddleVideo/applications/FootballAction/predict/action_detect/models/pptsm_infer.py` 文件中41行的
```python
self.output_tensor = self.predictor.get_output_handle(output_names[1])
```
替换为
```python
self.output_tensor = self.predictor.get_output_handle(output_names[0])
```


使用如下命令进行image和audio特征的提取，默认使用下载的模型进行特征提取，如果使用自己数据训练的模型，请注意修改配置文件中模型的文件路径:
```bash
cd ${FootballAcation}
cd extractor && python extract_feat.py
```

完成该步骤后，数据存储位置
```
   |--  datasets                   # 训练数据集和处理脚本
        |--  EuroCup2016            # 数据集
            |--  features          # 视频的图像+音频特征
```


推理特征以pkl文件保存，格式如下：
```txt
# 特征维度, image(2048) + audio(1024)
video_features = {'image_feature': np_image_features,
                  'audio_feature': np_audio_features}
```
此特征接下来会用于BMN模型的训练。


#### step2 BMN训练

BMN模型使用文档参考[BMN](../../docs/zh-CN/model_zoo/localization/bmn.md)

##### step2.1 BMN训练数据处理
使用如下命令得到BMN训练所需要的数据集，默认使用windows=40，根据gts和特征得到训练所需的proposal：
```bash
cd FootballAction/datasets/script && python get_instance_for_bmn.py
```

完成该步骤后，数据存储位置
```
   |--  datasets                   # 训练数据集和处理脚本
        |--  EuroCup2016            # 数据集
            |--  input_for_bmn     # bmn训练的proposal
                |--  feature
                |--  label.json  
```

特征文件保存在`label.json`文件中，数据格式如下：
```txt
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

##### step2.2  BMN模型训练
训练启动命令如下：
```bash
python -B -m paddle.distributed.launch \
     --gpus="0,1" \
     --log_dir=./football/logs_bmn \
     main.py  \
     --validate \
     -c applications/FootballAction/train_proposal/configs/bmn_football_v2.0.yaml \
     -o output_dir=./football/bmn
```

我们也提供了训练好的BMN模型，下载链接已在快速体验章节中给出。

##### step2.3 导出BMN推理模型
模型导出命令如下:
```bash
python tools/export_model.py -c applications/FootballAction/train_proposal/configs/bmn_football_v2.0.yaml \
                              -p ./football/bmn/BMN_epoch_00016.pdparams \
                               -o ./football/inference_model
```

##### step2.4  BMN模型预测
使用如下命令进行预测，得到动作proposal信息： start_id, end_id, score。如果使用自己数据训练的模型，请注意修改配置文件中模型的文件路径:
```
cd extractor && python extract_bmn.py
```

完成该步骤后，数据存储位置
```
   |--  datasets                   # 训练数据集和处理脚本
        |--  EuroCup2016            # 数据集
            |--  feature_bmn
                 |--  prop.json    # bmn 预测结果
```

预测结果数据格式如下：
```txt
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

#### step3 LSTM训练

AttentionLSTM模型使用文档参考[AttentionLSTM](../../docs/zh-CN/model_zoo/localization/bmn.md)，此处我们对原始对AttentionLSTM模型进行了改进，包括：

1. 不同模态特征在LSTM中使用不同的hiddne_size
2. 加入了一个回归分支用于回归iou
3. 模型中加入了BN层抑制过拟合


##### step3.1  LSTM训练数据处理
将BMN得到的proposal截断并处理成LSTM训练所需数据集。同理，注意数据集文件修改路径。
```
cd datasets/script && python get_instance_for_lstm.py
```

完成该步骤后，数据存储位置
```
   |--  datasets                    # 训练数据集和处理脚本
        |--  EuroCup2016            # 数据集
            |--  input_for_lstm     # lstm训练的proposal
                ├── feature         # 特征
                ├── label_info.json # 标签信息
                ├── train.txt       # 训练文件列表
                └── val.txt         # 测试文件列表
```

- `label_info.json`数据格式如下：
```
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
```

- LSTM训练所需要的feature数据格式如下:
```
{
    'features': np.array(feature_hit, dtype=np.float32),    # iamge和audio 特征
    'feature_fps': 5,                                       # fps = 5
    'label_info': {'norm_iou': 0.5, 'label': 3, ...},       # 数据格式1中的'proposal_actions'
    'video_name': 'c9516c903de3416c97dae91a59e968d7'        # video_name
}
```

- LSTM训练所需文件列表数据格式如下：
```
'{} {}'.format(filename, label)
```

##### step3.2  LSTM训练

训练启动命令如下:

```bash
python -B -m paddle.distributed.launch \
     --gpus="0,1,2,3" \
     --log_dir=./football/logs_lstm \
     main.py  \
     --validate \
     -c applications/FootballAction/train_proposal/configs/lstm_football.yaml \
     -o output_dir=./football/lstm
```

##### step3.3 导出LSTM推理模型

模型导出命令如下:
```bash
python tools/export_model.py -c applications/FootballAction/train_proposal/configs/lstm_football.yaml \
                              -p ./football/lstm/AttentionLSTM_best.pdparams  \
                               -o ./football/inference_model
```

<a name="模型推理"></a>
### 5.2 模型推理

运行预测代码
```
cd predict && python predict.py
```
- 默认使用我们提供的于训练文件进行预测，如使用个人训练的模型文件，请对应修改[配置文件](./predict/configs/configs.yaml)中的参数路径
- 产出文件：results.json


<a name="模型评估"></a>
### 5.3 模型评估

```
# 包括bmn proposal 评估和最终action评估
cd predict && python eval.py results.json
```

<a name="模型优化"></a>
### 5.4 模型优化

- 基础特征模型（图像）替换为PP-TSM，准确率由84%提升到94%
- 基础特征模型（音频）没变动
- 准确率提升，precision和recall均有大幅提升，F1-score从0.57提升到0.82


<a name="模型部署"></a>
### 5.5 模型部署

本代码解决方案在动作的检测和召回指标F1-score=82%


<a name="参考论文"></a>
### 6. 参考论文

- [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383.pdf), Ji Lin, Chuang Gan, Song Han
- [BMN: Boundary-Matching Network for Temporal Action Proposal Generation](https://arxiv.org/abs/1907.09702), Tianwei Lin, Xiao Liu, Xin Li, Errui Ding, Shilei Wen.
- [Attention Clusters: Purely Attention Based Local Feature Integration for Video Classification](https://arxiv.org/abs/1711.09550), Xiang Long, Chuang Gan, Gerard de Melo, Jiajun Wu, Xiao Liu, Shilei Wen
- [YouTube-8M: A Large-Scale Video Classification Benchmark](https://arxiv.org/abs/1609.08675), Sami Abu-El-Haija, Nisarg Kothari, Joonseok Lee, Paul Natsev, George Toderici, Balakrishnan Varadarajan, Sudheendra Vijayanarasimhan

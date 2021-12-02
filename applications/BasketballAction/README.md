
# 版本说明
 - v1.1 pptsm, vggish, bmn, lstm, 基于paddle-2.0环境


# 简介
该代码库用于体育动作检测+识别, 基于paddle2.0版本开发，结合PaddleVideo中的ppTSM, BMN, attentionLSTM的多个视频模型进行视频时空二阶段检测算法。
主要分为如下几步
 - 特征抽取
    - 图像特性，ppTSM
    - 音频特征，Vggsound
 - proposal提取，BMN
 - LSTM，动作分类 + 回归

# 基础镜像
```
docker pull tmtalgo/paddleaction:action-detection-v2
```

# 数据集
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

# 简单说明
 - image 采样频率fps=5，如果有些动作时间较短，可以适当提高采样频率
 - BMN windows=200，即40s，所以测试自己的数据时，视频时长需大于40s

# 代码结构
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
# 训练与评估步骤
```
参考足球动作检测的处理：https://github.com/PaddlePaddle/PaddleVideo/tree/application/FootballAction
```
# 预测步骤
```
cd predict 
# 测试数据处理，可使用样例：
wget https://bj.bcebos.com/v1/acg-algo/PaddleVideo_application/basketball/datasets.tar.gz
# 模型下载
wget https://bj.bcebos.com/v1/acg-algo/PaddleVideo_application/basketball/checkpoints_basketball.tar.gz
# 运行预测代码
python predict.py
```


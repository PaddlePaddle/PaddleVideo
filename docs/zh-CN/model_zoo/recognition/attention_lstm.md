简体中文 | [English](../../../en/model_zoo/recognition/attention_lstm.md)

# AttentionLSTM

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型推理](#模型推理)
- [参考论文](#参考论文)

## 模型简介

循环神经网络（RNN）常用于序列数据的处理，可建模视频连续多帧的时序信息，在视频分类领域为基础常用方法。
该模型采用了双向长短时记忆网络（LSTM），将视频的所有帧特征依次编码。与传统方法直接采用LSTM最后一个时刻的输出不同，该模型增加了一个Attention层，每个时刻的隐状态输出都有一个自适应权重，然后线性加权得到最终特征向量。参考论文中实现的是两层LSTM结构，而**本模型实现的是带Attention的双向LSTM**。

Attention层可参考论文[AttentionCluster](https://arxiv.org/abs/1711.09550)

## 数据准备

PaddleVide提供了在Youtube-8M数据集上训练和测试脚本。Youtube-8M数据下载及准备请参考[YouTube-8M数据准备](../../dataset/youtube8m.md)

## 模型训练

### Youtube-8M数据集训练

#### 开始训练

- Youtube-8M数据集使用8卡训练，feature格式下会使用视频和音频特征作为输入，数据的训练启动命令如下

  ```bash
  python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_attetion_lstm  main.py  --validate -c configs/recognition/attention_lstm/attention_lstm_youtube8m.yaml
  ```

## 模型测试

命令如下：

```bash
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_attetion_lstm  main.py  --test -c configs/recognition/attention_lstm/attention_lstm_youtube8m.yaml -w "output/AttentionLSTM/AttentionLSTM_best.pdparams"
```

当测试配置采用如下参数时，在Youtube-8M的validation数据集上的测试指标如下：

| Hit@1 | PERR | GAP  | checkpoints |
| :-----: | :---------: | :---: | ----- |
|  89.05  | 80.49 | 86.30 |   [AttentionLSTM_yt8.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/AttentionLSTM_yt8.pdparams)      |

## 模型推理

### 导出inference模型

```bash
python3.7 tools/export_model.py -c configs/recognition/attention_lstm/attention_lstm_youtube8m.yaml \
                                -p data/AttentionLSTM_yt8.pdparams \
                                -o inference/AttentionLSTM
```

上述命令将生成预测所需的模型结构文件`AttentionLSTM.pdmodel`和模型权重文件`AttentionLSTM.pdiparams`。

各参数含义可参考[模型推理方法](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-模型推理)

### 使用预测引擎推理

```bash
python3.7 tools/predict.py --input_file data/example.pkl \
                           --config configs/recognition/attention_lstm/attention_lstm_youtube8m.yaml \
                           --model_file inference/AttentionLSTM/AttentionLSTM.pdmodel \
                           --params_file inference/AttentionLSTM/AttentionLSTM.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```
输出示例如下：
```bash
Current video file: data/example.pkl
        top-1 class: 11
        top-1 score: 0.9841002225875854
```
可以看到，使用在Youtube-8M上训练好的AttentionLSTM模型对data/example.pkl进行预测，输出的top1类别id为11，置信度为0.98。
## 参考论文

- [Attention Clusters: Purely Attention Based Local Feature Integration for Video Classification](https://arxiv.org/abs/1711.09550), Xiang Long, Chuang Gan, Gerard de Melo, Jiajun Wu, Xiao Liu, Shilei Wen
- [YouTube-8M: A Large-Scale Video Classification Benchmark](https://arxiv.org/abs/1609.08675), Sami Abu-El-Haija, Nisarg Kothari, Joonseok Lee, Paul Natsev, George Toderici, Balakrishnan Varadarajan, Sudheendra Vijayanarasimhan


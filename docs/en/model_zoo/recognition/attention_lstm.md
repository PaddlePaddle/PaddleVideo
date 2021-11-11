[简体中文](../../../zh-CN/model_zoo/recognition/attention_lstm.md) | English

# AttentionLSTM

## content

- [Introduction](#Introduction)
- [Data](#Data)
- [Train](#Train)
- [Test](#Test)
- [Inference](#Inference)
- [Reference](#Reference)

## Introduction

Recurrent Neural Networks (RNN) are often used in the processing of sequence data, which can model the sequence information of multiple consecutive frames of video, and are commonly used methods in the field of video classification.
This model uses a two-way long and short-term memory network (LSTM) to encode all the frame features of the video in sequence. Unlike the traditional method that directly uses the output of the last moment of LSTM, this model adds an Attention layer, and the hidden state output at each moment has an adaptive weight, and then linearly weights the final feature vector. The reference paper implements a two-layer LSTM structure, while **this model implements a two-way LSTM with Attention**.

The Attention layer can refer to the paper [AttentionCluster](https://arxiv.org/abs/1711.09550)

## Data

PaddleVide provides training and testing scripts on the Youtube-8M dataset. Youtube-8M data download and preparation please refer to [YouTube-8M data preparation](../../dataset/youtube8m.md)

## Train

### Youtube-8M data set training

#### Start training

- The Youtube-8M data set uses 8 cards for training. In the feature format, video and audio features will be used as input. The training start command of the data is as follows

  ```bash
  python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_attetion_lstm main.py --validate -c configs/recognition/attention_lstm/attention_lstm_youtube8m.yaml
  ```

## Test

The command is as follows:

```bash
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_attetion_lstm main.py --test -c configs/recognition/attention_lstm/attention_lstm_youtube8m.yaml -w "output/AttentionLSTM/AttentionLSTM_best.pdparams"
```

When the test configuration uses the following parameters, the test indicators on the validation data set of Youtube-8M are as follows:

| Hit@1 | PERR | GAP | checkpoints |
| :-----: | :---------: | :---: | ----- |
| 89.05 | 80.49 | 86.30 | [AttentionLSTM_yt8.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/AttentionLSTM_yt8.pdparams) |

## Inference

### Export inference model
```bash
python3.7 tools/export_model.py -c configs/recognition/attention_lstm/attention_lstm_youtube8m.yaml \
                                -p data/AttentionLSTM_yt8.pdparams \
                                -o inference/AttentionLSTM
```

The above command will generate the model structure file `AttentionLSTM.pdmodel` and the model weight file `AttentionLSTM.pdiparams` required for prediction.

For the meaning of each parameter, please refer to [Model Reasoning Method](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0.0/docs/en/start.md#2-infer)

### Use prediction engine inference

```bash
python3.7 tools/predict.py --input_file data/example.pkl \
                           --config configs/recognition/attention_lstm/attention_lstm_youtube8m.yaml \
                           --model_file inference/AttentionLSTM/AttentionLSTM.pdmodel \
                           --params_file inference/AttentionLSTM/AttentionLSTM.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```
An example of the output is as follows:
```bash
Current video file: data/example.pkl
         top-1 class: 11
         top-1 score: 0.9841002225875854
```
It can be seen that using the AttentionLSTM model trained on Youtube-8M to predict data/example.pkl, the output top1 category id is 11, and the confidence is 0.98.
## Reference paper

- [Attention Clusters: Purely Attention Based Local Feature Integration for Video Classification](https://arxiv.org/abs/1711.09550), Xiang Long, Chuang Gan, Gerard de Melo, Jiajun Wu, Xiao Liu, Shilei Wen
- [YouTube-8M: A Large-Scale Video Classification Benchmark](https://arxiv.org/abs/1609.08675), Sami Abu-El-Haija, Nisarg Kothari, Joonseok Lee, Paul Natsev, George Toderici, Balakrishnan Varadarajan, Sudheendra Vijayanarasimhan

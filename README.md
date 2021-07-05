ndndndn`[简体中文](README_cn.md) | English

# PaddleVideo

## Introduction

![python version](https://img.shields.io/badge/python-3.7+-orange.svg) ![paddle version](https://img.shields.io/badge/PaddlePaddle-2.0-blue)


PaddleVideo is a toolset for video recognition, action localization, and spatio temporal action detection tasks prepared for the industry and academia. This repository provides examples and best practice guildelines for exploring deep learning algorithm in the scene of video area. We devote to support experiments and utilities which can significantly reduce the "time to deploy". By the way, this is also a proficiency verification and implementation of the newest PaddlePaddle 2.0 in the video field.

<div align="center">
  <img src="docs/images/home.gif" width="450px"/><br>
</div>

> If you think this repo is helpful to you, welcome to star us~


## Features

- **Various dataset and models**
    PaddleVideo supports more datasets and models, including [Kinectics400](docs/zh-CN/dataset/k400.md), ucf101, YoutTube8M datasets, and video recognition models, such as TSN, TSM, SlowFast, AttentionLSTM and action localization model, like [BMN](./docs/zh-CN/model_zoo/localization/bmn.md).

- **Higher performance**
    PaddleVideo has built-in solutions to improve accuracy on recognition models. [PP-TSM](docs/zh-CN/model_zoo/recognition/pp-tsm.md), which is based on the standard TSM, already archive the best performance in the 2D recognition network, has the same size of parameters but improve the Top1 Acc to 76.16%.

- **Faster training strategy**
    PaddleVideo suppors faster training strategy, such as [AMP training](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html), Distributed training, Multigrid method for Slowfast, OP fusion method, Faster reader and so on.

- **Deployable**
    PaddleVideo is powered by the Paddle Inference. There is no need to convert the model to ONNX format when deploying it, all you want can be found in this repository.

- **Applications**
    PaddleVideo provides some interesting and practical projects that are implemented using video recognition and detection techniques, such as [FootballAction](https://github.com/PaddlePaddle/PaddleVideo/tree/application/FootballAction) and VideoTag.

### Overview of the performance

| Field | Model | Dataset | Metrics | ACC% |
| :--------------- | :--------: | :------------: | :------------: | :------------: |
| action recgonition | [**PP-TSM**](./docs/zh-CN/model_zoo/recognition/pp-tsm.md) | [Kinetics-400](./docs/zh-CN/dataset/k400.md) | Top-1 | **76.16** |
| action recgonition | [**PP-TSN**](./docs/zh-CN/model_zoo/recognition/pp-tsn.md) | [Kinetics-400](./docs/zh-CN/dataset/k400.md) | Top-1 | **73.68** |
| action recgonition | [SlowFast](./docs/zh-CN/model_zoo/recognition/slowfast.md) | [Kinetics-400](./docs/zh-CN/dataset/k400.md) | Top-1 | 75.84 |
| action recgonition | [TSM](./docs/zh-CN/model_zoo/recognition/tsm.md) | [Kinetics-400](./docs/zh-CN/dataset/k400.md) | Top-1 | 71.06 |
| action recgonition | [TSN](./docs/zh-CN/model_zoo/recognition/tsn.md) | [Kinetics-400](./docs/zh-CN/dataset/k400.md) | Top-1 | 69.81 |
| action recgonition | [AttentionLSTM](./docs/zh-CN/model_zoo/recognition/attention_lstm.md) | [Youtube-8M](./docs/zh-CN/dataset/youtube8m.md) | Hit@1 | 89.0 |
| action detection| [BMN](./docs/zh-CN/model_zoo/localization/bmn.md) | [ActivityNet](./docs/zh-CN/dataset/ActivityNet.md) |  AUC | 67.23 |

### Changelog

release/2.1 was released in 20/05/2021. Please refer to [release notes](https://github.com/PaddlePaddle/PaddleVideo/releases) for details.

Plan
- ActBert
- TimeSformer

<a name="Community"></a>
## Community
- Scan the QR code below with your Wechat and reply "video", you can access to official technical exchange group. Look forward to your participation.

<div align="center">
<img src="./docs/images/joinus.PNG"  width = "200" height = "200" />
</div>

## Applications

- [VideoTag](https://github.com/PaddlePaddle/PaddleVideo/tree/application/VideoTag): 3k Large-Scale video classification model

<div align="center">
  <img src="docs/images/VideoTag.gif" width="450px"/><br>
</div>


- [FootballAction](https://github.com/PaddlePaddle/PaddleVideo/tree/application/FootballAction): Football action detection model

<div align="center">
  <img src="docs/images/FootballAction.gif" width="450px"/><br>
</div>


## Tutorials and Docs

- Tutorials and Slides
  - [2021.01](https://aistudio.baidu.com/aistudio/course/introduce/6742)
  - [Summarize of video understanding](docs/en/tutorials/summarize.md)
- Quick Start
  - [Install](docs/en/install.md)
  - [Start](docs/en/start.md)
- Project design
  - [Modular design](docs/en/tutorials/modular_design.md)
  - [Configuration design](docs/en/tutorials/config.md)
- Model zoo
  - [recognition](docs/en/model_zoo/README.md)
    - [Attention-LSTM](docs/en/model_zoo/recognition/attention_lstm.md)
    - [TSN](docs/en/model_zoo/recognition/tsn.md)
    - [TSM](docs/en/model_zoo/recognition/tsm.md)
    - [PP-TSM](docs/en/model_zoo/recognition/pp-tsm.md)
    - [PP-TSN](docs/en/model_zoo/recognition/pp-tsn.md)
    - [SlowFast](docs/en/model_zoo/recognition/slowfast.md)
  - [Localization](docs/en/model_zoo/README.md)
    - [BMN](docs/en/model_zoo/localization/bmn.md)
  - Spatio temporal action detection
    - Coming Soon!  
  - ActBERT: Learning Global-Local Video-Text Representations
    - Coming Soon!  
- Practice
  - [Higher performance PP-TSM](docs/en/tutorials/pp-tsm.md)
  - [Accelerate training](docs/en/tutorials/accelerate.md)
  - [Deployment](docs/en/tutorials/deployment.md)
- Others
  - [Benchmark](docs/en/benchmark.md)
  - [Tools](docs/en/tools.md)

## License

PaddleVideo is released under the [Apache 2.0 license](LICENSE).


## Contributing
This poject welcomes contributions and suggestions. Please see our [contribution guidelines](docs/CONTRIBUTING.md).

- Many thanks to [mohui37](https://github.com/mohui37) for contributing the code for prediction.

## [Call for suggestions](https://github.com/PaddlePaddle/PaddleVideo/issues/68)

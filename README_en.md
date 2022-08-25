[简体中文](README.md) | English

# PaddleVideo

![python version](https://img.shields.io/badge/python-3.7+-orange.svg) ![paddle version](https://img.shields.io/badge/PaddlePaddle-2.0-blue)

## Introduction

PaddleVideo is a toolset for video tasks prepared for the industry and academia. This repository provides examples and best practice guildelines for exploring deep learning algorithm in the scene of video area.

<div align="center">
  <img src="docs/images/home.gif" width="450px"/><br>
</div>


## Update:

- release **🔥[PP-TSMv2](./docs/zh-CN/model_zoo/recognition/pp-tsm.md)**, an lite action recognition model, top1_acc on Kinetics-400 is 74.38%，cpu inference time on 10s video with 25fps is only 433ms. [benchmark](./docs/zh-CN/benchmark.md).
- add [Knowledge Distilltion](./docs/zh-CN/distillation.md) framework code.
- add [TokenShift](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/tokenshift_transformer.md), [2s-ACGN](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/agcn2s.md) and [CTR-GCN](./docs/zh-CN/model_zoo/recognition/ctrgcn.md) model.

​ 💖 **Welcome to scan the code and join the group discussion** 💖

<div align="center">
  <img src="docs/images/user_group.png" width=250/></div>

- Scan the QR code below with your Wechat and reply "video", you can access to official technical exchange group. Look forward to your participation.

## Features
PaddleVideo support a variety of cutting-edge algorithms related to video, and developed industrial featured models/solution [PP-TSM](docs/zh-CN/model_zoo/recognition/pp-tsm.md) and [PP-TSMv2](docs/zh-CN/model_zoo/recognition/pp-tsm.md) on this basis, and get through the whole process of data production, model training, compression, inference and deployment.

<div align="center">
    <img src="./docs/images/features_en.png" width="700">
</div>

## Quick Start

- One line of code quick use: [Quick Start](./docs/zh-CN/quick_start.md)

## Tutorials


- [Quick Start](./docs/zh-CN/quick_start.md)
- [Installation](./docs/zh-CN/install.md)
- [Usage](./docs/zh-CN/usage.md)
- [PP-TSM🔥](./docs/zh-CN/model_zoo/recognition/pp-tsm.md)
  - [Model Zoo](./docs/zh-CN/model_zoo/recognition/pp-tsm.md#7)
  - [Model training](./docs/zh-CN/model_zoo/recognition/pp-tsm.md#4)
  - [Model Compression](./deploy/slim/)
      - [Model Quantization](./deploy/slim/readme.md)
      - [Knowledge Distillation](./docs/zh-CN/distillation.md)
  - [Inference and Deployment](./deploy/)
      - [Python Inference](./docs/zh-CN/model_zoo/recognition/pp-tsm.md#62)
      - [C++ Inference](./deploy/cpp_infer/readme.md)
      - [Serving](./deploy/python_serving/readme.md)
      - [Paddle2ONNX](./deploy/paddle2onnx/readme.md)
      - [Benchmark](./docs/zh-CN/benchmark.md)
- [Academic algorithms](./docs/en/model_zoo/README.md)🚀
- [Datasets](./docs/en/dataset/README.md)
- [Data Annotation](./applications/BILS)
- [Contribute](./docs/zh-CN/contribute/README.md)

## License

PaddleVideo is released under the [Apache 2.0 license](LICENSE).

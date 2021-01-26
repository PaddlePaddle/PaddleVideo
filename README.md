[简体中文](README_cn.md) | English

# PaddleVideo

## Introduction

![python version](https://img.shields.io/badge/python-3.7+-orange.svg) ![paddle version](https://img.shields.io/badge/PaddlePaddle-2.0-blue)


PaddleVideo is a toolset for video recognition, action localization, and spatio temporal action detection tasks prepared for the industry and academia. This repository provides examples and best practice guildelines for exploring deep learning algorithm in the scene of video area. We devote to support experiments and utilities which can significantly reduce the "time to deploy". By the way, this is also a proficiency verification and implementation of the newest PaddlePaddle 2.0 in the video field.

<div align="center">
  <img src="docs/images/home.gif" width="450px"/><br>
</div>

## Feature

- **Advanced model zoo design**
    PaddleVideo unifies the video understanding tasks, including recogniztion, localization, spatio temporal action detection, and so on. with the clear configuration system based on IOC/DI, we design a decoupling modular and extensible framework which can easily construct a customized network by combining different modules.

- **Various dataset and architectures**
    PaddleVideo supports more datasets and architectures, including Kinectics400, ucf101, YoutTube8M datasets, and video recognition models, such as TSN, TSM, SlowFast, AttentionLSTM and action localization model, like BMN.

- **Higher performance**
    PaddleVideo has built-in solutions to improve accuracy on the recognition models. PP-TSM, which is based on the standard TSM, already archive the best performance in the 2D recognition network, has the same size of parameters but improve the Top1 Acc to **73.5%** , and one can easily apply the soulutions on his own dataset.

- **Faster training strategy**
    PaddleVideo suppors faster training strategy, it accelerates by 100% compared with the standard Slowfast version, and it only takes 10 days to train from scratch on the kinetics400 dataset.

- **Deployable**
    PaddleVideo is powered by the Paddle Inference. There is no need to convert the model to ONNX format when deploying it, all you want can be found in this repository.

### Overview of the kit structures

<table>
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Architectures</b>
      </td>
      <td>
        <b>Frameworks</b>
      </td>
      <td>
        <b>Components</b>
      </td>
      <td>
        <b>Data Augmentation</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul><li><b>Recognition</b></li>
          <ul>
            <li>TSN</li>
            <li>TSM</li>
            <li>SlowFast</li>
            <li>PP-TSM</li>
            <li>VideoTag</li>
            <li>AttentionLSTM</li>
          </ul>
        </ul>
        <ul><li><b>Localization</b></li>
          <ul>
            <li>BMN</li>
          </ul>
        </ul>
      </td>
      <td>
          <li>Recognizer1D</li>
          <li>Recognizer2D</li>
          <li>Recognizer3D</li>
          <li>Localizer</li>
        <HR></HR>
        <ul>Backbone
            <li>resnet</li>
            <li>resnet_tsm</li>
            <li>resnet_tweaks_tsm</li>
            <li>bmn</li>
        </ul>
        <ul>Head
            <li>tsm_head</li>
            <li>tsn_head</li>
            <li>bmn_head</li>
            <slowfast_head></li>
            <bmn_head></li>
        </ul>
      </td>
      <td>
        <ul><li><b>Solver</b></li>
          <ul><li><b>Optimizer</b></li>
              <ul>
                <li>Momentum</li>
                <li>RMSProp</li>
              </ul>
          </ul>
          <ul><li><b>LearningRate</b></li>
              <ul>
                <li>PiecewiseDecay</li>
              </ul>
          </ul>
        </ul>
        <ul><li><b>Loss</b></li>
          <ul>
            <li>CrossEntropy</li>
            <li>BMNLoss</li>  
          </ul>  
        </ul>  
        <ul><li><b>Metrics</b></li>
          <ul>
            <li>CenterCrop</li>
            <li>MultiCrop</li>  
          </ul>  
        </ul>
      </td>
      <td>
        <ul><li><b>Video</b></li>
          <ul>
            <li>Mixup</li>
            <li>Cutmix</li>  
          </ul>  
        </ul>
        <ul><li><b>Image</b></li>
            <ul>
                <li>Scale</li>
                <li>Random FLip</li>
                <li>Jitter Scale</li>  
                <li>Crop</li>
                <li>MultiCrop</li>
                <li>Center Crop</li>
                <li>MultiScaleCrop</li>
                <li>Random Crop</li>
                <li>PackOutput</li>
            </ul>
         </ul>
      </td>  
    </tr>


</td>
    </tr>
  </tbody>
</table>

### Overview of the performance

The chart below illustrates the performance of the video recognition models both 2D and 3D architectures, including our implementation and Pytorch version. It shows the relationship between Acc Top1 and VPS on the Kinectics400 dataset. (Tested on the NVIDIA® Tesla® GPU V100.)

<div align="center">
  <img src="docs/images/acc_vps.jpeg" />
</div>

**Note：**
- PP-TSM improves almost 3.5% Top1 accuracy from standard TSM.
- all these models described by RED color can be obtained in the [Model Zoo](https://github.com/PaddlePaddle/PaddleVideo/blob/main/docs/en/model_zoo/README.md), and others are Pytorch results.

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


- Quick Start
  - [Summarize of video understanding](docs/en/tutorials/summarize.md)
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
    - [SlowFast](docs/en/model_zoo/recognition/slowfast.md)
  - [Localization](docs/en/model_zoo/README.md)
    - [BMN](docs/en/model_zoo/localization/bmn.md)
  - Spatio temporal action detection
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

[简体中文](../../../zh-CN/model_zoo/partition/transnetv2.md) | English

# TransNetV2

## Contents

- [Introduction](#Introduction)
- [Data](#Data)
- [Train](#Train)
- [Test](#Test)
- [Inference](#Inference)
- [Details](#Details)
- [Reference](#Reference)

Before getting started, you need to install additional dependencies as follows:
```bash
python -m pip install ffmpeg-python==0.2.0
```

## Introduction

TransNetV2 is a video segmentation model based on deep learning. It performs feature learning through the DDCNN V2 structure, and adds RGB color histograms and video frame similarity for more effective feature extraction, and finally obtains whether each frame is a shot boundary frame Probability, thereby completing the video segmentation. The algorithm has good effect and efficient calculation, which is very suitable for industrial landing.

![](../../../images/transnetv2.png)

This code currently only supports model inference, and model training and testing will be provided in the future.

Please refer to the paper for details. [TransNet V2: An effective deep network architecture for fast shot transition detection](https://arxiv.org/abs/2008.04838)

## Data

coming soon


## Train

coming soon


## Test

coming soon


## Inference


Load the TransNetV2 weights trained on ClipShots and TRECVID IACC.3 dataset [TransNetV2_shots.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/TransNetV2_shots.pdparams), or download through the command line

```bash
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.2/TransNetV2_shots.pdparams
```

### export inference model

```bash
python3.7 tools/export_model.py -c configs/partitioners/transnetv2/transnetv2.yaml -p data/TransNetV2_shots.pdparams -o inference/TransNetV2
```

The above command will generate the model structure file`TransNetV2.pdmodel`and the model weight file`TransNetV2.pdiparams`required for prediction.

For the meaning of each parameter, please refer to [Model Reasoning Method](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-Model Reasoning)

### infer

```bash
python3.7 tools/predict.py --input_file data/example.avi \
                           --config configs/partitioners/transnetv2/transnetv2.yaml \
                           --model_file inference/TransNetV2/TransNetV2.pdmodel \
                           --params_file inference/TransNetV2/TransNetV2.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

By defining the `output_path` parameters in `transnetv2.yaml`, the prediction probability of each frame can be output to `{output_path}/example_predictions.txt`, and the predicted lens boundary is output to `{output_path}/example_scenes.txt`.
By defining the `visualize` parameter in `transnetv2.yaml`, the predicted results can be visualized, and the visual results are saved to `{output_path}/example_vis.png`.

## Reference

- [TransNet V2: An effective deep network architecture for fast shot transition detection](https://arxiv.org/abs/2008.04838), Tomáš Souček, Jakub Lokoč

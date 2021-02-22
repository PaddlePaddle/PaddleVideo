[简体中文](../../../zh-CN/model_zoo/recognition/tsn_dali.md) | English

# TSN DALI

- [Introduction](#Introduction)
- [Requirement](#Requirement)
- [Data](#Data)
- [Train](#Train)
- [Test](#Test)
- [Inference](#Inference)
- [Reference](#Reference)

## Introduction

We aims to speed up TSN model training using DALI in this code. As [nvidia DALI](https://github.com/NVIDIA/DALI) not support TSN sampling way, we reimplemented segment sampling in VideoReader.

### Performance

Test Environment: 
```
Card: Tesla v100
Memory: 4 * 16G
Cuda: 9.0
batch_size of single card: 32
```

| Training way | batch cost/s  | reader cost/s | ips:instance/sec | Speed up |
| :--------------- | :--------: | :------------: | :------------: | :------------: |
| DALI | 2.083 | 1.804 | 15.36597  |  1.41x |
| Dataloader: num_workers=4 | 2.943 | 2.649 | 10.87460| base |
| pytorch实现 | TODO | TODO | TODO | TODO | 


## Requirement

docker image:

```
    huangjun12/paddlevideo:tsn_dali_cuda9_0
```

To build container, you can use:

```bash
nvidia-docker run --name tsn-DALI -v /home:/workspace --network=host -it --shm-size 64g -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video huangjun12/paddlevideo:tsn_dali_cuda9_0 /bin/bash
```

## Data

- Kinetics400 dataset please refer to [K400 data](../../dataset/k400.md)

- UCF101 dataset please refer to [UCF101 data](../../dataset/ucf101.md)

## Train

### download pretrain-model

- Please download [ResNet50_pretrain.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_pretrain.pdparams) as pretraind model:

```bash
wget https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_pretrain.pdparams
```

and add path to MODEL.framework.backbone.pretrained in config file as：

```yaml
MODEL:
    framework: "Recognizer2D"
    backbone:
        name: "ResNet"
        pretrained: your weight path
```

### Start training

You can start training by: 

```bash
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3" --log_dir=log_tsn main.py --train_dali -c configs/recognition/tsn/tsn_dali.yaml -o log_level="INFO"
```

- Args -c is used to specify config file，default is ```configs/recognition/tsn/tsn_dali.yaml```。

- For finetune please download our trained model [TSN.pdparams]()<sup>coming soon</sup>，and specify file path with --weights. 

- For the config file usage，please refer to [config](../../tutorials/config.md).

## Test

Please refer to [TSN Test](./tsn.md)

## Inference

Please refer to [TSN Inference](./tsn.md)

## Reference

- [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859), Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, Luc Van Gool

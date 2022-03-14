简体中文 | [English](../../../en/model_zoo/resolution/wafp.md)

# WAFP-Net模型

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型推理](#模型推理)
- [参考论文](#参考论文)

在开始使用之前，您需要按照以下命令安装额外的依赖包：
```bash
python -m pip install scipy
python -m pip install h5py
```

## 模型简介

本模型以百度机器人与自动驾驶实验室的**IEEE Transactions on Multimedia 2021论文 [WAFP-Net: Weighted Attention Fusion based Progressive Residual Learning for Depth Map Super-resolution](TODO)** 为参考，
复现了基于自适应融合注意力的深度图超分辨率模型，其针对真实场景存在的两种图像退化方式（间隔采样和带噪声的双三次采样），提出了一种自适应的融合注意力机制，在保持模型参数量优势的情况下，在多个数据集上取得了SOTA的精度。


## 数据准备

TODO


## 模型训练

### Oxford RobotCar dataset数据集训练

#### 开始训练

- Oxford RobotCar dataset数据集使用单卡训练，训练方式的启动命令如下：

    ```bash
    python3.7 main.py -c configs/resolution/wafp/wafp.yaml --seed 42
    ```


## 模型测试

- 训练好的模型下载地址：[WAFP.pdparams](TODO)

- 测试命令如下：

  ```bash
  python3.7 main.py --test -c configs/resolution/wafp/wafp.yaml -w "output/WAFP/WAFP_epoch_00080.pdparams"
  ```

    在Oxford RobotCar dataset的validation数据集上的测试指标如下：

  | version |  RMSE   |  SSIM   |
  | :------ | :-----: | :-----: |
  | ours    |  2.5762 |  0.9813 |

## 模型推理

### 导出inference模型

```bash
python3.7 tools/export_model.py -c configs/resolution/wafp/wafp.yaml -p data/WAFP.pdparams -o inference/WAFP
```

上述命令将生成预测所需的模型结构文件`WAFP.pdmodel`和模型权重文件`WAFP.pdiparams`以及`WAFP.pdiparams.info`文件，均存放在`inference/WAFP/`目录下

上述bash命令中各个参数含义可参考[模型推理方法](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)

### 使用预测引擎推理

```bash
python3.7 tools/predict.py --input_file data/example.mat \
                           --config configs/resolution/wafp/wafp.yaml \
                           --model_file inference/WAFP/WAFP.pdmodel \
                           --params_file inference/WAFP/WAFP.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

推理结束会默认以伪彩的方式保存下模型输出得到的深度图。

以下是样例图片和对应的预测深度图：

<img src="TODO" width = "512" height = "256" alt="image" align=center />

<img src="TODO" width = "512" height = "256" alt="depth" align=center />


## 参考论文

- [WAFP-Net: Weighted Attention Fusion based Progressive Residual Learning for Depth Map Super-resolution](TODO), Xibin Song, Dingfu Zhou, Wei Li∗, Yuchao Dai, Liu Liu, Hongdong Li, Ruigang Yang and Liangjun Zhang

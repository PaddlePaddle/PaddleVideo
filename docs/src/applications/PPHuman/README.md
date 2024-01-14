# PP-Human 行为识别模型

实时行人分析工具[PP-Human](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/deploy/pphuman)中集成了基于骨骼点的行为识别模块。本文档介绍如何基于[PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo/)，完成行为识别模型的训练流程。

## 行为识别模型训练
目前行为识别模型使用的是[ST-GCN](https://arxiv.org/abs/1801.07455)，并在[PaddleVideo训练流程](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/stgcn.md)的基础上修改适配，完成模型训练。

### 准备训练数据
STGCN是一个基于骨骼点坐标序列进行预测的模型。在PaddleVideo中，训练数据为采用`.npy`格式存储的`Numpy`数据，标签则可以是`.npy`或`.pkl`格式存储的文件。对于序列数据的维度要求为`(N,C,T,V,M)`。

以我们在PPhuman中的模型为例，其中具体说明如下：
| 维度 | 大小 | 说明 |
| ---- | ---- | ---------- |
| N | 不定 | 数据集序列个数 |
| C | 2 | 关键点坐标维度，即(x, y) |
| T | 50 | 动作序列的时序维度（即持续帧数）|
| V | 17 | 每个人物关键点的个数，这里我们使用了`COCO`数据集的定义，具体可见[这里](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/docs/tutorials/PrepareKeypointDataSet_cn.md#COCO%E6%95%B0%E6%8D%AE%E9%9B%86) |
| M | 1 | 人物个数，这里我们每个动作序列只针对单人预测 |

#### 1. 获取序列的骨骼点坐标
对于一个待标注的序列（这里序列指一个动作片段，可以是视频或有顺序的图片集合）。可以通过模型预测或人工标注的方式获取骨骼点（也称为关键点）坐标。
- 模型预测：可以直接选用[PaddleDetection KeyPoint模型系列](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/keypoint) 模型库中的模型，并根据`3、训练与测试 - 部署预测 - 检测+keypoint top-down模型联合部署`中的步骤获取目标序列的17个关键点坐标。
- 人工标注：若对关键点的数量或是定义有其他需求，也可以直接人工标注各个关键点的坐标位置，注意对于被遮挡或较难标注的点，仍需要标注一个大致坐标，否则后续网络学习过程会受到影响。

在完成骨骼点坐标的获取后，建议根据各人物的检测框进行归一化处理，以消除人物位置、尺度的差异给网络带来的收敛难度，这一步可以参考[这里](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/deploy/pphuman/pipe_utils.py#L352-L363)。

#### 2. 统一序列的时序长度
由于实际数据中每个动作的长度不一，首先需要根据您的数据和实际场景预定时序长度（在PP-Human中我们采用50帧为一个动作序列），并对数据做以下处理：
- 实际长度超过预定长度的数据，随机截取一个50帧的片段
- 实际长度不足预定长度的数据：补0，直到满足50帧
- 恰好等于预定长度的数据： 无需处理

注意：在这一步完成后，请严格确认处理后的数据仍然包含了一个完整的行为动作，不会产生预测上的歧义，建议通过可视化数据的方式进行确认。

#### 3. 保存为PaddleVideo可用的文件格式
在经过前两步处理后，我们得到了每个人物动作片段的标注，此时我们已有一个列表`all_kpts`，这个列表中包含多个关键点序列片段，其中每一个片段形状为(T, V, C) （在我们的例子中即(50, 17, 2)), 下面进一步将其转化为PaddleVideo可用的格式。
- 调整维度顺序： 可通过`np.transpose`和`np.expand_dims`将每一个片段的维度转化为(C, T, V, M)的格式。
- 将所有片段组合并保存为一个文件

注意：这里的`class_id`是`int`类型，与其他分类任务类似。例如`0：摔倒， 1：其他`。

至此，我们得到了可用的训练数据（`.npy`）和对应的标注文件（`.pkl`）。

#### 示例：基于UR Fall Detection Dataset的摔倒数据处理
[UR Fall Detection Dataset](http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html)是一个包含了不同摄像机视角及不同传感器下的摔倒检测数据集。数据集本身并不包含关键点坐标标注，在这里我们使用平视视角（camera 0）的RGB图像数据，介绍如何依照上面展示的步骤完成数据准备工作。

（1）使用[PaddleDetection关键点模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/keypoint)完成关键点坐标的检测
```bash
# current path is under root of PaddleDetection

# Step 1: download pretrained inference models.
wget https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip
wget https://bj.bcebos.com/v1/paddledet/models/pipeline/dark_hrnet_w32_256x192.zip
unzip -d output_inference/ mot_ppyoloe_l_36e_pipeline.zip
unzip -d output_inference/ dark_hrnet_w32_256x192.zip

# Step 2: Get the keypoint coordinarys

# if your data is image sequence
python deploy/python/det_keypoint_unite_infer.py --det_model_dir=output_inference/mot_ppyoloe_l_36e_pipeline/ --keypoint_model_dir=output_inference/dark_hrnet_w32_256x192 --image_dir={your image directory path} --device=GPU --save_res=True

# if your data is video
python deploy/python/det_keypoint_unite_infer.py --det_model_dir=output_inference/mot_ppyoloe_l_36e_pipeline/ --keypoint_model_dir=output_inference/dark_hrnet_w32_256x192 --video_file={your video file path} --device=GPU --save_res=True
```
这样我们会得到一个`det_keypoint_unite_image_results.json`的检测结果文件。内容的具体含义请见[这里](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/deploy/python/det_keypoint_unite_infer.py#L108)。

这里我们需要对UR Fall中的每一段数据执行上面介绍的步骤，在每一段执行完成后及时将检测结果文件妥善保存到一个文件夹中。
```bash

mkdir {root of PaddleVideo}/applications/PPHuman/datasets/annotations
mv det_keypoint_unite_image_results.json {root of PaddleVideo}/applications/PPHuman/datasets/annotations/det_keypoint_unite_image_results_{video_id}_{camera_id}.json
```

（2）将关键点坐标转化为训练数据


在完成上述步骤后，我们得到的骨骼点数据形式如下：
```
annotations/
├── det_keypoint_unite_image_results_fall-01-cam0-rgb.json
├── det_keypoint_unite_image_results_fall-02-cam0-rgb.json
├── det_keypoint_unite_image_results_fall-03-cam0-rgb.json
├── det_keypoint_unite_image_results_fall-04-cam0-rgb.json
    ...
├── det_keypoint_unite_image_results_fall-28-cam0-rgb.json
├── det_keypoint_unite_image_results_fall-29-cam0-rgb.json
└── det_keypoint_unite_image_results_fall-30-cam0-rgb.json
```
这里使用我们提供的脚本直接将数据转化为训练数据, 得到数据文件`train_data.npy`, 标签文件`train_label.pkl`。该脚本执行的内容包括解析json文件内容、前述步骤中介绍的整理训练数据及保存数据文件。
```bash
# current path is {root of PaddleVideo}/applications/PPHuman/datasets/

python prepare_dataset.py
```
几点说明：
- UR Fall的动作大多是100帧左右长度对应一个完整动作，个别视频包含一些无关动作，可以手工去除，也可以裁剪作为负样本
- 统一将数据整理为100帧，再抽取为50帧，保证动作完整性
- 上述包含摔倒的动作是正样本，在实际训练中也需要一些其他的动作或正常站立等作为负样本，步骤同上，但注意label的类型取1。

这里我们提供了我们处理好的更全面的[数据](https://bj.bcebos.com/v1/paddledet/data/PPhuman/fall_data.zip)，包括其他场景中的摔倒及非摔倒的动作场景。

### 训练与测试
在PaddleVideo中，使用以下命令即可开始训练：
```bash
# current path is under root of PaddleVideo
python main.py -c applications/PPHuman/configs/stgcn_pphuman.yaml

# 由于整个任务可能过拟合,建议同时开启验证以保存最佳模型
python main.py --validate -c applications/PPHuman/configs/stgcn_pphuman.yaml
```

在训练完成后，采用以下命令进行预测：
```bash
python main.py --test -c applications/PPHuman/configs/stgcn_pphuman.yaml  -w output/STGCN/STGCN_best.pdparams
```

### 导出模型推理

- 在PaddleVideo中，通过以下命令实现模型的导出，得到模型结构文件`STGCN.pdmodel`和模型权重文件`STGCN.pdiparams`，并增加配置文件：
```bash
# current path is under root of PaddleVideo
python tools/export_model.py -c applications/PPHuman/configs/stgcn_pphuman.yaml \
                                -p output/STGCN/STGCN_best.pdparams \
                                -o output_inference/STGCN

cp applications/PPHuman/configs/infer_cfg.yml output_inference/STGCN

# 重命名模型文件，适配PP-Human的调用
cd output_inference/STGCN
mv STGCN.pdiparams model.pdiparams
mv STGCN.pdiparams.info model.pdiparams.info
mv STGCN.pdmodel model.pdmodel
```
完成后的导出模型目录结构如下：
```
STGCN
├── infer_cfg.yml
├── model.pdiparams
├── model.pdiparams.info
├── model.pdmodel
```

至此，就可以使用[PP-Human](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/deploy/pphuman)进行行为识别的推理了。

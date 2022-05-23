

# 乒乓球动作检测模型

## 内容
- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [使用训练模型推理](#使用训练模型推理)
- [模型评估](#模型评估)
- [模型优化](#模型优化)
- [模型部署](#模型部署)
- [参考论文](#参考论文)

## 模型简介
该代码库用于体育动作检测, 基于paddle2.0版本开发，结合PaddleVideo中的BMN和CVPR-HACS冠军方案TCANet进行视频时空二阶段检测算法。

AIStudio项目： [基于飞桨实现乒乓球时序动作定位大赛: B榜第一名方案](https://aistudio.baidu.com/aistudio/projectdetail/3545680?shared=1)

## 数据准备
- 数据集包含了19-21赛季兵乓球国际（世界杯、世锦赛、亚锦赛，奥运会）国内（全运会，乒超联赛）比赛标准单机位高清转播画面特征信息。其中包含912条视频特征文件，每个视频时长在0～6分钟不等，特征维度为2048，以pkl格式保存。我们对特征数据中面朝镜头的运动员的回合内挥拍动作进行了标注，单个动作时常在0～2秒不等，训练数据为729条标注视频，A测数据为91条视频，训练数据标签以json格式给出

- [训练数据集](https://aistudio.baidu.com/aistudio/datasetdetail/122998/0)与[测试数据集](https://aistudio.baidu.com/aistudio/datasetdetail/123004)下载后存储在如下的目录结构中：
```python
| - data
	| - data123004
		| - Features_competition_test_A.tar.gz
	| - data122998
		| - Features_competition_train.tar.gz
		| - label_cls14_train.json 
```
- Features目录中包含912条ppTSM抽取的视频特征，特征保存为pkl格式，文件名对应视频名称，读取pkl之后以(num_of_frames, 2048)向量形式代表单个视频特征，如下示例
```python
{'image_feature': array([[-0.00178786, -0.00247065,  0.00754537, ..., -0.00248864,
        -0.00233971,  0.00536158],
       [-0.00212389, -0.00323782,  0.0198264 , ...,  0.00029546,
        -0.00265382,  0.01696528],
       [-0.00230571, -0.00363361,  0.01017699, ...,  0.00989012,
        -0.00283369,  0.01878656],
       ...,
       [-0.00126995,  0.01113492, -0.00036558, ...,  0.00343453,
        -0.00191288, -0.00117079],
       [-0.00129959,  0.01329842,  0.00051888, ...,  0.01843636,
        -0.00191984, -0.00067066],
       [-0.00134973,  0.02784026, -0.00212213, ...,  0.05027904,
        -0.00198008, -0.00054018]], dtype=float32)}
<class 'numpy.ndarray'>
```
- 训练标签见如下格式：
```javascript
# label_cls14_train.json
{
    'fps': 25,    #视频帧率
    'gts': [
        {
            'url': 'name_of_clip.mp4',      #名称
            'total_frames': 6341,    #总帧数（这里总帧数不代表实际视频帧数，以特征数据集维度信息为准）
            'actions': [
                {
                    "label_ids": [7],    #动作类型编号
                    "label_names": ["name_of_action"],     #动作类型
                    "start_id": 201,  #动作起始时间,单位为秒
                    "end_id": 111    #动作结束时间,单位为秒
                },
                ...
            ]
        },
        ...
    ]
}
```
- 解压数据集
```
cd script
sh unzip_tra_dataset.sh
sh unzip_val_dataset.sh
```
- 数据集预处理
```python
sh tra_preprocess.sh
# 如果需要做k折划分，执行sh split_k_fold.sh，后面需要相应修改yaml文件中的数据路径
sh val_preprocess.sh
```
## 模型训练

```
sh train.sh
```
## 模型推理
```
sh inference.sh
sh val_postprocess.sh
```


## 参考论文
- [BMN: Boundary-Matching Network for Temporal Action Proposal Generation](https://arxiv.org/abs/1907.09702)
- [Temporal Context Aggregation Network for Temporal Action
Proposal Refinement](https://openaccess.thecvf.com/content/CVPR2021/papers/Qing_Temporal_Context_Aggregation_Network_for_Temporal_Action_Proposal_Refinement_CVPR_2021_paper.pdf)


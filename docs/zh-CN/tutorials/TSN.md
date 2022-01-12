# TSN

## 背景
TSN 可以看作是对 two-stream 的改进，通过设计有效的卷积网络体系结构 TSN 解决视频动作分类中的两个主要问题：
* 长距离时序依赖问题（有些动作在视频中持续的时间较长）；
* 解决数据量较少的问题；

## 贡献
TSN 的贡献可概括为以下两点：
* TSN 模型基于 long-range temporal structure 建模，结合了 sparse temporal sampling strategy 和 video-level supervision 从而保证对整段视频学习的有效性和高效性；
* 提出了一系列最佳实践方案；

## 原理
由于 two-stream 网络处理的是单帧图像（空间网络）或者短片段中的一堆帧图像（时序网络），因此 two-stream 网络无法满足时间跨度较长的视频动作。为了能够处理长范围时序结构的情况，可以使用密集帧采样方式从视频中获取长时间信息，但这样会增加时间成本同时采样到的连续帧之间存在冗余。于是在 TSN 模型中作者使用稀疏采用的方式来替代密集采样，降低计算量的同时一定程度上也去除了冗余信息。

TSN 采用和 two-stream 相似的结构，网络由空间流卷积网络和时间流卷积组成。TSN 使用稀疏采样的方式从整段视频采出一系列的短片段，其中每个片段都会有一个对自身动作类别的初步预测，之后通过对这些片段的预测结果进行“融合”得出对整个视频的预测结果。

## 网络结构
如下图所示，一个视频被分为 ![formula](https://render.githubusercontent.com/render/math?math=K) 段（ segment ）；之后对每个段使用稀疏采样的方式采出一个片段（ snippet ）；然后使用“段共识函数”对不同片段的预测结果进行融合生成“段共识”，此时完成了一个视频级的预测；最后对所有模式的预测结果进行融合生成最终的预测结果。


<p align="center">
<img src="../../images/tsn_structure.jpg" height=200 width=500 hspace='10'/> <br />
</p>

> 这里注意 segment 和 snippet 的区别

TSN 采用与 two-stream 类似的结构，使用空间网络操作一帧 RGB 图像，时序卷积网络操作连续的光流图像。但由于更深的网络结构能够提升对物体的识别能力，因此 TSN 中作者采用 BN-Inception 构建网络。

## 损失函数

给定一段视频 ![formula](https://render.githubusercontent.com/render/math?math=V)，按相等间隔分为 ![formula](https://render.githubusercontent.com/render/math?math=K) 段 ![formula](https://render.githubusercontent.com/render/math?math={S_1,S_2,...,S_K})。 TSN 对一系列片段的建模如下：

<a href="https://www.codecogs.com/eqnedit.php?latex=TSN(T_1,T_2,...,T_K)=H(G(F(T_1;W),F(T_2;W),...,F(T_K;W)))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?TSN(T_1,T_2,...,T_K)=H(G(F(T_1;W),F(T_2;W),...,F(T_K;W)))" title="TSN(T_1,T_2,...,T_K)=H(G(F(T_1;W),F(T_2;W),...,F(T_K;W)))" /></a>

其中，<a href="https://www.codecogs.com/eqnedit.php?latex=(T_1,T_2,...,T_K)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(T_1,T_2,...,T_K)" title="(T_1,T_2,...,T_K)" /></a> 表示片段序列，从每个段 ![formula](https://render.githubusercontent.com/render/math?math=S_k) 中随机采样获取对应的片段 ![formula](https://render.githubusercontent.com/render/math?math=T_k)；<a href="https://www.codecogs.com/eqnedit.php?latex=F(T_k;W)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F(T_k;W)" title="F(T_k;W)" /></a> 表示作用于短片段 ![formula](https://render.githubusercontent.com/render/math?math=T_k) 的卷积网络，![formula](https://render.githubusercontent.com/render/math?math=W) 为网络的参数，返回值为 ![formula](https://render.githubusercontent.com/render/math?math=T_k) 相对于所有类别的得分；段共识函数 ![formula](https://render.githubusercontent.com/render/math?math=G) 用于融合所有片段的预测结果。预测函数 ![formula](https://render.githubusercontent.com/render/math?math=H)用于预测整段视频属于每个动作类别的概率，它的输入为段共识函数 ![formula](https://render.githubusercontent.com/render/math?math=G) 的结果。

最后，采用标准分类交叉熵计算部分共识的损失：

<a href="https://www.codecogs.com/eqnedit.php?latex=L\left(&space;y,G&space;\right)&space;=-\sum_{i=1}^C{y_i\left(&space;G_i-\log&space;\sum_{j=1}^C{\exp\text{\&space;}G_j}&space;\right)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L\left(&space;y,G&space;\right)&space;=-\sum_{i=1}^C{y_i\left(&space;G_i-\log&space;\sum_{j=1}^C{\exp\text{\&space;}G_j}&space;\right)}" title="L\left( y,G \right) =-\sum_{i=1}^C{y_i\left( G_i-\log \sum_{j=1}^C{\exp\text{\ }G_j} \right)}" /></a>


其中，![formula](https://render.githubusercontent.com/render/math?math=C) 是类别总数；![formula](https://render.githubusercontent.com/render/math?math=y_i) 是类别 ![formula](https://render.githubusercontent.com/render/math?math=i) 的 ![formula](https://render.githubusercontent.com/render/math?math=groundtruth)；论文中段的数量 ![formula](https://render.githubusercontent.com/render/math?math=K) 设置为 ![formula](https://render.githubusercontent.com/render/math?math=3)；共识函数 ![formula](https://render.githubusercontent.com/render/math?math=G) 采用取均值的方式，从所有片段的相同类别中推断出某个类别得分 ![formula](https://render.githubusercontent.com/render/math?math=G_i)。

## 模型输入
对于图像任务而言，只能够使用图像本身提取特征。但对视频来说，除了每一帧图像外，还有视频中的光流信息。为了探索更多输入形式对模型效果影响，TSN 模型在空间卷积网络中除了使用单一 RGB 图像外，还使用了 RGB difference；在时序卷积网络中除了将连续的光流场作为输入外还采用了扭曲的光流场。

<p align="center">
<img src="../../images/tsn_input.jpg" height=200 width=500 hspace='10'/> <br />
</p>

单一 RGB 图像只能表示静态信息，缺少上下文信息。但连续两帧之间的差异能够表示动作的改变，因此作者尝试将 RGB difference 作为模型的一种输入。

TSN 将光流场作为输入捕获运动信息；将扭曲光流场作为输入抑制背景运动，使得专注于视频中的人物运动。

## 训练
由于数据集较小，为了避免过拟合，作者提出了一系列的训练策略。

### 数据增强
通过数据增强可生成额外的训练样本，一定程度上能够避免模型的过拟合。two-stream 中采用的数据增强方式有随机裁剪和水平翻转，在 TSN 中作者新增了两种数据增强方法：
* 角裁剪：仅从图片的边角或中心提取区域，避免默认关注图片的中心；
* 尺度抖动：将输入图像或者光流场的大小固定为 <a href="https://www.codecogs.com/eqnedit.php?latex=256&space;\times&space;340" target="_blank"><img src="https://latex.codecogs.com/gif.latex?256&space;\times&space;340" title="256 \times 340" /></a>，裁剪区域的宽和高随机从 ![formula](https://render.githubusercontent.com/render/math?math={256,224,192,168}) 中选择。最终，裁剪区域将被 <a href="https://www.codecogs.com/eqnedit.php?latex=224&space;\times&space;224" target="_blank"><img src="https://latex.codecogs.com/gif.latex?224&space;\times&space;224" title="224 \times 224" /></a> 用于网络训练。

### 交叉预训练
由于空间网络以 RGB 图片作为输入，因此作者在空间网络上直接使用 ImageNet 预训练模型初始化网络的参数。对于以 RGB difference 和光流作为输入的模型，作者提出了交叉预训练技术，使用 RGB 预训练模型初始化时序网络。首先，通过线性变换将光流场离散到从 0 到 255 的区间，使得光流场和 RGB 的取值范围相同；之后修改 RGB 模型的第一个卷积层，对 RGB 通道上的权重进行取均值操作；然后依据时序网络的输入通道数复制 RGB 均值。该策略能够有效的避免时序网络出现过拟合现象。

### 正则化技术
由于光流分布和 RGB 分布不同，因此除了第一个 BN 层，其余 BN 层的参数都被固定。此外，为了进一步降低过拟合产生的影响，作者在 BN-Inception 的全局 pooling 层后添加一个额外的 dropout 层，其中空间卷积网络的 dropout 比例设置为 0.8；时序卷积网络的 dropout 比例设置为 0.7。

## 数据集
模型在 HMDB51 和 UCF101 两个主流的动作识别数据集上进行。其中，HMDB51 数据集包含 51 个动作分类的 6766 个视频剪辑；UCF101 数据集包含 13320 个视频剪辑，共 101 类动作。

## 实现细节
* 基于动量的小批量随机梯度下降算法，momentum 设置为 0.9；
* batch size 为 256；
* 使用 ImageNet 预训练模型对网络权重进行初始化；
* learning rate 调整，对于空间网络，初始化为 0.01，并且每 2000 次迭代后降变为原来的 0.1 倍，训练过程共迭代 4500 次；对于时序网络，初始化为 0.005，并且在第 12000 和 18000 次迭代之后降为原来的 0.1 倍，训练过程共迭代 20000 次；
* 使用 TVL1 光流算法来提取正常光流场和扭曲光流场。
* 8 块 TITANX GPUs

## PaddleVideo
为了加快 TSN 模型的推理速度，PaddleVideo 去掉了与 RGB difference、光流以及扭曲光流相关的部分。

PaddleVideo 中实现稀疏采样的关键代码：
```python
frames_len = results['frames_len']   # 视频中总的帧数
average_dur = int(int(frames_len) / self.num_seg)   # 每段中视频的数量
frames_idx = []   # 存放采样到的索引
for i in range(self.num_seg):
    idx = 0  # 采样的起始位置
    if not self.valid_mode:
        # 如果训练
        if average_dur >= self.seg_len:
            idx = random.randint(0, average_dur - self.seg_len)
            idx += i * average_dur
        elif average_dur >= 1:
            idx += i * average_dur
        else:
            idx = i
    else:
        # 如果测试
        if average_dur >= self.seg_len:
            idx = (average_dur - 1) // 2
            idx += i * average_dur
        elif average_dur >= 1:
            idx += i * average_dur
        else:
            idx = i
    # 从采样位置采连续的帧
    for jj in range(idx, idx + self.seg_len):
        if results['format'] == 'video':
            frames_idx.append(int(jj % frames_len))
        elif results['format'] == 'frame':
            frames_idx.append(jj + 1)
        else:
            raise NotImplementedError
```

PaddleVideo 中实现“段共识”的核心代码：
```
# [N * num_segs, in_channels, 7, 7]
x = self.avgpool2d(x)
# [N * num_segs, in_channels, 1, 1]
if self.dropout is not None:
    x = self.dropout(x)
# [N * num_seg, in_channels, 1, 1]
x = paddle.reshape(x, [-1, num_seg, x.shape[1]])
# [N, num_seg, in_channels]
x = paddle.mean(x, axis=1)
# [N, 1, in_channels]
x = paddle.reshape(x, shape=[-1, self.in_channels])
# [N, in_channels]
score = self.fc(x)
```

## 广告时间
如果文档对您理解 TSN 模型有帮助，欢迎👍star🌟，👏fork，您的支持是我们前进的动力⛽️。

## 参考
[Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859)

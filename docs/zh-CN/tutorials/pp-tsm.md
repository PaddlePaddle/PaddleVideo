# PP-TSM高效实用视频识别模型  

PP-TSM是PaddleVideo基于TSM优化和改进的视频模型，  
其精度(UCF101和Kinetics400数据集top1)和推理速度均优于TSM论文及其他开源的TSM模型5%，3%以上，  
要求使用PaddlePaddle2.0(可使用pip安装) 或适当的develop版本。  

在仅用ImageNet pretrain情况下，PP-TSM在UCF101和Kinetics400数据集top1分别达到89.5%和73.5%，  
在单卡V100上FP32推理速度为147 VPS （基于Kinectics400数据集）.
在单卡V100上开启TensorRT下FP16推理速度为TODO。

pp-TSM在Kinetics400上top1精度为73.5%，是至今为止开源的2D视频模型中在相同条件下的最高性能。  

PP-TSM从如下方面优化和提升TSM模型的精度和速度：  
1、基于知识蒸馏的预训练模型  ， +1.3%  
2、网络结构微调  ，+2.5%  
3、更优的batch size ，+0.2%   
4、更优的L2正则化  ，+0.3%  
5、label_smoothing  ，+0.2%  
6、更优的lr decay  ，+0.15%  
7、数据增广  ，+0.3%  
8、更优的epoch num  ，+0.15%  
9、bn策略  ，+0.4%  
10、集成PaddleInference进行预测推理  
11、知识蒸馏、优化器等更多TODO策略    
其中，每项策略的精度提升指标参考上述数据（基于ucf101及k400上进行实验）。

## preciseBN

在介绍preciseBN之前，我们先回顾一下BN(Batch Norm)。BN层是一种正则化层，在训练时，它根据当前batch的数据按通道计算的均值和方差，然后进行归一化运算，公式如图:

详细介绍可参考[BatchNorm文档](https://paddlepaddle.org.cn/documentation/docs/zh/2.0-rc1/api/paddle/fluid/dygraph/BatchNorm_cn.html#batchnorm)。

假设训练数据的分布和测试数据的分布是一致的，在训练时我们会计算并保存滑动均值和滑动方差，供测试时使用。滑动均值和滑动方差的计算方式如下:

简单的说，moving_mean等于当前batch计算的均值与历史保存的moving_mean的加权和，即为滑动均值。**但滑动均值并不等于真实的均值**，因此测试时的精度仍会受到一定影响。
为了提升测试精度，我们需要重新计算一个更加精确的均值，这就是preciseBN的目的。

真实的均值如何计算？最直观的想法是，把所有训练数据组成一个batch，输入网络进行前向传播，每经过一个BN层，计算一下当前特征的均值和方差。
由于训练样本过多，实际操作中不可能这么做。
所以近似做法是，网络训练完成后，固定住网络中的参数不动，将所有训练数据分成N个batch，依次输入网络进行前向计算，在这个过程中保存下来每个iter的均值和方差，最终得到所有训练样本精确的均值和方差。
这就是preciseBN的计算方法。具体实现参考[preciseBN](https://github.com/PaddlePaddle/PaddleVideo/blob/main/paddlevideo/utils/precise_bn.py)。

实际使用时，由于迭代所有训练样本比较耗费时间，一般只会跑200个iter左右。



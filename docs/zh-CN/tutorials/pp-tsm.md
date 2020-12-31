PP-TSM高效实用视频识别模型  
PP-TSM是PaddleVideo基于TSM优化和改进的视频模型，  
其精度(UCF101和Kinetics400数据集top1)和推理速度均优于TSM论文及其他开源的TSM模型5%，3%以上，  
要求使用PaddlePaddle2.0(可使用pip安装) 或适当的develop版本。  

在仅用ImageNet pretrain情况下，PP-TSM在UCF101和Kinetics400数据集top1分别达到89.5%和73.5%，  
在单卡V100上FP32推理速度为DOING, V100上开启TensorRT下FP16推理速度为TODO。

据我们所知，在相同条件下，在Kinetics400上top1精度为73.5%，是至今为止开源的2D视频模型中的最高性能。  

PP-TSM从如下方面优化和提升TSM模型的精度和速度：
1、基于知识蒸馏的预训练模型
2、网络结构微调
3、更优的batch size
4、更优的L2正则化
5、label_smoothing
6、更优的lr decay
7、数据增广
8、更优的epoch num
9、bn策略
10、集成PaddleInference进行预测推理
11、知识蒸馏、优化器等更多TODO策略

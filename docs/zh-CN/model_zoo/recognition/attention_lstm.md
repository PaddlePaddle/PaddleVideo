循环神经网络（RNN）常用于序列数据的处理，可建模视频连续多帧的时序信息，在视频分类领域为基础常用方法。  
该模型采用了双向长短时记忆网络（LSTM），将视频的所有帧特征依次编码。  
与传统方法直接采用LSTM最后一个时刻的输出不同，该模型增加了一个Attention层，  
每个时刻的隐状态输出都有一个自适应权重，然后线性加权得到最终特征向量。  
参考论文中实现的是两层LSTM结构，而本代码实现的是带Attention的双向LSTM，  
Attention层可参考论文AttentionCluster( https://arxiv.org/abs/1711.09550 )。  

AttentionLSTM模型使用2nd-Youtube-8M数据集，请参考数据准备页面（ https://github.com/PaddlePaddle/PaddleVideo/blob/main/docs/zh-CN/dataset/youtube8m.md ）。  
模型训练和评估，请参考教程页面(TODO)  
在Youtube-8M验证集上，AttentionLSTM模型的Hit@1为0.89, PERR 为0.8012, GAP为0.8594.

这三个指标是youtube8M数据集官方评估使用的  
Hit@k indicates the fraction of test samples that contain at least one of the ground truth labels in the top k predictions.  
PERR measures the video-level annotation precision when we retrieve the same number of entities per video as there are in the ground-truth.  
GAP is the global average precision.  
论文参考 https://arxiv.org/abs/1609.08675  
具体实现参考 https://github.com/google/youtube-8m/blob/master/eval_util.py

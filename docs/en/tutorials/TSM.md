# 1. Background&Motivation
At present, the video data on the Internet is increasing rapidly, and the time users spend watching short videos and small videos is also increasing rapidly. How to analyze, process and classify the massive video resources quickly and accurately is an urgent problem to be solved. The video understanding technology can analyze the video content in multiple dimensions, understand the video semantics, and automatically classify and label the video, which greatly saves the efficiency of manual audit and costs. At the same time, accurate user recommendation is realized to improve the experience effect.
In this paper, we will introduce the classic model **TSM (Temporal Shift Module)** in the field of video understanding, which is proposed by **MIT** and **IBM Watson AI Lab** `Ji Lin, Chuang Gan and Songhan, etc`, to achieve the balance between effect and performance through Temporal Shift Module and improve video understanding ability.

The most relevant video understanding model to TSM is the **Temporal Segment Network (TSN)** published by Limin Wang et al in ECCV, 2016. The TSN model samples N frames from a video and performs temporal information fusion by averaging the classification results of N frames in the simplest and direct way, which achieves the performance of state-of-the -art at that time and is applied in a large scale. Considering that TSN model is not sufficient in modeling temporal information, 
a series of works represented by I3D, S3D and P3D carry out end-to-end joint spatial-temporal modeling through 3D convolution. Although this series of works can capture spatial-temporal features, compared with TSN, the transition from 2D convolution to 3D convolution inevitably introduces extra computation. TSM cleverly uses the idea of temporal dimension feature map shift, theoretically achieving the purpose of feature fusion and joint modeling among different frames with zero extra computing overhead.

Paper Address: [Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383v2.pdf)

Take a look at the following example: if the image is played from left to right and from right to left respectively, the subjects will give different but correct interpretation of the video, indicating that the understanding of the video is strongly dependent on the timing of the video. This is the motivation proposed by TSM, which is to capture the timing information of the video.
<p align="center">
<img src="../../images/temporal.png" height=188 width=500 hspace='10'/> <br />
</p>

As interesting as this looks, let's dive into the core modules of TSM.

# 2. Analysis of key technologies used in TSM

On the basis of traditional picture analysis, video analysis needs researchers to supplement the modeling structure of temporal information. At present, 2D CNN and 3D CNN are the two most commonly used methods in video understanding: using 2D CNN model requires less computation but will lose part of the time information; While using 3D CNN has a good effect but a great amount of computation. Faced with such a situation, Ji Lin, Chuang Gan and Song Han et al. from MIT and IBM Watson AI Lab proposed the Temp Shift Module (TSM) Module. By embedding the time displacement module into 2D CNN, they can easily achieve the same video understanding ability as 3D CNN without adding any additional calculation amount and parameters.
<p align="center">
<img src="../../images/tsm_intr.png" height=188 width=500 hspace='10'/> <br />
</p>

he rows and columns of the matrix in the figure above represent the temporal and channel dimensions of the feature graph, respectively. In TSM module, some channels are moved forward one step in the temporal dimension, and some channels are moved backward one step in the temporal dimension, and the gaps after the displacement are zeroed. In this way, context interaction on the temporal dimension is introduced into the feature graph. The channel movement operation can make the current frame contain the channel information of the two frames before and after the current frame. In this way, the 2D convolution operation can directly extract the spatial-temporal information of the video just like the 3D convolution.
It improves the modeling ability of the model in time dimension. On this basis, the researchers further subdivided the module into TSM module suitable for online video and TSM module suitable for offline video.
<p align="center">
<img src="../../images/tsm_architecture.png" height=188 width=500 hspace='10'/> <br />
</p>

Bi-Direction TSM module can obtain past and future spatial and temporal information, which is suitable for offline video with high throughput. However, UNI-Direction TSM module is only suitable for low delay online video recognition compared with the present and past spatio-temporal information.
In addition, the author also considered the insertion position of TSM modules and compared two TSM insertion methods: **Residual TSM** and **in-place TSM**. The author found that **Residual TSM** had better effect than **in-place TSM**, and explained that **in-place TSM** would affect the extraction of spatial information In the model.
<p align="center">
<img src="../../images/residual_tsm.png" height=188 width=500 hspace='10'/> <br />
</p>

TSM module is  **So Easy!!**, the next question is how to implement the code?

# 3. The core codes of TSM

Now that the principle is clear, let's look at how the code works. First let's look at how the Torch version works. Unfortunately, the Torch framework does not provide an API for TSM, so we will have to do it ourselves. The code is shown below:
<p align="center">
<img src="../../images/torch_tsm.png" height=160 width=500 hspace='10'/> <br />
</p>

This means that you only need to add 4 lines of code to TSN's code base to ** * double the accuracy of Something-Something data sets!! Is ** a simple and efficient model? Have to bow to the big boss!

But...，

Fly paddle frame fully take into account the needs of the majority of users have been for you children's shoes to achieve TSM OP

<p align="center">
<img src="../../images/tsm_op.png" height=300 width=400 hspace='10'/> <br />
</p>

So you children's shoes no longer have to achieve their own, **direct call can be!! , it can be called directly!! , it can be called directly!!** The important thing must say three times.

Did you think that was the end of the matter? Alas! **Too young Too simple !!!**

We have also optimized it to increase speed by 5 times while reducing memory consumption. See the acceleration documentation at (./ acceler.md).

Let's take a look at how TSM is implemented using flying OARS:

`import paddle.nn.functional as F`


`shifts = F.temporal_shift(inputs, self.num_seg,1.0 / self.num_seg)`

TSM can be implemented in two lines of code, isn't it simple?

# Reference
[1] [Lin Ji , Gan Chuang , Han Song . TSM: Temporal Shift Module for Efficient Video Understanding. arXiv:1811.08383,2018](https://arxiv.org/pdf/1811.08383v2.pdf).


[2] [Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoo Tang,and Luc Van Gool. Temporal segment networks for action recognition in videos? In Proceedings of the European Conference on Computer Vision,pages 20–36. Springer, 2016](https://arxiv.org/abs/1608.00859).

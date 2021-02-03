# 1. Background&Motivation
At present, the video data on the Internet is increasing rapidly, and the time users spend watching short videos and small videos is also increasing rapidly. How to analyze, process and classify the massive video resources quickly and accurately is an urgent problem to be solved. The video understanding technology can analyze the video content in multiple dimensions, understand the video semantics, and automatically classify and label the video, which greatly saves the efficiency of manual audit and costs. At the same time, accurate user recommendation is realized to improve the experience effect.
In this paper, we will introduce the classic model **TSM (Temporal Shift Module)** in the field of video understanding, which is proposed by **MIT** and **IBM Watson AI Lab** `Ji Lin, Chuang Gan and Songhan, etc`, to achieve the balance between effeiciency and performance and improve video understanding ability.

The most relevant video understanding model to TSM is the **Temporal Segment Network (TSN)** published by Limin Wang
a series of works represented such as I3D, S3D and P3D, which carry out end-to-end joint spatial-temporal modeling through 3D convolution. Although this series of works can capture spatial-temporal features, compared with TSN, the transition from 2D convolution to 3D convolution inevitably introduces extra computation. TSM cleverly uses the idea of temporal dimension feature map shift, theoretically achieving the purpose of feature fusion and joint modeling among different frames with zero extra computing overhead compared with TSN.

**Paper Address:** [Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383v2.pdf)

Let's have a look at the following example: if the video is played from left to right and then from right to left respectively, the subjects will give different but correct interpretation of the video, indicating that the understanding of the video is strongly dependent on the temporal information of the video. Yes !, It is the motivation why TSM is proposed.
<p align="center">
<img src="../../images/temporal.png" height=188 width=500 hspace='10'/> <br />
</p>

It looks interesting, next,let's dive into the core modules of TSM.

# 2. Dark technologies used in TSM

On the basis of traditional image analysis, video analysis needs researchers to supplement the modeling structure of temporal information. At present, 2D CNN and 3D CNN are the two most commonly used methods in video understanding: using 2D CNN model requires less computation but will lose part of the time information; While using 3D CNN has a good effect but a large amount of computation. Faced with such a situation, Ji Lin, Chuang Gan and Song Han et al. from MIT and IBM Watson AI Lab proposed the Temp Shift Module (TSM) Module. By embedding the time displacement module into 2D CNN, they can easily achieve the same video understanding ability as 3D CNN without adding any additional calculation and parameters.
<p align="center">
<img src="../../images/tsm_intr.png" height=188 width=500 hspace='10'/> <br />
</p>

The rows and columns of the matrix in the figure above represent the temporal and channel dimensions of the feature graph, respectively. In TSM module, some channels are moved forward one step int the temporal dimension, and some channels are moved backward one step in the temporal dimension, and the gaps after the displacement are zeroed. In this way, context interaction on the temporal dimension is introduced into the feature graph. The channel movement operation can make the current frame contain the channel information of the two adjacent frames. In this way, the 2D convolution operation can directly extract the spatial-temporal information of the video just like the 3D convolution.
It improves the modeling ability of the model in time dimension. based on this basis, the researchers further subdivided the module into TSM module suitable for online video and TSM module suitable for offline video.
<p align="center">
<img src="../../images/tsm_architecture.png" height=188 width=500 hspace='10'/> <br />
</p>

Bi-Direction TSM module can obtain past and future spatial and temporal information, which is suitable for offline video with high throughput. However, UNI-Direction TSM module is only suitable for low delay online video recognition compared with the present and past spatio-temporal information.
In addition, the author also considered the insertion position of TSM modules and compared two TSM insertion methods: **Residual TSM** and **in-place TSM**. The author found that **Residual TSM** could achieve better performance than **in-place TSM**, At the same time, author explained that **in-place TSM** may affect the extraction of spatial information.
<p align="center">
<img src="../../images/residual_tsm.png" height=188 width=500 hspace='10'/> <br />
</p>

TSM module looks **So Easy!!**, the next question is how to implement ?

# 3. The core codes of TSM

Now that the principle is clear, let's look at how the code works. First let's have a look the torch version tsm. Unfortunately, the Torch framework does not provide an API for TSM, so we will have to do it by ourselves. The code is shown below:
<p align="center">
<img src="../../images/torch_tsm.png" height=160 width=500 hspace='10'/> <br />
</p>

This means that you only need to add four lines of code to TSN's codebase then you can **double the accuracy in Something-Something datasets!!** what a simple and efficient model!

But...，

**paddlepaddle** framework take the needs of the majority of users into account and have achieve TSM OP,then users can use it easily.
<p align="center">
<img src="../../images/tsm_op.png" height=300 width=400 hspace='10'/> <br />
</p>

So you no longer have to achieve it by yourself, **it cab be called directly!!! , it can be called directly!!! , it can be called directly!!!** The important thing must say three times.

Do you think that it is the end of the this topic?  **Too young Too simple !!!**

We have also optimized it to increase speed by 5 times while reducing memory consumption. See the acceleration documentation [accelerate.md](./accelerate.md) for more information.

Let's have a look at how TSM is implemented using **paddlepaddle**:

`import paddle.nn.functional as F`


`shifts = F.temporal_shift(inputs, self.num_seg,1.0 / self.num_seg)`

**Only two lines codes !!!**, isn't it easy ?

# Reference
[1] [Lin Ji , Gan Chuang , Han Song . TSM: Temporal Shift Module for Efficient Video Understanding. arXiv:1811.08383,2018](https://arxiv.org/pdf/1811.08383v2.pdf).


[2] [Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoo Tang,and Luc Van Gool. Temporal segment networks for action recognition in videos? In Proceedings of the European Conference on Computer Vision,pages 20–36. Springer, 2016](https://arxiv.org/abs/1608.00859).

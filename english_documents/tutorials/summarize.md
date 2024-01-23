[简体中文](../../zh-CN/tutorials/summarize.md) | English

# Introduction for video classification(action recognition)

## Wide range of application scenarios
Video classification has a wide range of applications in many fields, such as online video platforms such as short videos, offline such as security, transportation, quality inspection and other fields。


## Multiple subtasks
Similar to image tasks, video tasks can also be divided into two categories: **classification (recognition) and detection**, and these two types of tasks can be specifically subdivided by combining different scenes：

+ Task1：Trimmed Action Recognition. Users input a trimmed video,which contains only single action,then a video tag will be output by model as depicted in fig below:
<p align="center">
<img src="../../images/action_classification.png" height=300 width=700 hspace='10'/> <br />
 Action Classification
</p>

  In terms of the data modality used, classification tasks can be further subdivided into classification based on single modality data, classification based on multi-modality data, classification based on RGB images and classification based on human skeleton, etc, as shown in the figure below:

  <p align="center">
  <img src="../../images/multimodality.png" height=300 width=500 hspace='10'/> <br />
 multi-modality
  </p>
In terms of the perspective of video, it can also be divided into first-person action recognition, 
third-person action recognition, single perspective action recognition and multi-perspective fusion action recognition. 
Users who are interested in these fields can refer to relevant literatures.

+ Task2：Untrimmed Video Classification. 
Unlike trimmed videos, untrimmed videos often contain multiple actions and have a long time span. 
There are a lot of movements that we may need not paying attention to. Through the global analysis of the input long video, and then make a soft classify to mutiple categories.

+ Task3：Temporal Action Proposal. It is similar to the ROI extraction in the image detection task. 
The task is to find the video clips that may contain action in a long video with a lot of actions.

+ Task4：Temporal Action Localization. Compared with the temporal action proposal task as mentioned above, 
temporal action localization task is more consistent with detection task in the field of imgae, 
it requires not only to find the video segments with possible actions from the video but also to classify them,
as shown in the figure below
 <p align="center">
<img src="../../images/action_detection.png" height=200 width=1000 hspace='10'/> <br />
 Action Detection
</p>

+ Task5：Dense-Captioning Events. The reason why it is called dense captioning events is mainly 
because that this task requires video action description on the basis of temporal action localization 
(detection). That is to say, the task needs to locate the actions in a **untrimmed** video,in **temporal 
dimension** and describe the behavior of the **whole video** after obtaining many video segments which contain actions.

## Introduction of datasets

### Classification datasets

The training and validation of the model cannot be done without comprehensive, 
large and well annotated datasets. With the deepening of research on video action recognition, 
more and more datasets are applied to the research in this field. 
Typical datasets are as follows:

+ KTH[<sup>1</sup>](#1)

KTH dataset is an early small action recognition dataset, 
including 599 videos of 6 types of actions (walking, jumping, running, punching, waving and clapping). 
The background is relatively still, except for the zoom in and out of the camera, 
the camera movement is relatively slight. Since this data set is relatively small, 
it is easy to overfit when training heavy 3D networks, 
so most current researches are not based on this it.

+ UCF10[<sup>2</sup>](#2)

UCF101 is a medium-size dataset in which most videos are from YouTube. 
It contains 13,320 videos with 101 types of actions. 
Each type of action is performed by 25 people, each of whom performs 4-7 sets of actions. 
The UCF101 and HMDB51 datasets used to be the benchmarks to evaluate the effectiveness of action 
recognition model for a long time before the Kinetics dataset was released.

+ HMDB51[<sup>3</sup>](#3)

Brown University's proposed dataset named HMDB51 was released in 2011. 
Most of the videos come from movies, 
but some come from public databases and online video libraries such as YouTube. 
The datasets contains 6849 samples divided into 51 classes, 
each of which contains at least 101 samples.

+ Kinetics[<sup>4</sup>](#4)

Kinetics is the most important large-scale action recognition dataset, which was proposed by Google's DeepMind team in 2017. The video data also comes from YouTube, with 400 categories (now expanded to 700 categories) and more than 300,000 videos (now expanded to 600,000 videos), each lasting about 10 seconds. 
The action categories are mainly divided into three categories: "human", "human and animal", "human and human interaction". Kinetics can train 3D-RESNET up to 152 layers without over-fitting, 
which solves the problem that the previous training dataset is too small to train deep 3D network. 
Kinetics has replaced UCF101 and HMDB51 as the benchmark in the field of action recognition. 
At present, most studies use this dataset for evaluation and pre-training.

+ Something-Something[<sup>5</sup>](#5)

SomethingV1 contains 108,499 annotated videos (V2 has expanded to 220,847), each of which last two to six seconds. These videos contain 174 kinds of actions. Different from the previous dataset, 
the identification of this data set requires stronger time information, 
so this dataset has a very important reference value in testing the temporal modeling ability of the model.

In addition to the above datasets, there are Charades[<sup>6</sup>](#6) dataset for complex Action recognition, Breakfast Action[<sup>7</sup>](#7), and Sports 1M[<sup>8</sup>](#8).


### Detection datasets

+ THUMOS 2014

This dataset is from THUMOS Challenge 2014, Its training set is UCF101, validation set and test set include 1010 and 1574 undivided video clips respectively. In the action detection task, only 20 kinds of unsegmented videos of actions were labeled with sequential action fragments, 
including 200 validation sets (3007 action fragments) and 213 test sets (3358 action fragments).

+ MEXaction2

The Mexaction2 dataset contains two types of action: horse riding and bullfighting. 
The dataset consists of three parts: YouTube videos, horseback riding videos in UCF101, and INA videos. 
YouTube clips and horseback riding videos in UCF101 are short segmented video clips that are used as training sets. 
The INA video is a long unsegmented video with a total length of 77 hours, 
and it is divided into three parts: training, validation and test. 
There are 1336 action segments in the training set, 310 in the validation set and 329 in the test set. 
Moreover, the Mexaction2 dataset is characterized by very long unsegmented video lengths, 
and marked action segments only account for a very low proportion of the total video length.

+ ActivityNet

At present the largest database, also contains two tasks of classification and detection. 
This dataset only provides a YouTube link to the video, not a direct download of the video, 
so you also need to use the YouTube download tool in Python to automatically download the videos. 
The dataset contains 200 action categories, 20,000 (training + verification + test set) videos, 
and a total of about 700 hours of video.

## Introduction of classic models
As shown in the figure, 
the action recognition framework mainly includes three steps: 
feature extraction, motion representation and classification. 
How to extract spatiotemporal features of video is the core problem of action recognition and video classification.
 <p align="center">
<img src="../../images/action_framework.png" height=300 width=700 hspace='10'/> <br />
Framework of action recognition
</p>
According to different methods, action recognition (video classification) methods can be generally summarized into two stages: 
manual feature-based method and deep learning-based method. 
Typical motion descriptors in the manual feature-based method stage include DTP and IDT, 
which are also the most excellent motion descriptors accepted by most researchers before deep-learning is applied in this field. 
Interinterested readers may refer to the relevant references at the end of this paper. 
Since 2014, deep learning methods have been gradually applied to the field of video classification. 
At present, deep learning-based methods have become a hotspot of research in both academic and the practice, and the  effect is far beyond the motion features of manual design. 
Since 2014, many classic network structures have been put forward by the researchers regarding the problem of how to represent motion characteristics, 
as shown in the figure below:
 <p align="center">
<img src="../../images/classic_model.png" height=300 width=700 hspace='10'/> <br />
Classic Models
</p>

At present,Paddlevideo has contained several classic models such as:TSN[<sup>9</sup>](#9),TSM[<sup>10</sup>](#10),slowfast[<sup>11</sup>](#11),et al.In the future,
we will analyze the classic models and papers in these fields. Please look forward to it


## Introduction of competetion
+ [ActivityNet](http://activity-net.org/challenges/2020/challenge.html)

ActivityNet is a large-scale action recognition competition. Since 2016, 
it has been held simultaneously with CVPR every year. Up to this year, 
it has been held for 4 consecutive sessions. It focuses on identifying everyday, high-level, goal-oriented activities from 
user-generated videos taken from the Internet video portal YouTube. 
At present, ActivityNet competition has become the most influential competition in the field of action recognition.

## Reference

<div id='1'>
[1] Schuldt C, Laptev I, Caputo B.Recognizing Human Actions: A Local SVM Approach Proceedings of International Conference on Pattern Recognition. Piscataway, NJ: IEEE, 2004:23-26
</div>
<br/>
<div id='2'>
[2] Soomro K, Zamir A R, Shah M. UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild. arXiv:1212.0402,2012.
</div>
<br/>
<div id='3'>
[3] Kuehne H, Jhuang H, Garrote E, et al. HMDB: a large video database for human motion recognition Proceedings of IEEE International Conference on Computer Vision. Piscataway, NJ: IEEE, 2011:2556-2563.
</div>
<br/>
<div id='4'>
[4] Carreira J , Zisserman A . Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset Proceedings of IEEE Conference on Computer Vision and Pattern Recognition. Piscataway, NJ: IEEE, 2017:6299-6308.
</div>
<br/>
<div id='5'>
[5] Goyal R, Kahou S E, Michalski V. The “something something” video database for learning and evaluating visual common sense. arXiv:1706.04261,2017.
</div>
<br/>
<div id='6'>
[6] Sigurdsson G A , Varol Gül, Wang Xiaolong, et al. Hollywood in Homes: Crowdsourcing Data Collection for Activity Understanding. arXiv: 604.01753,2016
</div>
<br/>
<div id='7'>
[7] Kuehne H, Arslan A, Serre T. The Language of Actions Recovering the Syntax and Semantics of Goal-Directed Human Activities  Proceedings of IEEE Conference on Computer Vision and Pattern Recognition. Piscataway, NJ: IEEE, 2014.
</div>
<br/>
<div id='8'>
[8] Karpathy A , Toderici G , Shetty S , et al. Large-Scale Video Classification with Convolutional Neural Networks Proceedings of IEEE Conference on Computer Vision and Pattern Recognition. Piscataway, NJ: IEEE, 2014:1725-1732.
</div>
<br/>
<div id='9'>
[9] Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoo Tang,and Luc Van Gool. Temporal segment networks for action recognition in videos? In Proceedings of the European Conference on Computer Vision,pages 20–36. Springer, 2016.
</div>
<br/>
<div id='10'>
[10] Lin Ji , Gan Chuang , Han Song . TSM: Temporal Shift Module for Efficient Video Understanding. arXiv:1811.08383,2018.
</div>
<br/>
<div id='11'>
[11] Feichtenhofer C , Fan Haoqi , Malik J , et al. SlowFast Networks for Video Recognition. arXiv:1812.03982,2018.
</div>



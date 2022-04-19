 English | [中文](../../../zh-CN/model_zoo/estimation/ffa_ch.md)

##  FFA-Net Image Dehazing Model



[TOC]

## 1、 Introduction

This model is one of the top models in the field of image dehazing. The goal of this model is to dehaze the image, and the biggest feature is the large-scale convolution block structure based on the attention block. The structure of the attention block is shown in the following figure:

![img](../../../images/ffa_imgs/wps1.jpg)

The authors use two attention blocks for two different purposes. One is the channel attention block, which can automatically multiply each channel by different weights (between 0 and 1) to obtain the output of the attention block; the other is the pixel attention block, which can automatically Each pixel of the input is multiplied by a different weight to get the output of the attention block. The effect of attention is reflected by multiplying the weights. The ssim and psnr values between the dehaze image and the real haze-free image obtained by this model on the RESIDE dataset are significantly better than other models.



## 2、Dataset Preparation

For RESIDE-Standard dataset download and preparation, please refer to [RESIDE-Standard dataset download and preparation](../../dataset/RESIDE.md).



## 3、Model Training

### 3.1 Data preparation and model parameter preparation

Refer to the second part to download the dataset, and change the file path in the configuration file configs/FFA-cfg.yaml to your file path.



**Model parameter file and training log download address：**

Link：https://pan.baidu.com/s/1Q9RQI5bC35FUF2dhIqKamg   Code：gzao

file structure：

```
    PaddleVideo/data/FFA
        |-- vgg16_pretrained_weight.pdparams       #The parameter file of the VGG16 pre-training model 														used when the model loss function uses perloss
        |-- ITS2_3_19_400000_transform.pdparams    #The parameter file of the indoor dehazing model 														obtained by the reproduced model after 400000 steps 													of training
        |-- ITS_3_19_article_pretrained.pdparams   #The parameter file of the indoor dehazing model 														provided by the author
        |-- OTS_3_19_article_pretrained.pdparams   #The parameter file of the outdoor dehazing model 														provided by the author
        |-- logs                                   #training log folder
        	|-- train.log                          #full training log file
            |-- step 1-48000.ipynb                 #1-48000step reproduction training notebook file
            |-- step 48000-400000.ipynb            #48000-400000step to reproduce the training notebook 													file
```



### 3.2 model training

Train network on `ITS` dataset

 ```shell
python main.py -c configs/estimation/adds/FFA_cfg.yaml --validate
 ```


If you want to train network on `OTS` dataset，change the file_path of train dataset in configs/FFA-cfg.yaml and note the **suffix** parameter.

If you want to modify the parameters of the model, modify the parameters under MODEL in configs/FFA-cfg.yaml.

If you want to change the training epochs,  you need to change the **max_epoch** parameter under OPTIMIZER in configs/FFA-cfg.yaml at the same time, **max_epoch** must be consistent with **epochs**. For better results, train for at least 80 epochs.

The video memory required for this model training is too large, do not try to increase the **batchsize**.  If the video memory is insufficient, you can reduce the **batchsize** and PIPELINE/train/decode/**crop_size**.

The **gps** and **blocks** undel backbone related to the depth of the model.

The **perloss** under head related to the loss of the model, the default is False. If you want to change it to True, you need to download the vgg16 pre-training model parameter file in the link above to the corresponding location. In order to obtain better training results, it is recommended to download the corresponding file and change the parameter to True.. If you choose False，it means that only the l1 loss between the generated image and the clear image is used as the loss value.



### 3.3 Model evaluation

When testing the model, enter the following code in the console,  the downloaded model parameters mentioned above are used in the following code:

 ```shell
###Evaluate the model provided by the author###
python main.py --test -c configs/estimation/adds/FFA_cfg.yaml -w data/FFA/ITS_3_19_article_pretrained.pdparams

###Evaluate my reproduced model估###
python main.py --test -c configs/estimation/adds/FFA_cfg.yaml -w data/FFA/ITS2_3_19_400000_transform.pdparams
 ```

If you want to test your images, change the file_path of test dataset in configs/FFA-cfg.yaml,  and pay attention to whether the suffix is consistent.

If you want to test on your model, put the path of your model after -w.

The indoor model provided by the authors of the paper is in data/FFA/ITS_3_19_article_pretrained.pdparams，and theoutdoor  model is in data/FFA/OTS_3_19_article_pretrained.pdparams.

At the end of the model evaluation, an error will be encountered for unknown reasons, but the correct evaluation result will be output above the error part.

After the data set and model are successfully prepared, use the first command of the above example to evaluate,  It may take a long time to complete the evaluation of the model:

RESIDE dataset test accuracy:

| Backbone | Train dataset    | Test dataset        | SSIM   | PSNR  | checkpoints                          |
| -------- | ---------------- | ------------------- | ------ | ----- | ------------------------------------ |
| FFA      | RESIDE/ITS/train | RESIDE/SOTS/Indoor  | 0.9885 | 35.42 | ITS2_3_19_400000_transform.pdparams  |
| FFA      | RESIDE/ITS/train | RESIDE/SOTS/Indoor  | 0.9886 | 36.39 | ITS_3_19_article_pretrained.pdparams |
| FFA      | RESIDE/OTS/train | RESIDE/SOTS/outdoor | 0.9840 | 33.57 | OTS_3_19_article_pretrained.pdparams |



## 4、Tipc

### 4.1 Export the inference model

```bash
python tools/export_model.py -c configs/estimation/adds/FFA_cfg.yaml -p data/FFA/ITS2_3_19_400000_transform.pdparams -o inference/FFA
```

The above command will generate the model structure file `FFA.pdmodel` and model weight files `FFA.pdiparams` and `FFA.pdiparams.info` required for prediction, which are stored in the `inference/FFA/` directory.



### 4.2 Inference with a prediction engine

```bash
python tools/predict.py --input_file data/FFA/infer_example/ --config configs/estimation/adds/FFA_cfg.yaml --model_file inference/FFA/FFA.pdmodel --params_file inference/FFA/FFA.pdiparams --use_gpu=True --use_tensorrt=False
```

At the end of the inference, the dehazing image generated by the model will be saved by default, and the ssim and psnr values obtained from the test will be output.

The following is a sample image and the corresponding prediction (dehazing) map：

<p align='center'>
<img src="../../../images/ffa_imgs/1440_6.png" height="280px" width='360px'>
<img src='../../../images/ffa_imgs/1440_6_dehazed.png' height="280px" width='360px' >

An example of the output is as follows :

```
Current input image: 1440_6.png
pred dehazed image saved to: data/FFA/1440_6_dehazed.png
        ssim: 0.9845477233316967
        psnr: 33.88653046111545
```



### 4.3 Call the script to complete the training and push test in two steps

To test the `lite_train_lite_infer` mode of the basic training predict function, run：

```shell
# Prepare data
bash test_tipc/prepare.sh ./test_tipc/configs/FFA/train_infer_python.txt 'lite_train_lite_infer'
# run the test
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/FFA/train_infer_python.txt 'lite_train_lite_infer'
```



## 5、LICENSE

This project is released under the [Apache 2.0 license](https://github.com/PaddlePaddle/models/blob/release/2.2/community/repo_template/LICENSE) license.



## 6、References and Links

Paper address：https://arxiv.org/abs/1911.07559

Reference repoFFA-NET Github：https://github.com/zhilin007/FFA-Net

Thesis Reproduction Guide - CV Directionhttps://github.com/PaddlePaddle/models/blob/release%2F2.2/tutorials/article-implementation/ArticleReproduction_CV.md

How to integrate code into paddlevideo:https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/contribute/add_new_algorithm.md

Readme document template：https://github.com/PaddlePaddle/models/blob/release/2.2/community/repo_template/README.md




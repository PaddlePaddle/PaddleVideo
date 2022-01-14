# EIVideo - 交互式智能视频标注工具

[![Downloads](https://static.pepy.tech/personalized-badge/eivideo?period=total&units=international_system&left_color=grey&right_color=orange&left_text=EIVideo%20User)](https://pepy.tech/project/eivideo)
[![Downloads](https://static.pepy.tech/personalized-badge/qeivideo?period=total&units=international_system&left_color=grey&right_color=orange&left_text=QEIVideo%20User)](https://pepy.tech/project/qeivideo)
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/QPT-Family/EIVideo?include_prereleases)
![GitHub forks](https://img.shields.io/github/forks/QPT-Family/EIVideo)
![GitHub Repo stars](https://img.shields.io/github/stars/QPT-Family/EIVideo)
![GitHub](https://img.shields.io/github/license/QPT-Family/EIVideo)
![](https://img.shields.io/badge/%E6%B7%B1%E5%BA%A6%E9%80%82%E9%85%8D->Win7-9cf)

---


<div align="center">
<img width="600" alt="图片" src="https://user-images.githubusercontent.com/46156734/148925774-a04b641c-6a71-43ed-a7c0-d4b66e8d6e8a.png">
</div>
  
EIVideo，基于百度飞桨MA-Net交互式视频分割模型打造的交互式**智能视频**标注工具箱，只需简单标注几帧，即可完成全视频标注，若自动标注结果未达要求还可通过多次和视频交互而不断提升视频分割质量，直至对分割质量满意。  

戳 -> 了解相关[技术文章&模型原理](等待微信公众号)

<div align="center">
<img width="300" alt="图片" src="https://ai-studio-static-online.cdn.bcebos.com/f792bac0dd3b4f44ade7d744b58e908e2a85ed8718b541cfb6b2ce9fc8ad4374">
</div>

> 为了更好的解放双手，我们还提供了图形化界面工具QEIVideo，通过它我们可以不使用繁杂的命令方式来完成视频的智能标注工作。

---

### README目录

- [EAP - The Early Access Program 早期访问计划](#eap---the-early-access-program-早期访问计划)
- [使用方式](#使用方式)
  - [安装&运行](#安装运行)
    - [QPT包 - 适合无Python基础用户](#qpt包---适合无python基础用户)
    - [标准Python包 - 适合普通Python开发者](#标准python包---适合普通python开发者)
    - [开发版本 - 适合高阶开发者进行开发/社区贡献](#开发版本---适合高阶开发者进行开发社区贡献)
- [(Q)EIVideo产品规划安排](#qeivideo产品规划安排)
- [开源协议](#开源协议)

---

### EAP - The Early Access Program 早期访问计划

> Warning 当前图形化界面QEIVideo处于**极其初阶**的...建设阶段，并不能保证程序稳定性。

<div align="center"> <img width="100" alt="图片" src="https://user-images.githubusercontent.com/46156734/148927601-791362c0-0286-4fb9-b9d1-c193f7485de1.png"> </div>

当您选择使用QEIVideo作为图形化界面时，即可视为同意使用“可能会存在大量体验不佳”的EAP产品。

同样，您可选择借助基于[PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo) 实现的
交互式视频标注模型[EIVideo](https://github.com/QPT-Family/EIVideo/EIVideo) 进行二次开发，在此之上也可完成您需要的自定义图形化界面，后续也将提供二次开发指南。

<div align="center"> <img width="100" alt="图片" src="https://user-images.githubusercontent.com/46156734/148928046-b1490080-52f0-4a15-b7ff-11d54b135039.png"> </div>


> 如果您愿意参与到EIVideo或QEIVideo的建设中来，欢迎您与PMC取得联系 -> WX:GT_ZhangAcer  

## 使用方式
### 安装&运行
#### QPT包 - 适合无Python基础用户
自动化配置相关Python环境，但仅支持Windows7/10/11操作系统，且不对盗版Windows7做任何适配。  
下载地址：暂未上传
> 自动化部署工具由[QPT - 自动封装工具](https://github.com/QPT-Family/QPT) 支持  

#### 标准Python包 - 适合普通Python开发者
* 国际方式：
  ```shell
  python -m pip install eivideo
  python qeivideo
  ```
* 国内推荐：
  ```shell
  python -m pip install eivideo -i https://mirrors.bfsu.edu.cn/pypi/web/simple
  python qeivideo
  ```
> 上述命令仅适用于常规情况，若您安装了多个Python或修改了相关开发工具与配置，请自行修改相关命令使其符合您的开发环境。

#### 开发版本 - 适合高阶开发者进行开发/社区贡献

* 国际方式：
  ```shell
  git clone https://github.com/QPT-Family/EIVideo.git
  python -m pip install -r requirements.txt
  ```
* 国内推荐：
  ```shell
  # 请勿用于Push！！！
  git clone https://hub.fastgit.org/QPT-Family/EIVideo.git
  python -m pip install -r requirements.txt -i https://mirrors.bfsu.edu.cn/pypi/web/simple
  ```
* 运行程序
  ```shell
  # 进入工作目录
  cd 此处填写EIVideo所在的目录的绝对路径，且该目录下拥有EIVideo与QEIVideo两文件夹。
  # 运行
  python QEIVideo/start.py
  
  # 如运行时无法找到对应包，可选择下述方式添加环境变量来调整索引次序后执行python
  # Windows
  set PYTHONPATH=$pwd:$PYTHONPATH
  # Linux
  export PYTHONPATH=$pwd:$PYTHONPATH
  ```

> 上述命令仅适用于常规情况，若您安装了多个Python或修改了相关开发工具与配置，请自行修改相关命令使其符合您的开发环境。

## (Q)EIVideo产品规划安排  
> 由于QEIVideo由飞桨开源社区学生爱好者构成，所以在项目的产出过程中将会以学习为主进行开源贡献，如您原因与我们一同建设，我们也将非常欢迎~
<div align="center"> <img width="100" alt="图片" src="https://user-images.githubusercontent.com/46156734/148928475-b5b340b7-241d-4ddc-8155-70d98c6384a9.png"> </div>

- [x] EIVideo与Demo版QEIVideo发布0.1.0Alpha版本
- [ ] 完善QEIVideo，丰富基础标注功能，于Q1升级至1.0Alpha版本
- [ ] 回归QEIVideo稳定性，于Q2完成1.0正式版本发版
- [ ] 增加视频目标检测、分类任务的交互式标注功能。

### 开源协议
本项目使用GNU LESSER GENERAL PUBLIC LICENSE(LGPL)开源协议。  
> 因所使用的模型与数据集等原因，本项目中任一代码、参数均不可直接进行商用，如需商用请与我们取得联系。

### 引用来源
1. EIVideo模型以及相关源码、论文与项目 - [PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo)
2. 部分表情包来源 - [甘城なつき](https://www.pixiv.net/users/3036679)


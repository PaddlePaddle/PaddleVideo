# 花样滑冰动作识别

---
## 内容

- [视频数据处理方法](#视频数据处理方法)
- [模型训练预测方法](#模型训练预测方法)


<div align="center">
  <img src="Alex.gif" width=250/></div>

### 视频数据处理方法

 - 提供从视频中提取骨骼点数据的方法，方便用户自行提取数据进行测试。

 花样滑冰数据提取采用了openpose，通过其提供的demo或是相应的api来实现数据的提取，因此需要用户配置openpose环境。
 如下是通过花样滑冰数据集构建项目[Skeleton Scripts](https://github.com/HaxiSnake/skeleton_scripts)提取骨骼点数据方法的具体介绍。

 #### step1 安装openpose

 - 参考：https://github.com/CMU-Perceptual-Computing-Lab/openpose  

 #### step2 测试openpose提供demo

 - 这里通过测试openpose的demo程序来验证是否安装成功。

 demo1：检测视频中身体骨骼点（以linux系统为例）：

 ```bash
 ./build/examples/openpose/openpose.bin --video examples_video.avi --write_json output/ --display 0 --render_pose 0
 ```

 执行成功之后会在output/路径下生成视频每一帧骨骼点数据的json文件。

 demo2：检测视频中身体+面部+手部骨骼点（以linux系统为例）：

 ```bash
 ./build/examples/openpose/openpose.bin --video examples_video.avi --write_json output/ --display 0 --render_pose 0 --face --hand
 ```

 执行成功之后会在output/路径下生成视频每一帧身体+面部+手部骨骼点数据的json文件。

 #### step3 视频及相关信息处理

 - 由于[Skeleton Scripts](https://github.com/HaxiSnake/skeleton_scripts)为制作花样滑冰数据集所用，因此此处步骤可能存在不同程度误差，实际请用户自行调试代码。

 将要转化的花样滑冰视频储存到[Skeleton Scripts](https://github.com/HaxiSnake/skeleton_scripts)的指定路径（可自行创建）：
 ```bash
 ./skating2.0/skating63/
 ```

 同时需要用户自行完成对视频信息的提取，保存为label_skating63.csv文件，储存到如下路径中（可自行创建）：

 ```bash
 ./skating2.0/skating63/
 ./skating2.0/skating63_openpose_result/
 ```

 label_skating63.csv中格式如下：

 | 动作分类 | 视频文件名 | 视频帧数 | 动作标签 |
 | :----: | :----: | :----: | :---- |

 此处用户只需要输入视频文件名（无需后缀，默认后缀名为.mp4，其他格式需自行更改代码)，其他三项定义为空字符串即可，不同表项之间通过 ',' 分割。

 #### step4 执行skating_convert.py:

 - 注意，这一步需要根据用户对openpose的配置进行代码的更改，主要修改项为openpose路径、openpose-demo路径等，具体详见代码。

 本脚步原理是调用openpose提供的demo提取视频中的骨骼点，并进行数据格式清洗，最后将每个视频的提取结果结果打包成json文件，json文件储存在如下路径：

 ```bash
 ./skating2.0/skating63_openpose_result/label_skating63_data/
 ```

 #### step5 执行skating_gendata.py:

 将json文件整理为npy文件并保存，多个视频文件将保存为一个npy文件，保存路径为：

 ```bash
 ./skating2.0/skating63_openpose_result/skeleton_file/
 ```

 - 通过上述步骤就可以将视频数据转化为无标签的骨骼点数据。

 - 最后用户只需将npy数据输入送入网络开始模型测试，亦可通过预测引擎推理。


 ### 模型训练预测方法

 模型使用方法参考[ST-GCN模型文档](../../docs/zh-CN/model_zoo/recognition/stgcn.md)

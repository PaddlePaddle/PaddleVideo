简体中文 | [English](./readme_en.md)
# 模型服务化部署

## 简介

[Paddle Serving](https://github.com/PaddlePaddle/Serving) 旨在帮助深度学习开发者轻松部署在线预测服务，支持一键部署工业级的服务能力、客户端和服务端之间高并发和高效通信、并支持多种编程语言开发客户端。

该部分以 HTTP 预测服务部署为例，介绍怎样在 PaddleVideo 中使用 PaddleServing 部署模型服务。目前只支持 Linux 平台部署，暂不支持 Windows 平台。

## Serving 安装
Serving 官网推荐使用 docker 安装并部署 Serving 环境。首先需要拉取 docker 环境并创建基于 Serving 的 docker。

```bash
# 启动GPU docker
docker pull paddlepaddle/serving:0.7.0-cuda10.2-cudnn7-devel
nvidia-docker run -p 9292:9292 --name test -dit paddlepaddle/serving:0.7.0-cuda10.2-cudnn7-devel bash
nvidia-docker exec -it test bash

# 启动CPU docker
docker pull paddlepaddle/serving:0.7.0-devel
docker run -p 9292:9292 --name test -dit paddlepaddle/serving:0.7.0-devel bash
docker exec -it test bash
```

进入 docker 后，需要安装 Serving 相关的 python 包。
```bash
python3.7 -m pip install paddle-serving-client==0.7.0
python3.7 -m pip install paddle-serving-app==0.7.0

#若为CPU部署环境:
python3.7 -m pip install paddle-serving-server==0.7.0  # CPU
python3.7 -m pip install paddlepaddle==2.2.0           # CPU

#若为GPU部署环境
python3.7 -m pip install paddle-serving-server-gpu==0.7.0.post102  # GPU with CUDA10.2 + TensorRT6
python3.7 -m pip install paddlepaddle-gpu==2.2.0                   # GPU with CUDA10.2

#其他GPU环境需要确认环境再选择执行哪一条
python3.7 -m pip install paddle-serving-server-gpu==0.7.0.post101  # GPU with CUDA10.1 + TensorRT6
python3.7 -m pip install paddle-serving-server-gpu==0.7.0.post112  # GPU with CUDA11.2 + TensorRT8
```

* 如果安装速度太慢，可以通过 `-i https://pypi.tuna.tsinghua.edu.cn/simple` 更换源，加速安装过程。

* 更多环境和对应的安装包详见：https://github.com/PaddlePaddle/Serving/blob/v0.9.0/doc/Install_Linux_Env_CN.md

## 行为识别服务部署
### 模型转换
使用 PaddleServing 做服务化部署时，需要将保存的 inference 模型转换为 Serving 模型。下面以 PP-TSM 模型为例，介绍如何部署行为识别服务。
- 下载 PP-TSM 推理模型并转换为 Serving 模型：

  ```bash
  # 进入PaddleVideo目录
  cd PaddleVideo
  # 下载推理模型并解压到./inference下
  mkdir ./inference
  pushd ./inference
  wget  https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM.zip
  unzip ppTSM.zip
  popd

  # 转换成 Serving 模型
  pushd deploy/cpp_serving
  python3.7 -m paddle_serving_client.convert \
  --dirname ../../inference/ppTSM \
  --model_filename ppTSM.pdmodel \
  --params_filename ppTSM.pdiparams \
  --serving_server ./ppTSM_serving_server \
  --serving_client ./ppTSM_serving_client
  popd
  ```

  | 参数              | 类型 | 默认值             | 描述                                                         |
  | ----------------- | ---- | ------------------ | ------------------------------------------------------------ |
  | `dirname`         | str  | -                  | 需要转换的模型文件存储路径，Program结构文件和参数文件均保存在此目录。 |
  | `model_filename`  | str  | None               | 存储需要转换的模型Inference Program结构的文件名称。如果设置为None，则使用 `__model__` 作为默认的文件名 |
  | `params_filename` | str  | None               | 存储需要转换的模型所有参数的文件名称。当且仅当所有模型参数被保>存在一个单独的二进制文件中，它才需要被指定。如果模型参数是存储在各自分离的文件中，设置它的值为None |
  | `serving_server`  | str  | `"serving_server"` | 转换后的模型文件和配置文件的存储路径。默认值为serving_server |
  | `serving_client`  | str  | `"serving_client"` | 转换后的客户端配置文件存储路径。默认值为serving_client       |

- 推理模型转换完成后，会在`deploy/cpp_serving`文件夹下生成 `ppTSM_serving_client` 和 `ppTSM_serving_server` 两个文件夹，具备如下格式：
  ```bash
  PaddleVideo/deploy/cpp_serving
  ├── ppTSM_serving_client
  │   ├── serving_client_conf.prototxt
  │   └── serving_client_conf.stream.prototxt
  └── ppTSM_serving_server
      ├── ppTSM.pdiparams
      ├── ppTSM.pdmodel
      ├── serving_server_conf.prototxt
      └── serving_server_conf.stream.prototxt
  ```
  得到模型文件之后，需要分别修改 `ppTSM_serving_client` 下的 `serving_client_conf.prototxt` 和 `ppTSM_serving_server` 下的 `serving_server_conf.prototxt`，将两份文件中`fetch_var` 下的 `alias_name` 均改为 `outputs`

  **备注**:  Serving 为了兼容不同模型的部署，提供了输入输出重命名的功能。这样，不同的模型在推理部署时，只需要修改配置文件的`alias_name`即可，无需修改代码即可完成推理部署。
  修改后的`serving_server_conf.prototxt`如下所示:

  ```yaml
  feed_var {
    name: "data_batch_0"
    alias_name: "data_batch_0"
    is_lod_tensor: false
    feed_type: 1
    shape: 8
    shape: 3
    shape: 224
    shape: 224
  }
  fetch_var {
    name: "linear_2.tmp_1"
    alias_name: "outputs"
    is_lod_tensor: false
    fetch_type: 1
    shape: 400
  }
  ```
### 服务部署和请求
`cpp_serving` 目录包含了启动 pipeline 服务、C++ serving服务和发送预测请求的代码，具体包括：
  ```bash
  run_cpp_serving.sh          # 启动C++ serving server端的脚本
  pipeline_http_client.py     # client端发送数据并获取预测结果的脚本
  paddle_env_install.sh       # 安装C++ serving环境脚本
  preprocess_ops.py           # 存放预处理函数的文件
  ```
#### C++ Serving
- 进入工作目录：
  ```bash
  cd deploy/cpp_serving
  ```

- 启动服务：
  ```bash
  # 在后台启动，过程中打印输出的日志会重定向保存到nohup.txt中，可以使用tailf nohup.txt查看输出
  bash run_cpp_serving.sh
  ```

- 发送请求并获取结果：
  ```bash
  python3.7 serving_client.py \
  -n PPTSM \
  -c ./ppTSM_serving_client/serving_client_conf.prototxt \
  --input_file=../../data/example.avi
  ```
成功运行后，模型预测的结果会打印在 cmd 窗口中，结果如下：

  ```bash
  I0510 04:33:00.110025 37097 naming_service_thread.cpp:202] brpc::policy::ListNamingService("127.0.0.1:9993"): added 1
  I0510 04:33:01.904764 37097 general_model.cpp:490] [client]logid=0,client_cost=1640.96ms,server_cost=1623.21ms.
  {'class_id': '[5]', 'prob': '[0.9907387495040894]'}
  ```
**如果过程中报错显示找不到libnvinfer.so.6，可以执行脚本`paddle_env_install.sh`安装相关环境**
  ```bash
  bash paddle_env_install.sh
  ```


## FAQ
**Q1**： 发送请求后没有结果返回或者提示输出解码报错

**A1**： 启动服务和发送请求时不要设置代理，可以在启动服务前和发送请求前关闭代理，关闭代理的命令是：
```
unset https_proxy
unset http_proxy
```

# Paddle2ONNX 测试

Paddle2ONNX 测试的主程序为`test_paddle2onnx.sh`，可以测试基于Paddle2ONNX的模型转换和onnx预测功能。


## 1. 测试结论汇总

- 推理相关：

| 算法名称 | 模型名称 | device_CPU | device_GPU | batchsize |
|  :----:   |  :----: |   :----:   |  :----:  |   :----:   |
|  PP-TSN   |  pptsn_k400_videos |  支持 | 支持 | 1 |


## 2. 测试流程

### 2.1 准备数据

用于基础训练推理测试的数据位于`data\example.avi`, 已经存在于目录中无需下载


### 2.2 准备环境


- 安装PaddlePaddle：如果您已经安装了2.2或者以上版本的paddlepaddle，那么无需运行下面的命令安装paddlepaddle。
    ```
    # 需要安装2.2及以上版本的Paddle
    # 安装GPU版本的Paddle
    python3.7 -m pip install paddlepaddle-gpu==2.2.0
    # 安装CPU版本的Paddle
    python3.7 -m pip install paddlepaddle==2.2.0
    ```

- 安装依赖
    ```
    python3.7 -m pip install -r requirements.txt
    ```

- 安装 Paddle2ONNX
    ```
    python3.7 -m pip install paddle2onnx
    ```

- 安装 ONNXRuntime
    ```
    # 建议安装 1.9.0 版本，可根据环境更换版本号
    python3.7 -m pip install onnxruntime==1.9.0
    ```


### 2.3 功能测试

测试方法如下所示，希望测试不同的模型文件，只需更换为自己的参数配置文件，即可完成对应模型的测试。

```bash
bash test_tipc/test_paddle2onnx.sh ${your_params_file}
```

以`PP-TSN`的`Paddle2ONNX 测试`为例，命令如下所示。

```bash
bash test_tipc/prepare.sh test_tipc/configs/PP-TSN/paddle2onnx_infer_python.txt paddle2onnx_infer

bash test_tipc/test_paddle2onnx.sh test_tipc/configs/PP-TSN/paddle2onnx_infer_python.txt
```

输出结果如下，表示命令运行成功。

```
 Run successfully with command -  paddle2onnx --model_dir=./inference/ppTSN/ --model_filename=ppTSN.pdmodel --params_filename=ppTSN.pdiparams --save_file=./inference/ppTSN/ppTSN.onnx --opset_version=10 --enable_onnx_checker=True!
 Run successfully with command - python3.7 ./deploy/paddle2onnx/predict_onnx.py --config=./configs/recognition/pptsn/pptsn_k400_videos.yaml --input_file=./data/example.avi --onnx_file=./inference/ppTSN/ppTSN.onnx > ./log/PP-TSN//paddle2onnx_infer_cpu.log 2>&1 !
```

预测结果会自动保存在 `./log/PP-TSN/paddle2onnx_infer_cpu.log` ，可以看到onnx运行结果：
```
W0524 15:28:29.723601 75410 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.2, Runtime API Version: 10.2
W0524 15:28:29.982623 75410 gpu_resources.cc:91] device: 0, cuDNN Version: 7.6.
Inference model(ppTSN)...
Current video file: ./data/example.avi
	top-1 class: 5
	top-1 score: 0.9998553991317749
```

如果运行失败，也会在终端中输出运行失败的日志信息以及对应的运行命令。可以基于该命令，分析运行失败的原因。

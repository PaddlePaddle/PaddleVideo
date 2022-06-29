# Linux GPU/CPU PACT离线量化训练推理测试

Linux GPU/CPU PACT离线量化训练推理测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能。

## 1. 测试结论汇总

- 训练相关：

| 算法名称 | 模型名称 | 单机单卡 |
|  :----: |   :----:  |    :----:  |
|  PP-TSM  | pptsm_k400_frames_uniform | PACT离线量化训练 |


- 推理相关：

| 算法名称 | 模型名称 | device_CPU | device_GPU | batchsize |
|  :----:   |  :----: |   :----:   |  :----:  |   :----:   |
|  PP-TSM   |  pptsm_k400_frames_uniform |  支持 | 支持 | 1 |


## 2. 测试流程

### 2.1 准备数据和模型

用于量化训练推理测试的数据位于`./data/k400/k400_rawframes_small.tar`，直接解压即可（如果已经解压完成，则无需运行下面的命令）。

```bash
pushd ./data/k400
wget -nc https://videotag.bj.bcebos.com/Data/k400_rawframes_small.tar
tar -xf k400_rawframes_small.tar
popd
```

准备离线量化需要的inference模型，以PP-TSM为例，可以直接下载：
```bash
mkdir ./inference
pushd ./inference
wget -nc https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM.zip --no-check-certificate
unzip ppTSM.zip
popd
```

离线量化的训练流程，可参考[文档](../../deploy/slim/readme.md)。

### 2.2 准备环境


- 安装PaddlePaddle：如果您已经安装了2.2或者以上版本的paddlepaddle，那么无需运行下面的命令安装paddlepaddle。
    ```
    # 需要安装2.2及以上版本的Paddle
    # 安装GPU版本的Paddle
    python3.7 -m pip install paddlepaddle-gpu==2.2.0
    # 安装CPU版本的Paddle
    python3.7 -m pip install paddlepaddle==2.2.0
    ```
- 安装PaddleSlim
    ```
    python3.7 -m pip install paddleslim==2.2.0
    ```
- 安装依赖
    ```
    python3.7 -m pip install  -r requirements.txt
    ```
- 安装AutoLog（规范化日志输出工具）
    ```
    python3.7 -m pip install  https://paddleocr.bj.bcebos.com/libs/auto_log-1.2.0-py3-none-any.whl
    ```


### 2.3 功能测试

以`pptsm_k400_frames_uniform`的`Linux GPU/CPU PACT离线量化训练推理测试`为例，命令如下所示。

```bash
# 准备数据和推理模型
bash test_tipc/prepare.sh ./test_tipc/configs/PP-TSM/train_ptq_infer_python.txt klquant_whole_infer
# 执行测试命令
bash test_tipc/test_train_ptq_python.sh ./test_tipc/configs/PP-TSM/train_ptq_infer_python.txt klquant_whole_infer
```

在`./log/PP-TSM/klquant_whole_infer/results_python.log`中会记录命令的执行状态，出现以下信息表示命令运行成功。

```log
Run successfully with command - python3.7 deploy/slim/quant_post_static.py --use_gpu=True --config=./configs/recognition/pptsm/pptsm_k400_frames_uniform_quantization.yaml -o inference_model_dir=./inference/ppTSM -o DATASET.batch_nums=2 -o DATASET.batch_size=2 -o DATASET.quant.data_prefix=./data/k400/rawframes -o DATASET.quant.file_path=./data/k400/val_small_frames.list -o quant_output_dir=./inference/ppTSM/quant_model!
Run successfully with command - python3.7 ./tools/predict.py --use_gpu=True --config=./configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml --model_file=./inference/ppTSM/quant_model/__model__ --params_file=./inference/ppTSM/quant_model/__params__ --batch_size=1 --input_file=./data/example.avi --enable_benchmark=True > ./log/PP-TSM/klquant_whole_infer/python_infer_gpu_batchsize_1.log 2>&1 !
Run successfully with command - python3.7 ./tools/predict.py --use_gpu=False --config=./configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml --model_file=./inference/ppTSM/quant_model/__model__ --params_file=./inference/ppTSM/quant_model/__params__ --batch_size=1 --input_file=./data/example.avi --enable_benchmark=True > ./log/PP-TSM/klquant_whole_infer/python_infer_cpu_batchsize_1.log 2>&1 !
```
同时在命令窗口中会打印出统计的benchmark信息
```bash
......此处省略其它内容
[2022/05/07 05:23:38] root INFO: ---------------------- Env info ----------------------
[2022/05/07 05:23:38] root INFO:  OS_version: Ubuntu 16.04
[2022/05/07 05:23:38] root INFO:  CUDA_version: 10.2.89
[2022/05/07 05:23:38] root INFO:  CUDNN_version: 7.6.5
[2022/05/07 05:23:38] root INFO:  drivier_version: 440.33.01
[2022/05/07 05:23:38] root INFO: ---------------------- Paddle info ----------------------
[2022/05/07 05:23:38] root INFO:  paddle_version: 2.2.2
[2022/05/07 05:23:38] root INFO:  paddle_commit: b031c389938bfa15e15bb20494c76f86289d77b0
[2022/05/07 05:23:38] root INFO:  log_api_version: 1.0
[2022/05/07 05:23:38] root INFO: ----------------------- Conf info -----------------------
[2022/05/07 05:23:38] root INFO:  runtime_device: gpu
[2022/05/07 05:23:38] root INFO:  ir_optim: True
[2022/05/07 05:23:38] root INFO:  enable_memory_optim: True
[2022/05/07 05:23:38] root INFO:  enable_tensorrt: False
[2022/05/07 05:23:38] root INFO:  enable_mkldnn: False
[2022/05/07 05:23:38] root INFO:  cpu_math_library_num_threads: 1
[2022/05/07 05:23:38] root INFO: ----------------------- Model info ----------------------
[2022/05/07 05:23:38] root INFO:  model_name: ppTSM
[2022/05/07 05:23:38] root INFO:  precision: fp32
[2022/05/07 05:23:38] root INFO: ----------------------- Data info -----------------------
[2022/05/07 05:23:38] root INFO:  batch_size: 1
[2022/05/07 05:23:38] root INFO:  input_shape: dynamic
[2022/05/07 05:23:38] root INFO:  data_num: 15
[2022/05/07 05:23:38] root INFO: ----------------------- Perf info -----------------------
[2022/05/07 05:23:38] root INFO:  cpu_rss(MB): 2452.332, gpu_rss(MB): 23156.0, gpu_util: 100.0%
[2022/05/07 05:23:38] root INFO:  total time spent(s): 2.318
[2022/05/07 05:23:38] root INFO:  preprocess_time(ms): 159.4567, inference_time(ms): 31.3484, postprocess_time(ms): 2.3631
 Run successfully with command - python3.7 ./tools/predict.py --use_gpu=True --config=./configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml --model_file=./inference/ppTSM/quant_model/__model__ --params_file=./inference/ppTSM/quant_model/__params__ --batch_size=1 --input_file=./data/example.avi --enable_benchmark=True > ./log/PP-TSM/klquant_whole_infer/python_infer_gpu_batchsize_1.log 2>&1 !

......此处省略其它内容
[2022/05/07 05:24:24] root INFO: ---------------------- Env info ----------------------
[2022/05/07 05:24:24] root INFO:  OS_version: Ubuntu 16.04
[2022/05/07 05:24:24] root INFO:  CUDA_version: 10.2.89
[2022/05/07 05:24:24] root INFO:  CUDNN_version: 7.6.5
[2022/05/07 05:24:24] root INFO:  drivier_version: 440.33.01
[2022/05/07 05:24:24] root INFO: ---------------------- Paddle info ----------------------
[2022/05/07 05:24:24] root INFO:  paddle_version: 2.2.2
[2022/05/07 05:24:24] root INFO:  paddle_commit: b031c389938bfa15e15bb20494c76f86289d77b0
[2022/05/07 05:24:24] root INFO:  log_api_version: 1.0
[2022/05/07 05:24:24] root INFO: ----------------------- Conf info -----------------------
[2022/05/07 05:24:24] root INFO:  runtime_device: cpu
[2022/05/07 05:24:24] root INFO:  ir_optim: True
[2022/05/07 05:24:24] root INFO:  enable_memory_optim: True
[2022/05/07 05:24:24] root INFO:  enable_tensorrt: False
[2022/05/07 05:24:24] root INFO:  enable_mkldnn: False
[2022/05/07 05:24:24] root INFO:  cpu_math_library_num_threads: 1
[2022/05/07 05:24:24] root INFO: ----------------------- Model info ----------------------
[2022/05/07 05:24:24] root INFO:  model_name: ppTSM
[2022/05/07 05:24:24] root INFO:  precision: fp32
[2022/05/07 05:24:24] root INFO: ----------------------- Data info -----------------------
[2022/05/07 05:24:24] root INFO:  batch_size: 1
[2022/05/07 05:24:24] root INFO:  input_shape: dynamic
[2022/05/07 05:24:24] root INFO:  data_num: 15
[2022/05/07 05:24:24] root INFO: ----------------------- Perf info -----------------------
[2022/05/07 05:24:24] root INFO:  cpu_rss(MB): 2170.8125, gpu_rss(MB): None, gpu_util: None%
[2022/05/07 05:24:24] root INFO:  total time spent(s): 24.2831
[2022/05/07 05:24:24] root INFO:  preprocess_time(ms): 167.4745, inference_time(ms): 1854.3527, postprocess_time(ms): 1.765
 Run successfully with command - python3.7 ./tools/predict.py --use_gpu=False --config=./configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml --model_file=./inference/ppTSM/quant_model/__model__ --params_file=./inference/ppTSM/quant_model/__params__ --batch_size=1 --input_file=./data/example.avi --enable_benchmark=True > ./log/PP-TSM/klquant_whole_infer/python_infer_cpu_batchsize_1.log 2>&1 !
```

如果运行失败，也会在终端中输出运行失败的日志信息以及对应的运行命令。可以基于该命令，分析运行失败的原因。

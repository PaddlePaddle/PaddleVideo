# Linux GPU/CPU 混合精度训练推理测试

Linux GPU/CPU 混合精度训练推理测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能。

## 1. 测试结论汇总

- 训练相关：

    | 算法名称       | 模型名称 | 单机单卡 | 单机多卡 |
    | :----        |   :----  |    :----  |  :----   |
    | PP-TSM       | pptsm_k400_frames_uniform | 混合精度训练 | 混合精度训练 |
    | PP-TSN       | pptsn_k400_videos | 混合精度训练 | 混合精度训练 |
    | AGCN         | agcn_fsd | 混合精度训练 | - |
    | STGCN        | stgcn_fsd | 混合精度训练 | - |
    | TimeSformer  | timesformer_k400_videos | 混合精度训练 | 混合精度训练 |
    | SlowFast     | slowfast | 混合精度训练 | 混合精度训练 |
    | TSM          | tsm_k400_frames | 混合精度训练 | 混合精度训练 |
    | TSN          | tsn_k400_frames | 混合精度训练 | 混合精度训练 |
    | AttentionLSTM| attention_lstm_youtube8m | 混合精度训练 | 混合精度训练 |
    | BMN          | bmn | 混合精度训练 | 混合精度训练 |


- 推理相关：

    | 算法名称      | 模型名称 | device_CPU | device_GPU | batchsize |
    | :----      |  :---- |   :----   |  :----  |   :----   |
    | PP-TSM       |  pptsm_k400_frames_uniform  |  支持  | 支持 | 1/2 |
    | PP-TSN       |  pptsn_k400_videos  |  支持  | 支持 | 1/2 |
    | AGCN         |  agcn_fsd  |  支持  | 支持 | 1/2 |
    | STGCN        |  stgcn_fsd  |  支持  | 支持 | 1/2 |
    | TimeSformer  |  timesformer_k400_videos  |  支持  | 支持 | 1/2 |
    | SlowFast     |  slowfast  |  支持  | 支持 | 1/2 |
    | TSM          |  tsm_k400_frames  |  支持  | 支持 | 1/2 |
    | TSN          |  tsn_k400_frames  |  支持  | 支持 | 1/2 |
    | AttentionLSTM|  attention_lstm_youtube8m  |  支持  | 支持 | 1/2 |
    | BMN          |  bmn  |  支持  | 支持 | 1 |
## 2. 测试流程

### 2.1 准备环境


- 安装PaddlePaddle：如果您已经安装了2.2或者以上版本的paddlepaddle，那么无需运行下面的命令安装paddlepaddle。
    ```
    # 需要安装2.2及以上版本的Paddle
    # 安装GPU版本的Paddle
    pip install paddlepaddle-gpu==2.2.0
    # 安装CPU版本的Paddle
    pip install paddlepaddle==2.2.0
    ```

- 安装依赖
    ```
    pip install  -r requirements.txt
    ```
- 安装AutoLog（规范化日志输出工具）
    ```
    pip install  https://paddleocr.bj.bcebos.com/libs/auto_log-1.2.0-py3-none-any.whl
    ```

### 2.2 功能测试


测试方法如下所示，希望测试不同的模型文件，只需更换为自己的参数配置文件，即可完成对应模型的测试。

```bash
bash test_tipc/test_train_inference_python.sh ${your_params_file_path} lite_train_lite_infer
```

以`PP-TSM`的`Linux GPU/CPU 混合精度(默认优化等级为O2)训练推理测试`为例，命令如下所示。

```bash
bash test_tipc/prepare.sh test_tipc/configs/PP-TSM/train_amp_infer_python.txt lite_train_lite_infer
```

```bash
bash test_tipc/test_train_inference_python.sh test_tipc/configs/PP-TSM/train_amp_infer_python.txt lite_train_lite_infer
```

输出结果如下，表示命令运行成功。

```bash
Run successfully with command - python3.7 main.py --amp --amp_level='O1' --validate -c configs/recognition/tsm/tsm_k400_frames.yaml --seed 1234 --max_iters=30    -o output_dir=./test_tipc/output/TSM/amp_train_gpus_0_autocast_null -o epochs=2 -o MODEL.backbone.pretrained='data/ResNet50_pretrain.pdparams' -o DATASET.batch_size=2 -o DATASET.train.file_path='data/k400/train_small_frames.list' -o DATASET.valid.file_path='data/k400/val_small_frames.list' -o DATASET.test.file_path='data/k400/val_small_frames.list'     !

........

Run successfully with command - python3.7 tools/predict.py --config configs/recognition/tsm/tsm_k400_frames.yaml --use_gpu=False --enable_mkldnn=False --cpu_threads=6 --model_file=./test_tipc/output/TSM/amp_train_gpus_0,1_autocast_null/inference.pdmodel --batch_size=2 --input_file=./data/example.avi --enable_benchmark=False --precision=fp32 --params_file=./test_tipc/output/TSM/amp_train_gpus_0,1_autocast_null/inference.pdiparams > ./test_tipc/output/TSM/python_infer_cpu_usemkldnn_False_threads_6_precision_fp32_batchsize_2.log 2>&1 !

```

在开启benchmark选项时，可以得到测试的详细数据，包含运行环境信息（系统版本、CUDA版本、CUDNN版本、驱动版本），Paddle版本信息，参数设置信息（运行设备、线程数、是否开启内存优化等），模型信息（模型名称、精度），数据信息（batchsize、是否为动态shape等），性能信息（CPU/GPU的占用、运行耗时、预处理耗时、推理耗时、后处理耗时），内容如下所示：

```log
[2022/03/18 12:01:21] root INFO: ---------------------- Env info ----------------------
[2022/03/18 12:01:21] root INFO:  OS_version: Ubuntu 16.04
[2022/03/18 12:01:21] root INFO:  CUDA_version: 10.2.89
[2022/03/18 12:01:21] root INFO:  CUDNN_version: 7.6.5
[2022/03/18 12:01:21] root INFO:  drivier_version: 440.64.00
[2022/03/18 12:01:21] root INFO: ---------------------- Paddle info ----------------------
[2022/03/18 12:01:21] root INFO:  paddle_version: 0.0.0
[2022/03/18 12:01:21] root INFO:  paddle_commit: 6849d33b62cacccb27797375a212e37a47ca9484
[2022/03/18 12:01:21] root INFO:  log_api_version: 1.0
[2022/03/18 12:01:21] root INFO: ----------------------- Conf info -----------------------
[2022/03/18 12:01:21] root INFO:  runtime_device: gpu
[2022/03/18 12:01:21] root INFO:  ir_optim: True
[2022/03/18 12:01:21] root INFO:  enable_memory_optim: True
[2022/03/18 12:01:21] root INFO:  enable_tensorrt: False
[2022/03/18 12:01:21] root INFO:  enable_mkldnn: False
[2022/03/18 12:01:21] root INFO:  cpu_math_library_num_threads: 1
[2022/03/18 12:01:21] root INFO: ----------------------- Model info ----------------------
[2022/03/18 12:01:21] root INFO:  model_name: ppTSM
[2022/03/18 12:01:21] root INFO:  precision: fp32
[2022/03/18 12:01:21] root INFO: ----------------------- Data info -----------------------
[2022/03/18 12:01:21] root INFO:  batch_size: 2
[2022/03/18 12:01:21] root INFO:  input_shape: dynamic
[2022/03/18 12:01:21] root INFO:  data_num: 30
[2022/03/18 12:01:21] root INFO: ----------------------- Perf info -----------------------
[2022/03/18 12:01:21] root INFO:  cpu_rss(MB): 2062.625, gpu_rss(MB): 2111.0, gpu_util: 100.0%
[2022/03/18 12:01:21] root INFO:  total time spent(s): 5.5024
[2022/03/18 12:01:21] root INFO:  preprocess_time(ms): 247.8535, inference_time(ms): 26.6164, postprocess_time(ms): 0.6504
```

该信息可以在运行log中查看，以`PP-TSM`为例，上述的log完整信息文件位置在`./test_tipc/output/PP-TSM/python_infer_gpu_usetrt_False_precision_fp32_batchsize_2.log`。

如果运行失败，也会在终端中输出运行失败的日志信息以及对应的运行命令。可以基于该命令，分析运行失败的原因。

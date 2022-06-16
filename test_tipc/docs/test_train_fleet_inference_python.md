# Linux GPU/CPU 多机多卡训练推理测试

Linux GPU/CPU 多机多卡训练推理测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的多机多卡模型训练、评估、推理等基本功能。

## 1. 测试结论汇总

- 训练相关：

  | 算法名称 |         模型名称          |  多机多卡  |
  | :------: | :-----------------------: | :--------: |
  |  PP-TSM  | pptsm_k400_frames_uniform | 分布式训练 |
  |  PP-TSN  |     pptsn_k400_frames     | 分布式训练 |


- 推理相关：

  | 算法名称 |         模型名称          | device_CPU | device_GPU | batchsize |
  | :------: | :-----------------------: | :--------: | :--------: | :-------: |
  |  PP-TSM  | pptsm_k400_frames_uniform |    支持    |    支持    |     1     |
  |  PP-TSN  |     pptsn_k400_frames     |    支持    |    支持    |     1     |


## 2. 测试流程

运行环境配置请参考[文档](./install.md)的内容配置TIPC的运行环境。

**下面以 PP-TSM 模型为例，介绍测试流程**

### 2.1 功能测试

#### 2.1.1 修改配置文件

首先，修改配置文件`./test_tipc/configs/PP-TSM/train_linux_gpu_fleet_normal_infer_python_linux_gpu_cpu.txt.txt`中的`gpu_list`设置：假设两台机器的`ip`地址分别为`192.168.0.1`和`192.168.0.2`，则对应的配置文件`gpu_list`字段需要修改为`gpu_list:192.168.0.1,192.168.0.2;0,1`。

**`ip`地址查看命令为`ifconfig`，在`inet addr:`字段后的即为ip地址**。


#### 2.1.2 准备数据

运行`prepare.sh`准备数据和模型，数据准备命令如下所示。

```shell
bash test_tipc/prepare.sh ./test_tipc/configs/PP-TSM/train_linux_gpu_fleet_normal_infer_python_linux_gpu_cpu.txt.txt lite_train_lite_infer
```

**注意：** 由于是多机训练，这里需要在所有节点上都运行一次启动上述命令来准备数据。

#### 2.1.3 修改起始端口开始测试

在多机的节点上使用下面的命令设置分布式的起始端口（否则后面运行的时候会由于无法找到运行端口而hang住），一般建议设置在`10000~20000`之间。

```shell
export FLAGS_START_PORT=17000
```
**注意：** 上述修改起始端口命令同样需要在所有节点上都执行一次。

接下来就可以开始执行测试，命令如下所示。
```shell
bash test_tipc/test_train_inference_python.sh  ./test_tipc/configs/PP-TSM/train_linux_gpu_fleet_normal_infer_python_linux_gpu_cpu.txt.txt
```

**注意：** 由于是多机训练，这里需要在所有的节点上均运行启动上述命令进行测试。


#### 2.1.4 输出结果

输出结果保存在`./test_tipc/output/PP-TSM/results_python.log`，内容如下，以`Run successfully`开头表示测试命令正常，否则为测试失败。

```bash
Run successfully with command - python3.7 -B -m paddle.distributed.launch --ips=10.255.125.25,10.255.128.22 --gpus="0,1" main.py --validate -c configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml --seed 1234 --max_iters=30 -o output_dir=./test_tipc/output/PP-TSM/norm_train_gpus_0,1_autocast_null_nodes_2 -o MODEL.backbone.pretrained='data/ResNet50_vd_ssld_v2_pretrained.pdparams' -o epochs=2 -o DATASET.batch_size=2 -o DATASET.train.file_path='data/k400/train_small_frames.list' -o DATASET.valid.file_path='data/k400/val_small_frames.list' -o DATASET.test.file_path='data/k400/val_small_frames.list'  !
...
...
Run successfully with command - python3.7 tools/predict.py --config configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml --use_gpu=True --use_tensorrt=True --precision=fp16 --model_file=./test_tipc/output/PP-TSM/norm_train_gpus_0,1_autocast_null_nodes_2/inference.pdmodel --batch_size=2 --input_file=./data/example.avi --enable_benchmark=True --params_file=./test_tipc/output/PP-TSM/norm_train_gpus_0,1_autocast_null_nodes_2/inference.pdiparams > ./test_tipc/output/PP-TSM/python_infer_gpu_usetrt_True_precision_fp16_batchsize_2.log 2>&1
 !
```

在配置文件中默认设置`-o Global.benchmark:True`表示开启benchmark选项，此时可以得到测试的详细数据，包含运行环境信息（系统版本、CUDA版本、CUDNN版本、驱动版本），Paddle版本信息，参数设置信息（运行设备、线程数、是否开启内存优化等），模型信息（模型名称、精度），数据信息（batchsize、是否为动态shape等），性能信息（CPU,GPU的占用、运行耗时、预处理耗时、推理耗时、后处理耗时），内容如下所示：

```log
[2022/06/08 15:44:23] root INFO: ---------------------- Env info ----------------------
[2022/06/08 15:44:23] root INFO:  OS_version: CentOS 6.10
[2022/06/08 15:44:23] root INFO:  CUDA_version: 10.1.243
[2022/06/08 15:44:23] root INFO:  CUDNN_version: None.None.None
[2022/06/08 15:44:23] root INFO:  drivier_version: 460.32.03
[2022/06/08 15:44:23] root INFO: ---------------------- Paddle info ----------------------
[2022/06/08 15:44:23] root INFO:  paddle_version: 2.3.0-rc0
[2022/06/08 15:44:23] root INFO:  paddle_commit: 5d4980c052583fec022812d9c29460aff7cdc18b
[2022/06/08 15:44:23] root INFO:  log_api_version: 1.0
[2022/06/08 15:44:23] root INFO: ----------------------- Conf info -----------------------
[2022/06/08 15:44:23] root INFO:  runtime_device: gpu
[2022/06/08 15:44:23] root INFO:  ir_optim: True
[2022/06/08 15:44:23] root INFO:  enable_memory_optim: True
[2022/06/08 15:44:23] root INFO:  enable_tensorrt: False
[2022/06/08 15:44:23] root INFO:  enable_mkldnn: False
[2022/06/08 15:44:23] root INFO:  cpu_math_library_num_threads: 1
[2022/06/08 15:44:23] root INFO: ----------------------- Model info ----------------------
[2022/06/08 15:44:23] root INFO:  model_name: ppTSM
[2022/06/08 15:44:23] root INFO:  precision: fp32
[2022/06/08 15:44:23] root INFO: ----------------------- Data info -----------------------
[2022/06/08 15:44:23] root INFO:  batch_size: 2
[2022/06/08 15:44:23] root INFO:  input_shape: dynamic
[2022/06/08 15:44:23] root INFO:  data_num: 8
[2022/06/08 15:44:23] root INFO: ----------------------- Perf info -----------------------
[2022/06/08 15:44:23] root INFO:  cpu_rss(MB): 2941.0625, gpu_rss(MB): 1212.0, gpu_util: 3.0%
[2022/06/08 15:44:23] root INFO:  total time spent(s): 1.3603
[2022/06/08 15:44:23] root INFO:  preprocess_time(ms): 244.0361, inference_time(ms): 27.6035, postprocess_time(ms): 0.4259
```

该信息可以在运行log中查看，log位置在`./test_tipc/output/PP-TSM/python_infer_gpu_usetrt_False_precision_fp32_batchsize_2.log`。

如果运行失败，也会在终端中输出运行失败的日志信息以及对应的运行命令。可以基于该命令，分析运行失败的原因。

**注意：** 由于分布式训练时，仅在`trainer_id=0`所在的节点中保存模型，因此其他的节点中在运行模型导出与推理时会因为找不到保存的模型而报错，为正常现象。


# TIPC Linux端Benchmark测试文档

该文档为Benchmark测试说明，Benchmark预测功能测试的主程序为`benchmark_train.sh`，用于验证监控模型训练的性能。

# 1. 测试流程
## 1.1 准备数据和环境安装
运行`test_tipc/prepare.sh`，完成训练数据准备和安装环境流程（以TSM模型为例）。

```shell
# 运行格式：bash test_tipc/prepare.sh  train_benchmark.txt  mode
bash test_tipc/prepare.sh test_tipc/configs/TSM/train_infer_python.txt benchmark_train
```

## 1.2 功能测试
执行`test_tipc/benchmark_train.sh`，完成模型训练和日志解析（以TSM模型为例）。

```shell
# 运行格式：bash test_tipc/benchmark_train.sh train_benchmark.txt mode
bash test_tipc/benchmark_train.sh test_tipc/configs/TSM/train_infer_python.txt benchmark_train

```

`test_tipc/benchmark_train.sh`支持根据传入的第三个配置参数实现只运行某一个训练配置，如下（以TSM模型为例）：
```shell
# 运行格式：bash test_tipc/benchmark_train.sh train_benchmark.txt mode config_pram
## 动态图, batchsize=30, fp32, 数据并行模式, 单机单卡训练配置
bash test_tipc/benchmark_train.sh test_tipc/configs/TSM/train_infer_python.txt benchmark_train dynamic_bs30_fp32_DP_N1C1
## 动态图, batchsize=30, fp16, 数据并行模式, 单机4卡训练配置
bash test_tipc/benchmark_train.sh test_tipc/configs/TSM/train_infer_python.txt benchmark_train dynamic_bs30_fp16_DP_N1C4
# 动转静训练, batchsize=30, fp32, 数据并行模式, 单机单卡训练配置
bash test_tipc/benchmark_train.sh test_tipc/configs/TSM/train_infer_python.txt benchmark_train dynamicTostatic_bs30_fp32_DP_N1C1
```
dynamic_bs30_fp16_DP_N1C4/benchmark_train.sh传入的参数，格式如下：

`${modeltype}_${batch_size}_${fp_item}_${run_mode}_${device_num}`

包含的信息有：模型类型、batchsize大小、训练精度如fp32,fp16等、分布式运行模式以及分布式训练使用的机器信息如单机单卡（N1C1）。


## 2. 日志输出

运行后将会把修改的配置文件临时保存到`test_tipc/benchmark_train.txt`，然后使用该临时文件进行训练与分析，并保存模型的训练日志和解析日志。

如TSM模型某一参数文件的训练日志解析结果是：

```json
{
	"model_branch": "tipc_benchmark",
	"model_commit": "c8f93c7fd9908391371bcccf36a4db4398c49777",
	"model_name": "TSM_bs1_fp16_MultiP_DP",
	"batch_size": 1,
	"fp_item": "fp16",
	"run_process_type": "MultiP",
	"run_mode": "DP",
	"convergence_value": 0,
	"convergence_key": "loss:",
	"ips": 40.237,
	"speed_unit": "instance/sec",
	"device_num": "N1C4",
	"model_run_time": "28",
	"frame_commit": "828f87aecd8a47d19f19f0a83155f8dd340eeaa9",
	"frame_version": "0.0.0"
}
```

训练日志和日志解析结果保存在4个目录下，文件组织格式如下（以TSM模型为例）：
```
PaddleVideo
├── train_log
│   ├── PaddleVideo_TSM_bs1_fp16_MultiP_DP_N1C4_log
│   ├── PaddleVideo_TSM_bs1_fp32_MultiP_DP_N1C4_log
│
├── index
│   ├── PaddleVideo_TSM_bs1_fp16_MultiP_DP_N1C4_speed
│   ├── PaddleVideo_TSM_bs1_fp32_MultiP_DP_N1C4_speed
│
├── profiling_log
│   ├── PaddleVideo_TSM_bs1_fp32_SingleP_DP_N1C1_profiling
│   ├── PaddleVideo_TSM_bs1_fp32_SingleP_DP_N1C1_profiling
│
├── benchmark_log
    └── results.log
```

# PaddleServing预测功能测试

PaddleServing预测功能测试的主程序为`test_serving.sh`，可以测试基于PaddleServing的部署功能。

## 1. 测试结论汇总

基于训练是否使用量化，进行本测试的模型可以分为`正常模型`和`量化模型`(TODO)，这两类模型对应的Serving预测功能汇总如下：

| 模型类型 |device | batchsize | tensorrt | mkldnn | cpu多线程 |
|  ----   |  ---- |   ----   |  :----:  |   :----:   |  :----:  |
| 正常模型 | GPU | 1 | fp32 | - | - |


## 2. 测试流程
运行环境配置请参考[文档](./install.md)的内容配置TIPC的运行环境。
下面以**PP-TSM模型**为例，说明功能测试的过程

### 2.1 功能测试
先运行`prepare.sh`准备数据和模型，然后运行`test_serving.sh`进行测试，最终在`./test_tipc/output/log/PP-TSM/serving_infer_cpp/`目录下生成`serving_infer_*.log`后缀的日志文件。

```bash
bash test_tipc/prepare.sh test_tipc/configs/PP-TSM/serving_infer_cpp.txt serving_infer_cpp

# 用法:
bash test_tipc/test_serving.sh test_tipc/configs/PP-TSM/serving_infer_cpp.txt
```

#### 运行结果

各测试的运行情况会打印在 `./test_tipc/output/log/PP-TSM/serving_infer_cpp/results_serving.log` 中：
运行成功时会输出：

```bash
Run successfully with command - python3.7 -m paddle_serving_server.serve --model ./ppTSM_serving_server/ --port 9993 &!
...
```

运行失败时会输出：

```bash
Run failed with command - python3.7 -m paddle_serving_server.serve --model ./ppTSM_serving_server/ --port 9993 &!
...
```

详细的预测结果会存在 test_tipc/output/ 文件夹下，例如`./test_tipc/output/log/PP-TSM/serving_infer_cpp/server_infer_gpu_batchsize_1.log`中会返回动作分类的结果:

```
{'class_id': '[5]', 'prob': '[0.9907387495040894]'}

```


## 3. 更多教程

本文档为功能测试用，更详细的Serving预测使用教程请参考：[PaddleVideo 服务化部署](../../deploy/cpp_serving/readme.md)

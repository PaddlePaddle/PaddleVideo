简体中文 | [English](../../en/tutorials/deployment.md)

# 推理

## 如何导出一个用于预测的模型？

为了之后的模型预测和部署，我们需要导出模型结构，模型参数，这里应用了PaddlePaddle最新的动转静能力
执行脚本 ```tools.export_model.py```
```python
python3.7 tools/export_model.py -c 配置文件 -o 输出地址 -p 权重文件
```

`export_model.py` 中，首先会重新build一个网络，这里注意，有些用于预测的模型初始化参数可能和训练时不一致，请注意更改。
`export_model.py` 添加了针对TSM的`num_seg`等参数，会用to_static动转静，并调用jit.save来保存预测模型，注意：这里的inputspec需要指定一个`假` 输入来运行网路。

具体原理请参考 [动转静](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/04_dygraph_to_static/index_cn.html) 官方文档。

## 如何检查保存的预测模型正确性？

这里我们提供了```tools/test_export_model.py```脚本用于检查预测模型的正确性。

```python
python3 tools/test_export_model.py -p 权重文件 -i 导出的模型文件夹地址 -c 配置文件
```

`test_export_model.py`只是打印了输出的shape信息，可根据实际需求进行更改，完整的测试流程应该包含下一步：使用预测引擎进行推理

## 如何使用预测引擎进行推理？

这里我们提供了```tools/predict.py``` 进行模型推理。

```python
 python3.7 tools/predict.py -v example.avi --model_file "./inference/example.pdmodel" --params_file "./inference/example.pdiparams" --enable_benchmark=False --model="example" --num_seg=8
 ```
 
 对example.avi进行预测并返回预测结果
 
 ## 如何测试推理速度
 我们提供了统一的测试脚本
 
 ```python
 python3.7 tools/predict.py --enable_benchmark=True --model_file=模型文件 --params_file=参数文件
 ```
 
 ## 如何使用服务器端C++推理?
 
 <sup> coming soon </sup>

 # 部署
 
 ## 如何使用PaddleHub Serving进行部署？
 
 <sup> coming soon </sup>
 
 ## 如何使用PaddleLite进行端上部署？
 
 <sup> coming soon </sup>
 

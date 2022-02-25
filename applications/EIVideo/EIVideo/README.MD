# 交互式视频智能标注工具 - CLI(Command Line Interface)

在开始使用之前，您可能需要按照以下命令安装额外的依赖包：
```bash
python -m pip install scikit-image
```

## 推理运行方式
```shell

C:\Python\Python37\python.exe main.py --test -c E:/PaddlePaddle_Project/EIVideo/resources/backend/configs/manet.yaml -w E:/PaddlePaddle_Project/EIVideo/resources/backend/model/save_step_80000.pdparams
C:\Python\Python37\python.exe resources/backend/main.py --test -c E:/PaddlePaddle_Project/EIVideo/resources/backend/configs/manet.yaml -w E:/PaddlePaddle_Project/EIVideo/resources/backend/model/save_step_80000.pdparams
```
## 参考文档
[manet](docs/zh-CN/manet.md)
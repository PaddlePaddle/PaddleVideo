# paddle2onnx 模型转化与预测

本章节介绍 PP-TSN 模型如何转化为 ONNX 模型，并基于 ONNX 引擎预测。

## 1. 环境准备

需要准备 Paddle2ONNX 模型转化环境，和 ONNX 模型预测环境。

Paddle2ONNX 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式，算子目前稳定支持导出 ONNX Opset 9~11，部分Paddle算子支持更低的ONNX Opset转换。
更多细节可参考 [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/README_zh.md)

- 安装 Paddle2ONNX
```bash
python3.7 -m pip install paddle2onnx
```

- 安装 ONNXRuntime
```bash
# 建议安装 1.9.0 版本，可根据环境更换版本号
python3.7 -m pip install onnxruntime==1.9.0
```

## 2. 模型转换

- PP-TSN inference模型下载

    ```bash
    # 下载inference模型到PaddleVideo/inference/ppTSN/ 目录下
    mkdir -p ./inference
    wget -P ./inference/ https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSN.zip

    # 解压inference模型
    pushd ./inference
    unzip ppTSN.zip
    popd
    ```

- 模型转换

    使用 Paddle2ONNX 将 Paddle inference模型转换为 ONNX 格式模型：

    ```bash
    paddle2onnx \
    --model_dir=./inference/ppTSN \
    --model_filename=ppTSN.pdmodel \
    --params_filename=ppTSN.pdiparams \
    --save_file=./inference/ppTSN/ppTSN.onnx \
    --opset_version=10 \
    --enable_onnx_checker=True
    ```
执行完毕后，可以发现 `./inference/ppTSN` 目录下生成了一个 ONNX 格式的模型文件 `ppTSN.onnx`

## 3. onnx 预测

接下来就可以用 ONNX 格式模型进行预测，其用法与paddle 预测模型类似
执行如下命令：
```bash
python3.7 deploy/paddle2onnx/predict_onnx.py \
--input_file data/example.avi \
--config configs/recognition/pptsn/pptsn_k400_videos.yaml \
--onnx_file=./inference/ppTSN/ppTSN.onnx
```

结果如下：
```bash
Current video file: data/example.avi
        top-1 class: 5
        top-1 score: 0.9998553991317749
```
可以验证该结果与Paddle inference的预测结果完全一致

# paddle2onnx model conversion and prediction

This chapter describes how the PP-TSN model is transformed into an ONNX model and predicted based on the ONNX engine.

## 1. Environment preparation

Need to prepare Paddle2ONNX model conversion environment, and ONNX model prediction environment.

Paddle2ONNX supports converting the PaddlePaddle model format to the ONNX model format. The operator currently supports exporting ONNX Opset 9~11 stably, and some Paddle operators support lower ONNX Opset conversion.
For more details, please refer to [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/README_zh.md)

- Install Paddle2ONNX
```bash
python3.7 -m pip install paddle2onnx
```

- Install ONNXRuntime
```bash
# It is recommended to install version 1.9.0, and the version number can be changed according to the environment
python3.7 -m pip install onnxruntime==1.9.0
```

## 2. Model conversion

- PP-TSN inference model download

    ```bash
    # Download the inference model to the PaddleVideo/inference/ppTSN/ directory
    mkdir -p ./inference
    wget -P ./inference/ https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSN.zip

    # Decompress the inference model
    pushd ./inference
    unzip ppTSN.zip
    popd
    ```

- Model conversion

    Convert Paddle inference models to ONNX format models using Paddle2ONNX:

    ```bash
    paddle2onnx \
    --model_dir=./inference/ppTSN \
    --model_filename=ppTSN.pdmodel \
    --params_filename=ppTSN.pdiparams \
    --save_file=./inference/ppTSN/ppTSN.onnx \
    --opset_version=10 \
    --enable_onnx_checker=True
    ```
After execution, you can find that a model file `ppTSN.onnx` in ONNX format is generated in the `./inference/ppTSN` directory

## 3. onnx prediction

Next, you can use the ONNX format model for prediction, which is similar to the paddle prediction model
Execute the following command:
```bash
python3.7 deploy/paddle2onnx/predict_onnx.py \
--input_file data/example.avi \
--config configs/recognition/pptsn/pptsn_k400_videos.yaml \
--onnx_file=./inference/ppTSN/ppTSN.onnx
```

The result is as follows:
```bash
Current video file: data/example.avi
        top-1 class: 5
        top-1 score: 0.9998553991317749
```
It can be verified that the result is completely consistent with the prediction result of Paddle inference

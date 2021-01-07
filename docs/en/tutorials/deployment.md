[简体中文](../../zh-CN/tutorials/deployment.md) | English

# Inference

## How to convert dygraph model to static model?
To infer and deploy a model, we need export an inference model at first.

```python
    python3 tools/export_model.py -c config_file -o output_path -p params_file
```

Note: In `export_model.py`, It will build a model again, and then loading the prarams. But some init params in the infer phase is different from the train phase.
we add `num_seg` for TSM in advanced, please add more params or modify them if it is necessary.
please refer to [official documents](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/04_dygraph_to_static/index_cn.html) for more information.

## How to test the export model?

PaddleVideo supports a test script to test the exported model.

```python

python3 tools/test_export_model.py -p params_file -i inference_folder -c config_file
```

We just print the output shape, please feel free to extend it. Avtually, only test a video file by PaddleInference can make sure the exported model is right.

## How to use PaddleInference?
PaddleVideo supports ```tools/predict.py``` to infer

```python
 python3.7 tools/predict.py -v example.avi --model_file "./inference/example.pdmodel" --param    s_file "./inference/example.pdiparams" --enable_benchmark=False --model="example" --num_seg=8
 ```

## How to test inference speed?


# Deployment

## How to release and deploy a model?
<sup> coming soon</sup>

## How to use PaddleHub?
<sup> coming soon</sup>

## How to use PaddleServing?
<sup> coming soon</sup>

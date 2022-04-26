## Slim function introduction
A complex model is beneficial to improve the performance of the model, but it also leads to some redundancy in the model. This part provides the function of reducing the model, including two parts: model quantization (quantization training, offline quantization), model pruning.

Among them, model quantization reduces the full precision to fixed-point numbers to reduce this redundancy, so as to reduce the computational complexity of the model and improve the inference performance of the model.
Model quantization can convert FP32-precision model parameters to Int8-precision without losing the accuracy of the model, reducing the size of model parameters and speeding up the calculation. Using the quantized model has a speed advantage when deploying on mobile terminals.

Model pruning cuts out the unimportant convolution kernels in the CNN, reduces the amount of model parameters, and thus reduces the computational complexity of the model.

This tutorial will introduce how to use PaddleSlim, a paddle model compression library, to compress PaddleVideo models.
[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim) integrates model pruning, quantization (including quantization training and offline quantization), distillation and neural network search and other commonly used and leading model compression functions in the industry. If you are interested, you can follow and understand.

Before starting this tutorial, it is recommended to understand [PaddleVideo model training method](../../docs/zh-CN/usage.md) and [PaddleSlim](https://paddleslim.readthedocs.io/zh_CN/ latest/index.html)


## quick start
After training a model, if you want to further compress the model size and speed up prediction, you can use quantization or pruning to compress the model.

Model compression mainly includes five steps:
1. Install PaddleSlim
2. Prepare the trained model
3. Model Compression
4. Export the quantitative inference model
5. Quantitative Model Prediction Deployment

### 1. Install PaddleSlim

* It can be installed by pip install.

```bash
python3.7 -m pip install paddleslim -i https://pypi.tuna.tsinghua.edu.cn/simple
```

* If you get the latest features of PaddleSlim, you can install it from source.

```bash
git clone https://github.com/PaddlePaddle/PaddleSlim.git
cd Paddleslim
python3.7 setup.py install
```

### 2. Prepare the trained model

PaddleVideo provides a series of trained [models](../../docs/zh-CN/model_zoo/README.md). If the model to be quantized is not in the list, you need to follow the [regular training](../ ../docs/zh-CN/usage.md) method to get the trained model.

### 3. Model Compression

Go to PaddleVideo root directory

```bash
cd PaddleVideo
```

The offline quantization code is located in `deploy/slim/quant_post_static.py`.

#### 3.1 Model Quantization

Quantization training includes offline quantization training and online quantization training (TODO). The effect of online quantization training is better. The pre-training model needs to be loaded, and the model can be quantized after the quantization strategy is defined.

##### 3.1.1 Online quantitative training
TODO

##### 3.1.2 Offline Quantization

**Note**: For offline quantization, you must use the `inference model` exported from the trained model for quantization. For general model export `inference model`, please refer to [Tutorial](../../docs/zh-CN/usage.md#5-Model Inference).

Generally speaking, the offline quantization loss model has more accuracy.

Taking the PP-TSM model as an example, after generating the `inference model`, the offline quantization operation is as follows

```bash
# download a small amount of data for calibration
pushd ./data/k400
wget -nc https://videotag.bj.bcebos.com/Data/k400_rawframes_small.tar
tar -xf k400_rawframes_small.tar
popd

# then switch to deploy/slim
cd deploy/slim

# execute quantization script
python3.7 quant_post_static.py \
-c ../../configs/recognition/pptsm/pptsm_k400_frames_uniform_quantization.yaml \
--use_gpu=True
```

All quantization environment parameters except `use_gpu` are configured in `pptsm_k400_frames_uniform_quantization.yaml` file
Where `inference_model_dir` represents the directory path of the `inference model` exported in the previous step, and `quant_output_dir` represents the output directory path of the quantization model

After successful execution, the `__model__` file and the `__params__` file are generated in the `quant_output_dir` directory, which are used to store the generated offline quantization model
Similar to the usage of `inference model`, you can directly use these two files for prediction deployment without re-exporting the model.

```bash
# Use PP-TSM offline quantization model for prediction
# Go back to the PaddleVideo directory
cd ../../

# Use the quantized model to make predictions
python3.7 tools/predict.py \
--input_file data/example.avi \
--config configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml \
--model_file ./inference/ppTSM/quant_model/__model__ \
--params_file ./inference/ppTSM/quant_model/__params__ \
--use_gpu=True \
--use_tensorrt=False
```

The output is as follows:
```bash
Current video file: data/example.avi
        top-1 class: 5
        top-1 score: 0.9997928738594055
```
#### 3.2 Model pruning
TODO


### 4. Export the model
TODO


### 5. Model Deployment

The model exported in the above steps can be converted through the opt model conversion tool of PaddleLite.
Reference for model deployment
[Serving Python Deployment](../python_serving/readme.md)
[Serving C++ Deployment](../cpp_serving/readme.md)


## Training hyperparameter suggestions

* During quantitative training, it is recommended to load the pre-trained model obtained from regular training to accelerate the convergence of quantitative training.
* During quantitative training, it is recommended to modify the initial learning rate to `1/20~1/10` of conventional training, and modify the number of training epochs to `1/5~1/2` of conventional training. In terms of learning rate strategy, add On Warmup, other configuration information is not recommended to be modified.

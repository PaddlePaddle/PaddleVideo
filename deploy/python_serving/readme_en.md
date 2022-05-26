English | [简体中文](./readme.md)
# Model service deployment

## Introduction

[Paddle Serving](https://github.com/PaddlePaddle/Serving) aims to help deep learning developers easily deploy online prediction services, support one-click deployment of industrial-grade service capabilities, high concurrency between client and server Efficient communication and support for developing clients in multiple programming languages.

This section takes the HTTP prediction service deployment as an example to introduce how to use PaddleServing to deploy the model service in PaddleVideo. Currently, only Linux platform deployment is supported, and Windows platform is not currently supported.

## Serving installation
The Serving official website recommends using docker to install and deploy the Serving environment. First, you need to pull the docker environment and create a Serving-based docker.

```bash
# start GPU docker
docker pull paddlepaddle/serving:0.7.0-cuda10.2-cudnn7-devel
nvidia-docker run -p 9292:9292 --name test -dit paddlepaddle/serving:0.7.0-cuda10.2-cudnn7-devel bash
nvidia-docker exec -it test bash

# start CPU docker
docker pull paddlepaddle/serving:0.7.0-devel
docker run -p 9292:9292 --name test -dit paddlepaddle/serving:0.7.0-devel bash
docker exec -it test bash
```

After entering docker, you need to install Serving-related python packages.
```bash
python3.7 -m pip install paddle-serving-client==0.7.0
python3.7 -m pip install paddle-serving-app==0.7.0
python3.7 -m pip install faiss-cpu==1.7.1post2

#If it is a CPU deployment environment:
python3.7 -m pip install paddle-serving-server==0.7.0 #CPU
python3.7 -m pip install paddlepaddle==2.2.0 # CPU

#If it is a GPU deployment environment
python3.7 -m pip install paddle-serving-server-gpu==0.7.0.post102 # GPU with CUDA10.2 + TensorRT6
python3.7 -m pip install paddlepaddle-gpu==2.2.0 # GPU with CUDA10.2

#Other GPU environments need to confirm the environment and then choose which one to execute
python3.7 -m pip install paddle-serving-server-gpu==0.7.0.post101 # GPU with CUDA10.1 + TensorRT6
python3.7 -m pip install paddle-serving-server-gpu==0.7.0.post112 # GPU with CUDA11.2 + TensorRT8
```

* If the installation speed is too slow, you can change the source through `-i https://pypi.tuna.tsinghua.edu.cn/simple` to speed up the installation process
* For more environment and corresponding installation packages, see: https://github.com/PaddlePaddle/Serving/blob/v0.9.0/doc/Install_Linux_Env_CN.md

## Behavior recognition service deployment
### Model conversion
When using PaddleServing for service deployment, you need to convert the saved inference model into a Serving model. The following uses the PP-TSM model as an example to introduce how to deploy the behavior recognition service.
- Download the trained PP-TSM model and convert it into an inference model:
  ```bash
  # Enter PaddleVideo directory
  cd PaddleVideo

  wget -P data/ https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_uniform.pdparams

  python3.7 tools/export_model.py \
  -c configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml \
  -p data/ppTSM_k400_uniform.pdparams \
  -o inference/ppTSM
  ```

- We also provide the converted inference model, download and unzip by the following command
  ```bash
  mkdir ./inference
  wget -nc -P ./inference https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM.zip --no-check-certificate
  pushd ./inference
  unzip ppTSM.zip
  popd
  ```
- Use paddle_serving_client to convert the converted inference model into a model format that is easy for server deployment:
  ```bash
  python3.7 -m paddle_serving_client.convert \
  --dirname inference/ppTSM \
  --model_filename ppTSM.pdmodel \
  --params_filename ppTSM.pdiparams \
  --serving_server ./deploy/python_serving/ppTSM_serving_server/ \
  --serving_client ./deploy/python_serving/ppTSM_serving_client/
  ```
  | parameter | type | default value | description |
  | ----------------- | ---- | ------------------ | ------- -------------------------------------------------- --- |
  | `dirname` | str | - | The storage path of the model file to be converted. The program structure file and parameter file are saved in this directory. |
  | `model_filename` | str | None | The name of the file storing the model Inference Program structure that needs to be converted. If set to None, use `__model__` as the default filename |
  | `params_filename` | str | None | File name where all parameters of the model to be converted are stored. It needs to be specified if and only if all model parameters are stored in a single binary file. If the model parameters are stored in separate files, set it to None |
  | `serving_server` | str | `"serving_server"` | The storage path of the converted model files and configuration files. Default is serving_server |
  | `serving_client` | str | `"serving_client"` | The converted client configuration file storage path. Default is serving_client |

After the PP-TSM inference model is converted, there will be additional folders of `ppTSM_serving_server` and `ppTSM_serving_client` in the current folder, with the following formats:
  ```bash
  PaddleVideo/deploy/python_serving
  ├── ppTSM_serving_server
      ├── ppTSM.pdiparams
      ├── ppTSM.pdmodel
      ├── serving_server_conf.prototxt
      └── serving_server_conf.stream.prototxt
  ├── ppTSM_serving_client
      ├── serving_client_conf.prototxt
      └── serving_client_conf.stream.prototxt
  ```
After getting the model files, you need to modify the files `serving_server_conf.prototxt` under `ppTSM_serving_server` and `ppTSM_serving_client` respectively, and change `alias_name` under `fetch_var` in both files to `outputs`

**Remarks**: In order to be compatible with the deployment of different models, Serving provides the function of input and output renaming. In this way, when different models are inferred and deployed, they only need to modify the `alias_name` of the configuration file, and the inference deployment can be completed without modifying the code.
The modified `serving_server_conf.prototxt` looks like this:

```yaml
feed_var {
  name: "data_batch_0"
  alias_name: "data_batch_0"
  is_lod_tensor: false
  feed_type: 1
  shape: 8
  shape: 3
  shape: 224
  shape: 224
}
fetch_var {
  name: "linear_2.tmp_1"
  alias_name: "outputs"
  is_lod_tensor: false
  fetch_type: 1
  shape: 400
}

```
### Service deployment and requests
The `python_serving` directory contains the code for starting the pipeline service, C++ serving service (TODO) and sending prediction requests, including:
```bash
__init__.py
configs/xxx.yaml            # start the configuration file of the pipeline service
pipeline_http_client.py     # python script for sending pipeline prediction request via http
pipeline_rpc_client.py      # python script for sending pipeline prediction request in rpc mode
recognition_web_service.py  # python script that starts the pipeline server
utils.py                    # common functions used in inference, such as parse_file_paths, numpy_to_base64, video_to_numpy
```
#### Python Serving
- Go to the working directory:
```bash
cd deploy/python_serving
```

- Start the service:
```bash
# Start in the current command line window and stay in front
python3.7 recognition_web_service.py -n PPTSM -c configs/PP-TSM.yaml
# Start in the background, the logs printed during the process will be redirected and saved to log.txt
python3.7 recognition_web_service.py -n PPTSM -c configs/PP-TSM.yaml &>log.txt &
```

- send request:
```bash
# Send a prediction request in http and receive the result
python3.7 pipeline_http_client.py -i ../../data/example.avi

# Send a prediction request in rpc and receive the result
python3.7 pipeline_rpc_client.py -i ../../data/example.avi
```
After a successful run, the results of the model prediction will be printed in the cmd window, and the results are as follows:

```bash
# http method print result
{'err_no': 0, 'err_msg': '', 'key': ['label', 'prob'], 'value': ["['archery']", '[0.9907388687133789]'], 'tensors ': []}

# The result of printing in rpc mode
PipelineClient::predict pack_data time:1645631086.764019
PipelineClient::predict before time:1645631086.8485317
key: "label"
key: "prob"
value: "[\'archery\']"
value: "[0.9907388687133789]"
```

## FAQ
**Q1**: No result is returned after the request is sent or an output decoding error is prompted

**A1**: Do not set the proxy when starting the service and sending the request. You can close the proxy before starting the service and sending the request. The command to close the proxy is:
```
unset https_proxy
unset http_proxy
```

**Q2**: There is no response after the server is started, and it has been stopped at `start proxy service`

**A2**: It is likely that a problem was encountered during the startup process. You can view the detailed error message in the `./deploy/python_serving/PipelineServingLogs/pipeline.log` log file

For more service deployment types, such as `RPC prediction service`, you can refer to Serving's [github official website](https://github.com/PaddlePaddle/Serving/tree/v0.7.0/examples)

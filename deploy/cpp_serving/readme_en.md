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

* If the installation speed is too slow, you can change the source through `-i https://pypi.tuna.tsinghua.edu.cn/simple` to speed up the installation process.

## Action recognition service deployment
### Model conversion
When using PaddleServing for service deployment, you need to convert the saved inference model into a Serving model. The following takes the PP-TSM model as an example to introduce how to deploy the image classification service.
- Download PP-TSM inference model and convert to Serving model:
  ```bash
  # Download the inference model and extract it to ./inference
  mkdir ./inference
  pushd ./inference
  wget https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM.zip
  unzip ppTSM.zip
  popd

  # Convert to Serving model
  pushd deploy/cpp_serving
  python3.7 -m paddle_serving_client.convert \
  --dirname ../../inference/ppTSM \
  --model_filename ppTSM.pdmodel \
  --params_filename ppTSM.pdiparams \
  --serving_server ./ppTSM_serving_server \
  --serving_client ./ppTSM_serving_client
  popd
  ```

After the PP-TSM inference model conversion is completed, two additional folders, `ppTSM_serving_client` and `ppTSM_serving_server`, will be added to the `deploy/cpp_serving` folder, with the following formats:
  ```bash
  PaddleVideo
  └── deploy
      ├── ppTSM_serving_client
      │ ├── serving_client_conf.prototxt
      │ └── serving_client_conf.stream.prototxt
      └── ppTSM_serving_server
          ├── ppTSM.pdiparams
          ├── ppTSM.pdmodel
          ├── serving_server_conf.prototxt
          └── serving_server_conf.stream.prototxt
  ```
After getting the model file, you need to modify `serving_client_conf.prototxt` under `ppTSM_serving_client` and `serving_server_conf.prototxt` under `ppTSM_serving_server` respectively, and change `alias_name` under `fetch_var` in both files to `outputs`

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
The cpp_serving directory contains the code for starting the pipeline service, the C++ serving service and sending the prediction request, including:
```bash
run_cpp_serving.sh # Start the script on the C++ serving server side
pipeline_http_client.py # The script on the client side to send data and get the prediction results
```
#### C++ Serving
- Go to the working directory:
```bash
cd deploy/cpp_serving
```

- Start the service:
```bash
# Start in the background, the logs printed during the process will be redirected and saved to nohup.txt
bash run_cpp_serving.sh
```
**If an error is reported during the process and it shows that libnvinfer.so.6 cannot be found, you can execute the script `paddle_env_install.sh` to install the relevant environment**
```bash
bash paddle_env_install.sh
```

- Send the request and get the result:
```bash
python3.7 serving_client.py \
-n PPTSM \
-c ./ppTSM_serving_client/serving_client_conf.prototxt \
--input_file=../../data/example.avi
```
After a successful run, the results of the model prediction will be printed in the cmd window, and the results are as follows:

```bash
I0510 04:33:00.110025 37097 naming_service_thread.cpp:202] brpc::policy::ListNamingService("127.0.0.1:9993"): added 1
I0510 04:33:01.904764 37097 general_model.cpp:490] [client]logid=0,client_cost=1640.96ms,server_cost=1623.21ms.
{'class_id': '[5]', 'prob': '[0.9907387495040894]'}
```


## FAQ
**Q1**: No result is returned after the request is sent or an output decoding error is prompted

**A1**: Do not set the proxy when starting the service and sending the request. You can close the proxy before starting the service and sending the request. The command to close the proxy is:
```
unset https_proxy
unset http_proxy
```

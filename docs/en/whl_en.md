# ppvideo package

## Get started quickly

### install package

install by pypi
```bash
pip install paddlevideo==0.0.1
```

build own whl package and install
```bash
python3 setup.py bdist_wheel
pip3 install dist/paddlevideo-x.x.x-py3-none-any.whl
```

### 1. Quick Start

* Assign `video_file='data/example.mp4'`, Use inference model that Paddle provides `model_name='ppTSM'`


```python
from ppvideo import PaddleVideo
clas = PaddleVideo(model_name='ppTSM',use_gpu=False,use_tensorrt=False)
video_file='data/example.mp4.'
result=clas.predict(video_file)
print(result)
```

```
    >>> result
    [{'filename': 'data/example.mp4', 'class_ids': [5], 'scores': [0.999963], 'label_names': ['archery']}]
```

### 2. Definition of Parameters
* model_name(str): model's name. If not assigning `model_file`and`params_file`, you can assign this param. If using inference model based on Kinectics-400 provided by Paddle, set as default='ppTSM'.
* video_file(str): video's path. Support assigning single local image, internet image and folder containing series of images. Also Support numpy.ndarray.
* use_gpu(bool): Whether to use GPU or not, defalut=False.
* num_seg(int): Number of segments while using the sample strategies proposed in TSN.
* seg_len(int): Number of frames for each segment.
* short_size(int): resize the minima between height and width into resize_short(int), default=256.
* target_size(int): resize image into resize(int), default=224.
* normalize(bool): whether normalize image or not, default=True.
* model_file(str): path of inference.pdmodel. If not assign this param，you need assign `model_name` for downloading.
* params_file(str): path of inference.pdiparams. If not assign this param，you need assign `model_name` for downloading.
* batch_size(int): batch number, default=1.
* use_fp16(bool): Whether to use float16 in memory or not, default=False.
* ir_optim(bool): whether enable IR optimization or not, default=True.
* use_tensorrt(bool): whether to open tensorrt or not. Using it can greatly promote predict preformance, default=False.
* gpu_mem(int): GPU memory usages，default=8000.
* top_k(int): Assign top_k, default=1.
* enable_mkldnn(bool): whether enable MKLDNN or not, default=False.


### 3. Different Usages of Codes

**We provide two ways to use: 1. Python interative programming 2. Bash command line programming**

* check `help` information
```bash
ppvideo -h
```

* Use user-specified model, you need to assign model's path `model_file` and parameters's path`params_file`

###### python
```python
from ppvideo import PaddleVideo
clas = PaddleVideo(model_file='user-specified model path',
    params_file='parmas path', use_gpu=False, use_tensorrt=False)
video_file = ''
result=clas.predict(video_file)
print(result)
```

* Use inference model which PaddlePaddle provides to predict, you need to choose one of model when initializing ppvideo to assign `model_name`. You may not assign `model_file` , and the model you chosen will be download in `BASE_INFERENCE_MODEL_DIR` ,which will be saved in folder named by `model_name`,avoiding overlay different inference model.

###### python
```python
from ppvideo import PaddleVideo
clas = PaddleVideo(model_name='ppTSM',use_gpu=False, use_tensorrt=False)
video_file = ''
result=clas.predict(video_file)
print(result)
```

* You can assign input as format`np.ndarray` which has been preprocessed `--image_file=np.ndarray`.

###### python
```python
from ppvideo import PaddleVideo
clas = PaddleVideo(model_name='ppTSM',use_gpu=False, use_tensorrt=False)
image_file =np.ndarray
result=clas.predict(image_file)
```

* You can assign `image_file` as a folder path containing series of images, also can assign `top_k`.

###### python
```python
from ppvideo import PaddleVideo
clas = PaddleVideo(model_name='ppTSM',use_gpu=False, use_tensorrt=False,top_k=5)
image_file = '' # it can be image_file folder path which contains all of images you want to predict.
result=clas.predict(image_file)
print(result)
```

* You can assign `--pre_label_image=True`, `--pre_label_out_idr= './output_pre_label/'`.Then images will be copied into folder named by top-1 class_id.

###### python
```python
from ppvideo import PaddleVideo
clas = PaddleVideo(model_name='ppTSM',use_gpu=False, use_tensorrt=False,top_k=5, pre_label_image=True,pre_label_out_idr='./output_pre_label/')
image_file = '' # it can be image_file folder path which contains all of images you want to predict.
result=clas.predict(image_file)
print(result)
```

* You can assign `--label_name_path` as your own label_dict_file, format should be as(class_id<space>class_name<\n>).

```
0 tench, Tinca tinca
1 goldfish, Carassius auratus
2 great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias
......
```

* If you use inference model that Paddle provides, you do not need assign `label_name_path`. Program will take `ppcls/utils/imagenet1k_label_list.txt` as defaults. If you hope using your own training model, you can provide `label_name_path` outputing 'label_name' and scores, otherwise no 'label_name' in output information.

###### python
```python
from ppvideo import PaddleVideo
clas = PaddleVideo(model_file= './inference.pdmodel',params_file = './inference.pdiparams',label_name_path='./ppcls/utils/imagenet1k_label_list.txt',use_gpu=False)
image_file = '' # it can be image_file folder path which contains all of images you want to predict.
result=clas.predict(image_file)
print(result)
```

###### python
```python
from ppvideo import PaddleVideo
clas = PaddleVideo(model_name='ppTSM',use_gpu=False)
image_file = '' # it can be image_file folder path which contains all of images you want to predict.
result=clas.predict(image_file)
print(result)
```

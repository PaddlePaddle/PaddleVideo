# Start
---

Please refer to [installation documents](./install.md) to prepare the enviroment, and follow the steps mentioned in the [data preparation documents](./data/) to construct dataset, we will take you through the basic functions supported by PaddleVideo, all of it takes the ucf101 dataset as example.

PaddleVideo only support linux operation system and GPU running time environment

<a name="1"></a>
## 1. Train and Test

Start running multi-cards training scripts or test scripts by `paddle.distributed.launch`, or run the `run.sh` directly.

```shell
sh run.sh
```

<a name="model_train"></a>
### 1.1 Train

Switch `--validate` on to validating while training.

```bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    main.py \
        --validate \
        -c ./configs/example.yaml
```

Indicating `-c` to set configuration, and one can flexible add `-o` in the script to update it.

```bash
python -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    main.py \
        -c ./configs/example.yaml \
        --validate \
        -o DATASET.batch_size=16
```
Indicating `-o DATASET.batch_size=16` can update batch size to 16, please refer to [configuration](./config.md) for more information.

After starting training, log file will generated, and its format is shown as below, it will output to both the screen and files. Default destination of log is under the `.log` folder, and stored in the files named like `worker.0`, `worker.1` ...

[train phase]:

    
    [12/28 17:31:26] epoch:[ 1/80 ] train step:0   loss: 0.04656 lr: 0.000100 top1: 1.00000 top5: 1.00000 elapse: 0.326 reader: 0.001s ips: 98.22489 instance/sec.
    
    
[eval phase]
    
    [12/28 17:31:32] epoch:[ 80/80 ] val step:0    loss: 0.20538 top1: 0.88281 top5: 0.99219 elapse: 1.589 reader: 0.000s ips: 20.14003 instance/sec.
    
    
[epoch end]
    
    [12/28 17:31:38] END epoch:80  val loss_avg: 0.52208 top1_avg: 0.84398 top5_avg: 0.97393 elapse_avg: 0.234 reader_avg: 0.000 elapse_sum: 7.021s ips: 136.73686 instance/sec.
    
    
[the best Acc]    
    [12/28 17:28:42] Already save the best model (top1 acc)0.8494
    

<a name="model_resume"></a>
### 1.2 resume

Indicate `-o resume_epoch` to resume, It will training from ```resume_epoch``` epoch, PaddleVideo will auto load optimizers parameters and checkpoints from `./output` folder, as it is the default output destination.

```
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    main.py \
        -c ./configs/example.yaml \
        --validate \
        -o resume_epoch=5

```


<a name="model_finetune"></a>
### 1.3 Finetune

Indicate `--weights` to load pretrained parameters, PaddleVideo will auto treat it as a finetune mission.
```
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    main.py \
        -c ./configs/example.yaml \
        --validate \
        --weights=path_to_weights
```

Note: PaddleVideo will NOT load shape unmatched parameters.

<a name="model_test"></a>
### 1.4 Test

Switch `--test` on to start test mode, and indicate `--weights` to load pretrained model.

```bash
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    main.py \
        -c ./configs/example.yaml \
        --test \
        --weights=path_to_weights
```



<a name="model_inference"></a>
## 2. Infer

First, export model.
Indicate `-c` to set configuration, `-p` to load pretrained model, `-o` to set outputs destination.

```bash
python tools/export_model.py \
    -c ./configs/example.yaml \
    -p ./output/path_to_weights \
    -o ./inference
```


It will generate `model_name.pdmodel` and `model_name.pdiparams`.
Second, start PaddleInference engine to infer a video.

```bash
python tools/infer/predict.py \
    --video_file \
    --model_file "./inference/example.pdmodel" \
    --params_file "./inference/example.pdiparams" \
    --use_gpu=True \
    --use_tensorrt=False
```

Attributes:
+ `video_file`: video file path.
+ `model_file`: pdmodel file path.
+ `params_file`: pdiparams file path.
+ `use_tensorrt`: use tensorrt to acclerate or not, default: True.
+ `use_gpu`: use gpu to infer or not, default: True.

benchmark results are shown in th [benchmark](./benchmark.md).

# 提供可稳定复现性能的脚本，默认在标准docker环境内py37执行： paddlepaddle/paddle:latest-gpu-cuda10.2-cudnn7  paddle=2.1.2  py=37
# 执行目录：需说明
sed -i '/set\ -xe/d' run_benchmark.sh
cd ../../ # cd到PaddleVideo项目根目录下
git checkout benchmark_dev
log_path=${LOG_PATH_INDEX_DIR:-$(pwd)}  #  benchmark系统指定该参数,不需要跑profile时,log_path指向存speed的目录

# 1 安装该模型需要的依赖 (如需开启优化策略请注明)
python -m pip install -r requirements.txt

# 2 拷贝该模型需要数据、预训练模型
unalias cp
cp -f benchmark/TimeSformer/timesformer_ucf101_videos_benchmark_bs1.yaml configs/recognition/timesformer/
cp -f benchmark/TimeSformer/timesformer_ucf101_videos_benchmark_bs1_mp.yaml configs/recognition/timesformer/
cp -f benchmark/TimeSformer/timesformer_ucf101_videos_benchmark_bs14.yaml configs/recognition/timesformer/
cp -f benchmark/TimeSformer/timesformer_ucf101_videos_benchmark_bs14_mp.yaml configs/recognition/timesformer/
if [ ! -f "data/ucf101/trainlist_benchmark_mp.txt" ]; then
    wget -P data/ucf101/ https://videotag.bj.bcebos.com/PaddleVideo-release2.2/trainlist_benchmark_mp.txt
fi
wget -P data/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams
alias cp='cp -i'

cd data/ucf101 # 进入PaddleVideo/data/ucf101
if [ $1 = "down_data" ];then
    wget --no-check-certificate "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar" # 下载训练数据
    unrar x UCF101.rar # 解压
    mv ./UCF-101 ./videos # 重命名文件夹为./videos
    rm -rf ./UCF101.rar
else    # 使用本地数据
    rm -rf videos
    ln -s ${data_path}/dygraph_data/TSM/ucf101/videos ./videos
fi
cd ../../ # 返回PaddleVideo

# 3 批量运行（如不方便批量，1，2需放到单个模型中）

model_mode_list=(TimeSformer)
fp_item_list=(fp32 fp16)
bs_item_list=(1)    #  14
for model_mode in ${model_mode_list[@]}; do
      for fp_item in ${fp_item_list[@]}; do
          for bs_item in ${bs_item_list[@]}
            do
            run_mode=sp
            log_name=video_${model_mode}_${run_mode}_bs${bs_item}_${fp_item}   # 如:clas_MobileNetv1_mp_bs32_fp32_8
            echo "index is speed, 1gpus, begin, ${log_name}"
            CUDA_VISIBLE_DEVICES=0 bash benchmark/${model_mode}/run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} ${model_mode} | tee ${log_path}/${log_name}_speed_1gpus 2>&1
            sleep 60

            run_mode=mp
            log_name=video_${model_mode}_${run_mode}_bs${bs_item}_${fp_item}   # 如:clas_MobileNetv1_mp_bs32_fp32_8
            echo "index is speed, 8gpus, run_mode is multi_process, begin, ${log_name}"
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash benchmark/${model_mode}/run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} ${model_mode} | tee ${log_path}/${log_name}_speed_8gpus8p 2>&1
            sleep 60
            done
      done
done

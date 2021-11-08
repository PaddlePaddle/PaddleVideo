# 提供可稳定复现性能的脚本，默认在标准docker环境内py37执行： paddlepaddle/paddle:latest-gpu-cuda10.2-cudnn7  paddle=2.1.2  py=37

# 执行目录：需说明
cd ../../ # cd到PaddleVideo项目根目录下
git checkout benchmark_dev

# 1 安装该模型需要的依赖 (如需开启优化策略请注明)
python3.7 -m pip install -r requirements.txt

# 2 拷贝该模型需要数据、预训练模型
unalias cp
cp -f benchmark/TimeSformer/timesformer_ucf101_videos_benchmark_bs1.yaml configs/recognition/timesformer/
cp -f benchmark/TimeSformer/timesformer_ucf101_videos_benchmark_bs1_mp.yaml configs/recognition/timesformer/
cp -f benchmark/TimeSformer/timesformer_ucf101_videos_benchmark_bs14.yaml configs/recognition/timesformer/
cp -f benchmark/TimeSformer/timesformer_ucf101_videos_benchmark_bs14_mp.yaml configs/recognition/timesformer/
if [ ! -f "data/ucf101/trainlist_benchmark_mp.txt" ]; then
    wget -P data/ucf101/ https://videotag.bj.bcebos.com/PaddleVideo-release2.2/trainlist_benchmark_mp.txt
fi
wget -P data/ wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams
alias cp='cp -i'

cd data/ucf101 # 进入PaddleVideo/data/ucf101
wget --no-check-certificate "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar" # 下载训练数据
unrar x UCF101.rar # 解压
mv ./UCF-101 ./videos # 重命名文件夹为./videos
rm -rf ./UCF101.rar
cd ../../ # 返回PaddleVideo

# 3 批量运行（如不方便批量，1，2需放到单个模型中）

model_mode_list=(TimeSformer)
fp_item_list=(fp32 fp16)
bs_item_list=(1 14)
for model_mode in ${model_mode_list[@]}; do
      for fp_item in ${fp_item_list[@]}; do
          for bs_item in ${bs_item_list[@]}
            do
            echo "index is speed, 1gpus, begin, ${model_name}"
            run_mode=sp
            CUDA_VISIBLE_DEVICES=0 bash benchmark/${model_mode}/run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} ${model_mode}
            sleep 60
            echo "index is speed, 8gpus, run_mode is multi_process, begin, ${model_name}"
            run_mode=mp
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash benchmark/${model_mode}/run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} ${model_mode} 
            sleep 60
            done
      done
done

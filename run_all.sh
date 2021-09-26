# 提供可稳定复现性能的脚本，默认在标准docker环境内py37执行： paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7  paddle=2.1.2  py=37
# 执行目录：需说明
# git clone https://github.com/HydrogenSulfate/PaddleVideo.git # 克隆修改过的repo
# cd PaddleVideo # 进入目录
# git checkout dev_benchmark # 切换为benchmark专用分支
# 1 安装该模型需要的依赖 (如需开启优化策略请注明)
# python3.7 -m pip install pip==21.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/
# python3.7 -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
# python3.7 -m pip install pathlib==1.0.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/
# python3.7 -m pip install Pillow==8.0.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/
# python3.7 -m pip install six==1.15.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
# python3.7 -m pip install scipy==1.5.3 -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 2 拷贝该模型需要数据、预训练模型
# 本模型不用加载预训练模型也能训练，且不影响速度
cd ./data/ucf101 # 进入数据目标存放目录
# bash download_videos.sh # 下载并解压到 data/ucf101/videos里
cd .. # 返回
cd .. # 返回

# 3 批量运行（如不方便批量，1，2需放到单个模型中）
 
model_mode_list=(timesformer)
fp_item_list=(fp32 fp16)
bs_list=(1 7)
for model_mode in ${model_mode_list[@]}; do
      for fp_item in ${fp_item_list[@]}; do
          for bs_item in ${bs_list[@]}
            do
            echo "index is speed, 1gpus, begin, ${model_name}"
            run_mode=sp
            num_gpu_devices=1
            CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} ${model_mode}     #  (5min)
            # CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 14 fp32 timesformer
            # sleep 60
            python3.7 ./scripts/analysis.py \
                    --filename ${model_mode}_${run_mode}_bs${bs_item}_${fp_item}_${num_gpu_devices} \
                    --keyword "ips:" \
                    --model_name video_${model_mode}_bs${bs_item}_${fp_item} \
                    --mission_name "action recognition" \
                    --direction_id 0 \
                    --run_mode ${run_mode} \
                    --gpu_num ${num_gpu_devices} \
                    --index 1
                    # --log_with_profiler ${log_with_profiler:-"not found!"} \
                    # --profiler_path ${profiler_path:-"not found!"} \

            echo "index is speed, 8gpus, run_mode is multi_process, begin, ${model_name}"
            run_mode=mp
            CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} ${model_mode}  # 开发机4卡，上传时记得改为8卡
            # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 1 fp32 timesformer
            # sleep 60
            num_gpu_devices=4
            python3.7 ./scripts/analysis.py \
                    --filename ${model_mode}_${run_mode}_bs${bs_item}_${fp_item}_${num_gpu_devices} \
                    --keyword "ips:" \
                    --model_name video_${model_mode}_bs${bs_item}_${fp_item} \
                    --mission_name "action recognition" \
                    --direction_id 0 \
                    --run_mode ${run_mode} \
                    --gpu_num ${num_gpu_devices} \
                    --index 1
                    # --log_with_profiler ${log_with_profiler:-"not found!"} \
                    # --profiler_path ${profiler_path:-"not found!"} \
            done
      done
done
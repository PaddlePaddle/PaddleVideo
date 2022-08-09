batch_size_list=(30)
precision_list=(fp16_o2) # fp16_o1 tf32 fp32)
opt_level_list=(4)
# 1: base 2: num= 8 3:auto_tune 4: workspace  5:old prof
#data_format=(NCHW)
data_format=(NHWC)
export FLAGS_cudnn_batchnorm_spatial_persistent=1
for batch_size in ${batch_size_list[@]}; do
    for precision in ${precision_list[@]}; do
        for opt_level in ${opt_level_list[@]}; do
            bash run_TSM_bs30_fp32.sh ${batch_size} ${precision} ${opt_level} ${data_format}
            sleep 60
        done
    done
done


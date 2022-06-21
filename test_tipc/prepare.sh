#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1

# set -xe

:<<!
MODE be one of ['lite_train_lite_infer' 'lite_train_whole_infer' 'whole_train_whole_infer',
#                 'whole_infer',
#                 'cpp_infer', ]
!

MODE=$2

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})

# determine python interpreter version
python=$(func_parser_value "${lines[2]}")

# install auto-log package.
${python} -m pip install unrar
${python} -m pip install https://paddleocr.bj.bcebos.com/libs/auto_log-1.2.0-py3-none-any.whl

# The training params
model_name=$(func_parser_value "${lines[1]}")

trainer_list=$(func_parser_value "${lines[14]}")




if [ ${MODE} = "lite_train_lite_infer" ];then
    if [ ${model_name} == "PP-TSM" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_rawframes_small.tar
        tar -xf k400_rawframes_small.tar
        popd
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams --no-check-certificate
    elif [ ${model_name} == "PP-TSN" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_videos_small.tar
        tar -xf k400_videos_small.tar
        popd
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams --no-check-certificate
    elif [ ${model_name} == "AGCN" ]; then
        # pretrain lite train data
        pushd data/fsd10
        wget -nc https://videotag.bj.bcebos.com/Data/FSD_train_data.npy
        wget -nc https://videotag.bj.bcebos.com/Data/FSD_train_label.npy
        popd
    elif [ ${model_name} == "STGCN" ]; then
        # pretrain lite train data
        pushd data/fsd10
        wget -nc https://videotag.bj.bcebos.com/Data/FSD_train_data.npy
        wget -nc https://videotag.bj.bcebos.com/Data/FSD_train_label.npy
        popd
    elif [ ${model_name} == "AGCN2s_joint" ] || [ ${model_name} == "AGCN2s_bone" ]; then
        # pretrain lite train data
        pushd data/fsd10
        wget -nc https://videotag.bj.bcebos.com/Data/FSD_train_data.npy
        wget -nc https://videotag.bj.bcebos.com/Data/FSD_train_label.npy
        popd
    elif [ ${model_name} == "TSM" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_rawframes_small.tar
        tar -xf k400_rawframes_small.tar
        popd
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_pretrain.pdparams --no-check-certificate
    elif [ ${model_name} == "TSN" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_rawframes_small.tar
        tar -xf k400_rawframes_small.tar
        popd
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_pretrain.pdparams --no-check-certificate
    elif [ ${model_name} == "TimeSformer" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_videos_small.tar
        tar -xf k400_videos_small.tar
        popd
        # download pretrained weights
        wget -nc -P ./data https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams --no-check-certificate
    elif [ ${model_name} == "AttentionLSTM" ]; then
        pushd data/yt8m
        ## download & decompression training data
        wget -nc https://videotag.bj.bcebos.com/Data/yt8m_rawframe_small.tar
        tar -xf yt8m_rawframe_small.tar
        ${python} -m pip install tensorflow-gpu==1.14.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
        ${python} tf2pkl.py ./frame ./pkl_frame/
        ls pkl_frame/train*.pkl > train_small.list # 将train*.pkl的路径写入train_small.list
        ls pkl_frame/validate*.pkl > val_small.list # 将validate*.pkl的路径写入val_small.list

        ${python} split_yt8m.py train_small.list # 拆分每个train*.pkl变成多个train*_split*.pkl
        ${python} split_yt8m.py val_small.list # 拆分每个validate*.pkl变成多个validate*_split*.pkl

        ls pkl_frame/train*_split*.pkl > train_small.list # 将train*_split*.pkl的路径重新写入train_small.list
        ls pkl_frame/validate*_split*.pkl > val_small.list # 将validate*_split*.pkl的路径重新写入val_small.list
        popd
    elif [ ${model_name} == "SlowFast" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_videos_small.tar
        tar -xf k400_videos_small.tar
        popd
    elif [ ${model_name} == "BMN" ]; then
        # pretrain lite train data
        pushd ./data
        mkdir bmn_data
        cd bmn_data
        wget -nc https://paddlemodels.bj.bcebos.com/video_detection/bmn_feat.tar.gz
        tar -xf bmn_feat.tar.gz
        wget -nc https://paddlemodels.bj.bcebos.com/video_detection/activitynet_1.3_annotations.json
        wget -nc https://paddlemodels.bj.bcebos.com/video_detection/activity_net_1_3_new.json
        popd
    elif [ ${model_name} == "TokenShiftVisionTransformer" ]; then
        # download pretrained weights
        wget -nc -P ./data https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams --no-check-certificate
    elif [ ${model_name} == "YOWO" ]; then
        # pretrain lite train data
        pushd ./data
        unzip -qo ucf-24-lite.zip
        pushd ./ucf24
        wget -nc -O yolo.weights https://bj.bcebos.com/v1/ai-studio-online/8559f34317e642c888783b81b9fef55f8750ef4b7e924feaac8899f3ab0b151f?responseContentDisposition=attachment%3B%20filename_yolo.weights --no-check-certificate
        wget -nc -O resnext-101-kinetics.pdparams https://bj.bcebos.com/v1/ai-studio-online/cf96b0278a944363bc238cfca727474f889a17f7015746d99523e7068d64e96b?responseContentDisposition=attachment%3B%20filename%3Dresnext-101-kinetics.pdparams&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-05-17T11%3A35%3A40Z%2F-1%2F%2F4a587997708b3c582cbb2f98c8b7d02c395de3ef34e71b918789c761b3feae14 --no-check-certificate
        wget -nc -O YOWO_epoch_00005.pdparams https://bj.bcebos.com/v1/ai-studio-online/cb4f08d38add4d4ab62bc2619eed04d045f9df6513794a8395defdf5ed26d69a?responseContentDisposition=attachment%3B%20filename%3DYOWO_epoch_00005.pdparams&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-05-15T11%3A49%3A23Z%2F-1%2F%2Ffe6055b5abbd28774a27614da64787f8c1794aa1f739d21a6a15276ff1b23e0d --no-check-certificate
        wget -nc -O YOWO_epoch_00005.pdopt https://bj.bcebos.com/v1/ai-studio-online/98f76310ce9b4d53b6ce517238c97ab67c2c2c6dc52c40159ebbf77f61f42c54?responseContentDisposition=attachment%3B%20filename%3DYOWO_epoch_00005.pdopt&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-05-15T11%3A49%3A47Z%2F-1%2F%2Fcb6e6131e47eae31d1856d2e8b029a7259e29f1540d63370a8812373676439d3 --no-check-certificate
        popd
    else
        echo "Not added into TIPC yet."
    fi

elif [ ${MODE} = "whole_train_whole_infer" ];then
    if [ ${model_name} == "PP-TSM" ]; then
        # pretrain whole train data
        pushd ./data/k400
        wget -nc https://ai-rank.bj.bcebos.com/Kinetics400/train_link.list
        wget -nc https://ai-rank.bj.bcebos.com/Kinetics400/val_link.list
        bash download_k400_data.sh train_link.list
        bash download_k400_data.sh val_link.list
        ${python} extract_rawframes.py ./videos/ ./rawframes/ --level 2 --ext mp4 # extract frames from video file
        # download annotations
        wget -nc https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/train_frames.list
        wget -nc https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/val_frames.list
        popd
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams --no-check-certificate
    elif [ ${model_name} == "PP-TSN" ]; then
        # pretrain whole train data
        pushd ./data/k400
        wget -nc https://ai-rank.bj.bcebos.com/Kinetics400/train_link.list
        wget -nc https://ai-rank.bj.bcebos.com/Kinetics400/val_link.list
        bash download_k400_data.sh train_link.list
        bash download_k400_data.sh val_link.list
        # download annotations
        wget -nc https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/train.list
        wget -nc https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/val.list
        popd
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams --no-check-certificate
    elif [ ${model_name} == "AGCN" ]; then
        # pretrain whole train data
        pushd data/fsd10
        wget -nc https://videotag.bj.bcebos.com/Data/FSD_train_data.npy
        wget -nc https://videotag.bj.bcebos.com/Data/FSD_train_label.npy
        popd
    elif [ ${model_name} == "STGCN" ]; then
        # pretrain whole train data
        pushd data/fsd10
        wget -nc https://videotag.bj.bcebos.com/Data/FSD_train_data.npy
        wget -nc https://videotag.bj.bcebos.com/Data/FSD_train_label.npy
        popd
    elif [ ${model_name} == "TSM" ]; then
        # pretrain whole train data
        pushd ./data/k400
        wget -nc https://ai-rank.bj.bcebos.com/Kinetics400/train_link.list
        wget -nc https://ai-rank.bj.bcebos.com/Kinetics400/val_link.list
        bash download_k400_data.sh train_link.list
        bash download_k400_data.sh val_link.list
        ${python} extract_rawframes.py ./videos/ ./rawframes/ --level 2 --ext mp4 # extract frames from video file
        # download annotations
        wget -nc https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/train_frames.list
        wget -nc https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/val_frames.list
        popd
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_pretrain.pdparams --no-check-certificate
    elif [ ${model_name} == "TSN" ]; then
        # pretrain whole train data
        pushd ./data/k400
        wget -nc https://ai-rank.bj.bcebos.com/Kinetics400/train_link.list
        wget -nc https://ai-rank.bj.bcebos.com/Kinetics400/val_link.list
        bash download_k400_data.sh train_link.list
        bash download_k400_data.sh val_link.list
        ${python} extract_rawframes.py ./videos/ ./rawframes/ --level 2 --ext mp4 # extract frames from video file
        # download annotations
        wget -nc https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/train_frames.list
        wget -nc https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/val_frames.list
        popd
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_pretrain.pdparams --no-check-certificate
    elif [ ${model_name} == "TimeSformer" ]; then
        # pretrain whole train data
        pushd ./data/k400
        wget -nc https://ai-rank.bj.bcebos.com/Kinetics400/train_link.list
        wget -nc https://ai-rank.bj.bcebos.com/Kinetics400/val_link.list
        bash download_k400_data.sh train_link.list
        bash download_k400_data.sh val_link.list
        # download annotations
        wget -nc https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/train.list
        wget -nc https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/val.list
        popd
        # download pretrained weights
        wget -nc -P ./data https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams --no-check-certificate
    elif [ ${model_name} == "AttentionLSTM" ]; then
        # pretrain whole train data
        pushd data/yt8m
        mkdir frame
        cd frame
        ## download & decompression training data
        curl data.yt8m.org/download.py | partition=2/frame/train mirror=asia python
        curl data.yt8m.org/download.py | partition=2/frame/validate mirror=asia python
        ${python} -m pip install tensorflow-gpu==1.14.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
        cd ..
        ${python} tf2pkl.py ./frame ./pkl_frame/
        ls pkl_frame/train*.pkl > train.list # 将train*.pkl的路径写入train.list
        ls pkl_frame/validate*.pkl > val.list # 将validate*.pkl的路径写入val.list

        ${python} split_yt8m.py train.list # 拆分每个train*.pkl变成多个train*_split*.pkl
        ${python} split_yt8m.py val.list # 拆分每个validate*.pkl变成多个validate*_split*.pkl

        ls pkl_frame/train*_split*.pkl > train.list # 将train*_split*.pkl的路径重新写入train.list
        ls pkl_frame/validate*_split*.pkl > val.list # 将validate*_split*.pkl的路径重新写入val.list
        popd
    elif [ ${model_name} == "SlowFast" ]; then
        # pretrain whole train data
        pushd ./data/k400
        wget -nc https://ai-rank.bj.bcebos.com/Kinetics400/train_link.list
        wget -nc https://ai-rank.bj.bcebos.com/Kinetics400/val_link.list
        bash download_k400_data.sh train_link.list
        bash download_k400_data.sh val_link.list
        # download annotations
        wget -nc https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/train.list
        wget -nc https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/val.list
        popd
    elif [ ${model_name} == "BMN" ]; then
        # pretrain whole train data
        pushd ./data
        mkdir bmn_data
        cd bmn_data
        wget -nc https://paddlemodels.bj.bcebos.com/video_detection/bmn_feat.tar.gz
        tar -xf bmn_feat.tar.gz
        wget -nc https://paddlemodels.bj.bcebos.com/video_detection/activitynet_1.3_annotations.json
        wget -nc https://paddlemodels.bj.bcebos.com/video_detection/activity_net_1_3_new.json
        popd
    else
        echo "Not added into TIPC yet."
    fi
elif [ ${MODE} = "lite_train_whole_infer" ];then
    if [ ${model_name} == "PP-TSM" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_rawframes_small.tar
        tar -xf k400_rawframes_small.tar
        popd
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams --no-check-certificate
    elif [ ${model_name} == "PP-TSN" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_videos_small.tar
        tar -xf k400_videos_small.tar
        popd
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams --no-check-certificate
    elif [ ${model_name} == "AGCN" ]; then
        # pretrain lite train data
        pushd data/fsd10
        wget -nc https://videotag.bj.bcebos.com/Data/FSD_train_data.npy
        wget -nc https://videotag.bj.bcebos.com/Data/FSD_train_label.npy
        popd
    elif [ ${model_name} == "STGCN" ]; then
        # pretrain lite train data
        pushd data/fsd10
        wget -nc https://videotag.bj.bcebos.com/Data/FSD_train_data.npy
        wget -nc https://videotag.bj.bcebos.com/Data/FSD_train_label.npy
        popd
    elif [ ${model_name} == "TSM" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_rawframes_small.tar
        tar -xf k400_rawframes_small.tar
        popd
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_pretrain.pdparams --no-check-certificate
    elif [ ${model_name} == "TSN" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_rawframes_small.tar
        tar -xf k400_rawframes_small.tar
        popd
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_pretrain.pdparams --no-check-certificate
    elif [ ${model_name} == "TimeSformer" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_videos_small.tar
        tar -xf k400_videos_small.tar
        popd
        # download pretrained weights
        wget -nc -P ./data https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams --no-check-certificate
    elif [ ${model_name} == "AttentionLSTM" ]; then
        # pretrain lite train data
        pushd data/yt8m
        ## download & decompression training data
        wget -nc https://videotag.bj.bcebos.com/Data/yt8m_rawframe_small.tar
        tar -xf yt8m_rawframe_small.tar
        ${python} -m pip install tensorflow-gpu==1.14.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
        ${python} tf2pkl.py ./frame ./pkl_frame/
        ls pkl_frame/train*.pkl > train_small.list # 将train*.pkl的路径写入train_small.list
        ls pkl_frame/validate*.pkl > val_small.list # 将validate*.pkl的路径写入val_small.list

        ${python} split_yt8m.py train_small.list # 拆分每个train*.pkl变成多个train*_split*.pkl
        ${python} split_yt8m.py val_small.list # 拆分每个validate*.pkl变成多个validate*_split*.pkl

        ls pkl_frame/train*_split*.pkl > train_small.list # 将train*_split*.pkl的路径重新写入train_small.list
        ls pkl_frame/validate*_split*.pkl > val_small.list # 将validate*_split*.pkl的路径重新写入val_small.list
        popd
    elif [ ${model_name} == "SlowFast" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_videos_small.tar
        tar -xf k400_videos_small.tar
        popd
    elif [ ${model_name} == "BMN" ]; then
        # pretrain lite train data
        pushd ./data
        mkdir bmn_data
        cd bmn_data
        wget -nc https://paddlemodels.bj.bcebos.com/video_detection/bmn_feat.tar.gz
        tar -xf bmn_feat.tar.gz
        wget -nc https://paddlemodels.bj.bcebos.com/video_detection/activitynet_1.3_annotations.json
        wget -nc https://paddlemodels.bj.bcebos.com/video_detection/activity_net_1_3_new.json
        popd
    else
        echo "Not added into TIPC yet."
    fi
elif [ ${MODE} = "whole_infer" ];then
    if [ ${model_name} = "PP-TSM" ]; then
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_uniform.pdparams --no-check-certificate
    elif [ ${model_name} = "PP-TSN" ]; then
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ppTSN_k400.pdparams --no-check-certificate
    elif [ ${model_name} == "AGCN" ]; then
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo-release2.2/AGCN_fsd.pdparams --no-check-certificate
    elif [ ${model_name} == "STGCN" ]; then
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo-release2.2/STGCN_fsd.pdparams --no-check-certificate
    elif [ ${model_name} == "TSM" ]; then
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo-release2.1/TSM/TSM_k400.pdparams --no-check-certificate
    elif [ ${model_name} == "TSN" ]; then
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo-release2.2/TSN_k400.pdparams --no-check-certificate
    elif [ ${model_name} == "TimeSformer" ]; then
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo-release2.2/TimeSformer_k400.pdparams --no-check-certificate
    elif [ ${model_name} == "AttentionLSTM" ]; then
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo-release2.2/AttentionLSTM_yt8.pdparams --no-check-certificate
    elif [ ${model_name} == "SlowFast" ]; then
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/SlowFast/SlowFast.pdparams --no-check-certificate
    elif [ ${model_name} == "BMN" ]; then
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/BMN/BMN.pdparams --no-check-certificate
    else
        echo "Not added into TIPC yet."
    fi
fi

if [ ${MODE} = "benchmark_train" ];then
    ${python} -m pip install -r requirements.txt
    if [ ${model_name} == "PP-TSM" ]; then
        echo "Not added into TIPC yet."
    elif [ ${model_name} == "PP-TSN" ]; then
        echo "Not added into TIPC yet."
    elif [ ${model_name} == "AGCN" ]; then
        echo "Not added into TIPC yet."
    elif [ ${model_name} == "STGCN" ]; then
        echo "Not added into TIPC yet."
    elif [ ${model_name} == "TSM" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_rawframes_small.tar
        tar -xf k400_rawframes_small.tar
        popd
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_pretrain.pdparams --no-check-certificate
    elif [ ${model_name} == "TSN" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_rawframes_small.tar
        tar -xf k400_rawframes_small.tar
        popd
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_pretrain.pdparams --no-check-certificate
    elif [ ${model_name} == "TimeSformer" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_videos_small.tar
        tar -xf k400_videos_small.tar
        popd
        # download pretrained weights
        wget -nc -P ./data https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams --no-check-certificate
    elif [ ${model_name} == "AttentionLSTM" ]; then
        echo "Not added into TIPC yet."
    elif [ ${model_name} == "SlowFast" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_videos_small.tar
        tar -xf k400_videos_small.tar
        popd
    elif [ ${model_name} == "BMN" ]; then
        # pretrain lite train data
        pushd ./data
        mkdir bmn_data
        cd bmn_data
        wget -nc https://paddlemodels.bj.bcebos.com/video_detection/bmn_feat.tar.gz
        tar -xf bmn_feat.tar.gz
        wget -nc https://paddlemodels.bj.bcebos.com/video_detection/activitynet_1.3_annotations.json
        wget -nc https://paddlemodels.bj.bcebos.com/video_detection/activity_net_1_3_new.json
        popd
    else
        echo "Not added into TIPC yet."
    fi
fi

if [ ${MODE} = "klquant_whole_infer" ]; then
    if [ ${model_name} = "PP-TSM" ]; then
        # download lite data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_rawframes_small.tar
        tar -xf k400_rawframes_small.tar
        popd
        # download inference model
        mkdir ./inference
        pushd ./inference
        wget -nc https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM.zip --no-check-certificate
        unzip ppTSM.zip
        popd
    else
        echo "Not added into TIPC yet."
    fi
fi

if [ ${MODE} = "cpp_infer" ];then
    # install required packages
    apt-get update
    apt install libavformat-dev
    apt install libavcodec-dev
    apt install libswresample-dev
    apt install libswscale-dev
    apt install libavutil-dev
    apt install libsdl1.2-dev
    apt-get install ffmpeg

    if [ ${model_name} = "PP-TSM" ]; then
        # download pretrained weights
        wget -nc -P data/ https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_uniform.pdparams --no-check-certificate
        # export inference model
        ${python} tools/export_model.py -c configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml -p data/ppTSM_k400_uniform.pdparams -o ./inference/ppTSM
    elif [ ${model_name} = "PP-TSN" ]; then
        # download pretrained weights
        wget -nc -P data/ https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ppTSN_k400.pdparams --no-check-certificate
        # export inference model
        ${python} tools/export_model.py -c configs/recognition/pptsn/pptsn_k400_videos.yaml -p data/ppTSN_k400.pdparams -o ./inference/ppTSN
    else
        echo "Not added into TIPC now."
    fi
fi

if [ ${MODE} = "serving_infer_python" ];then
    if [[ ${model_name} == "PP-TSM" ]];then
        # prepare lite infer data for serving
        pushd ./data
        mkdir python_serving_infer_video_dir
        cp ./example.avi python_serving_infer_video_dir/
        popd
        # prepare inference model
        mkdir ./inference
        pushd ./inference
        wget -nc https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM.zip --no-check-certificate
        unzip ppTSM.zip
        popd
    elif [[ ${model_name} == "PP-TSN" ]];then
        # prepare lite infer data for serving
        pushd ./data
        mkdir python_serving_infer_video_dir
        cp ./example.avi python_serving_infer_video_dir/
        popd
        # prepare inference model
        mkdir ./inference
        pushd ./inference
        wget -nc https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSN.zip --no-check-certificate
        unzip ppTSN.zip
        popd
    else
        echo "Not added into TIPC now."
    fi
fi

if [ ${MODE} = "paddle2onnx_infer" ];then
    echo "Not added into TIPC now."
fi

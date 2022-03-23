mkdir MSRVTT
cd MSRVTT
wget https://videotag.bj.bcebos.com/Data/MSRVTT/aggregated_text_feats.tar
wget https://videotag.bj.bcebos.com/Data/MSRVTT/mmt_feats.tar
wget https://videotag.bj.bcebos.com/Data/MSRVTT/raw-captions.pkl
wget https://videotag.bj.bcebos.com/Data/MSRVTT/train_list_jsfusion.txt
wget https://videotag.bj.bcebos.com/Data/MSRVTT/val_list_jsfusion.txt
tar -xvf aggregated_text_feats.tar
tar -xvf mmt_feats.tar

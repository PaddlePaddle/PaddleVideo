cd /home/aistudio/work/

python tools/predict_ensemble.py --input_file /home/aistudio/data/Features_competition_test_B/npy \
 --config configs/localization/bmn.yaml \
 --config1 configs/localization/bmna.yaml \
 --model_file inference/BMN/BMN.pdmodel \
 --params_file inference/BMN/BMN.pdiparams \
 --model_file_1 inference/BMN1/BMN.pdmodel \
 --params_file_1 inference/BMN1/BMN.pdiparams \
 --model_file_2 inference/BMN2/BMN.pdmodel \
 --params_file_2 inference/BMN2/BMN.pdiparams \
 --model_file_3 inference/BMN3/BMN.pdmodel \
 --params_file_3 inference/BMN3/BMN.pdiparams \
 --model_file_4 inference/BMN4/BMN.pdmodel \
 --params_file_4 inference/BMN4/BMN.pdiparams \
 --model_file_5 inference/BMN5/ABMN.pdmodel \
 --params_file_5 inference/BMN5/ABMN.pdiparams \
 --use_gpu=True \
 --use_tensorrt=False




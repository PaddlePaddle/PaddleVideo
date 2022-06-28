# inference sh 
export CUDA_VISIBLE_DEVICES=0
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_fraction_of_gpu_memory_to_use=0.5
export FLAGS_reallocate_gpu_memory_in_mb=0
export FLAGS_memory_fraction_of_eager_deletion=1
python scenario_lib/inference.py --model_name=AttentionLstmErnie \
--config=./conf/conf.txt \
--save_inference_model=inference_models_save \
--output='output.json'

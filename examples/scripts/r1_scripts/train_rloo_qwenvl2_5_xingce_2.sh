export DATASET="/mnt/afs/wangjiahao/workspace/o1_r1/lmm-r1/workspace/dataset/xingce_v1/vlm_xingce_dataset/all_data_easy_sys.jsonl"
#export DATASET="/mnt/afs/wangjiahao/workspace/o1_r1/lmm-r1/examples/data/test_message_wo_image_multi.jsonl"
#MODEL_CPK_NAME="qwenvl25_7B_rej_sample_ins_rloo_xingcev1_lr1e-5"
MODEL_CPK_NAME="qwenvl25_7B_ins_rloo_xingcev1_lr1e-6_easy_sys"
PRETRAIN_MODEL="/mnt/afs/wangjiahao/workspace/hf_home/Qwen2.5-VL-7B-Instruct"
#PRETRAIN_MODEL="/mnt/afs/wangjiahao/workspace/o1_r1/lmm-r1/workspace/qwen25_7b_sampled_sft_1k_v1"
SAVE_PATH="/mnt/afs/wangjiahao/workspace/o1_r1/lmm-r1/workspace"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}/logs"
T=`date +%Y%m%d_%H%M%S`
LOG_PATH="${SAVE_PATH}/${MODEL_CPK_NAME}/logs/${MODEL_CPK_NAME}_${T}.log"
ray stop
python -m openrlhf.models.remote_rm.option_verifier --dataset $DATASET --input_key message --prompt-template chatml > "${SAVE_PATH}/${MODEL_CPK_NAME}/logs/remote_rm_${T}.log" 2>&1 &
childpid=$!

ray start --head --node-ip-address 0.0.0.0 --num-gpus 8 --temp-dir ~/.cache/ray

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/mnt/afs/wangjiahao/workspace/o1_r1/lmm-r1"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --remote_rm_url http://127.0.0.1:5000/get_reward \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 8 \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_enable_sleep \
   --vllm_gpu_memory_utilization 0.6 \
   --vllm_sync_backend gloo \
   --enable_prefix_caching \
   --pretrain $PRETRAIN_MODEL \
   --save_path $SAVE_PATH/$MODEL_CPK_NAME \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 256 \
   --temperature 1 \
   --n_samples_per_prompt 16 \
   --max_epochs 1 \
   --num_episodes 30 \
   --prompt_max_len 1024 \
   --max_samples 100000 \
   --generate_max_len 3000 \
   --advantage_estimator rloo \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 1e-6 \
   --init_kl_coef 0.0 \
   --prompt_data $DATASET \
   --normalize_reward \
   --flash_attn \
   --input_key message \
   --gradient_checkpointing \
   --save_steps 10 \
   --ckpt_path $SAVE_PATH/$MODEL_CPK_NAME/ckpt \
   --save_hf_ckpt \
   --use_tensorboard $SAVE_PATH/$MODEL_CPK_NAME/logs \
   --max_ckpt_num 15 \
   --train_vlm \
   2>&1 | tee ${LOG_PATH}

# also supports --advantage_estimator rloo
ray stop
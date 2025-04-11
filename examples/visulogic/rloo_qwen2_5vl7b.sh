NODE_RANK=${1:-0}
export DATASET="visulogic_train.jsonl"
PRETRAIN_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
SAVE_PATH="log"
mkdir -p "ckpt"
T=`date +%Y%m%d_%H%M%S`
childpid=$!

export RAY_ADDRESS='http://127.0.0.1:8265'
python -m openrlhf.models.remote_rm.option_verifier --dataset $DATASET --input_key message --prompt-template chatml > "log/remote_rm_node$NODE_RANK_${T}.log" 2>&1 &
if [ "$NODE_RANK" = "0" ]; then
ray job submit \
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
   --save_path ckpt/ \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 256 \
   --temperature 1 \
   --n_samples_per_prompt 16 \
   --max_epochs 1 \
   --num_episodes 5 \
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
   --ckpt_path ckpt/ \
   --save_hf_ckpt \
   --use_tensorboard log/ \
   --max_ckpt_num 15 \
   --train_vlm  \
   --model_family qwen 
fi
# also supports --advantage_estimator rloo

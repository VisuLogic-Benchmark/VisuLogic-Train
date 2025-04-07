#!/bin/bash

#SBATCH -p Intern5
#SBATCH -J wwy_vl
#SBATCH --gres=gpu:8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=8
#SBATCH --cpus-per-task=12
#SBATCH --quotatype=reserved
#SBATCH -o logs/internvl_v2_5/grpo/m3cot.log
#SBATCH -e logs/internvl_v2_5/grpo/m3cot.log
#SBATCH --overcommit               # needed for pytorch

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..."

OUTPUT_DIR="/mnt/petrelfs/wangweiyun/workspace_wwy/OpenRLHF-250228/outputs/internvl_v2_5/grpo_v1"

# setup environment
export PYTHONPATH="${PYTHONPATH}:/mnt/petrelfs/wangweiyun/workspace_cz/InternVL/internvl_chat_dev/petrel-oss-python-sdk"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TRITON_CACHE_DIR="/tmp/triton_wwy/"

export ACC_REWARD_PATH="/mnt/petrelfs/wangweiyun/workspace_wwy/OpenRLHF-250228/logs/acc_reward.log"
export FORMAT_REWARD_PATH="/mnt/petrelfs/wangweiyun/workspace_wwy/OpenRLHF-250228/logs/format_reward.log"


# launch ray daemon
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names
nodes_array=( $nodes )
node_1=${nodes_array[0]}
ip=$node_1

port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 -w "$node_1" bash -c \
    "ray start --head --node-ip-address=${ip} --port=${port} --num-gpus 8 --block" &

sleep 10s

worker_num=$((SLURM_JOB_NUM_NODES))
echo "worker_num: $worker_num"

for ((i = 1; i < worker_num; i++)); do
node_i=${nodes_array[$i]}
echo "STARTING WORKER $i at $node_i"
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 -w "$node_i" bash -c \
    "ray start --address ${ip_head} --num-gpus 8 --block" &

sleep 1s;
done

sleep 30s

mkdir -p ${OUTPUT_DIR}

# NOTE: working_dir中的所有文件都会上传到ray服务端，太大了会报错网络异常
# NOTE: 相对路径是相对服务端/tmp的路径，不是起脚本的路径了

# ===== submit ray job =====
# Job start
srun --overlap --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:0 -w "$node_1" bash -c \
"   ray job submit --address=http://localhost:8265 \
    --runtime-env-json='{\"working_dir\": \"./ray_working_dir\"}' \
    -- python3 -u /mnt/petrelfs/wangweiyun/workspace_wwy/OpenRLHF-250228/internvl/train_ppo_ray_vl.py \
    --ref_num_nodes 8 \
    --ref_num_gpus_per_node 8 \
    --actor_num_nodes 8 \
    --actor_num_gpus_per_node 8 \
    --vllm_num_engines 64 \
    --vllm_tensor_parallel_size 1 \
    --colocate_critic_reward \
    --colocate_actor_ref \
    --colocate_all_models \
    --advantage_estimator group_norm \
    --n_samples_per_prompt 8 \
    --remote_rm_url /mnt/petrelfs/wangweiyun/workspace_wwy/OpenRLHF-250228/internvl/reward_fcuntions/reward_func.py \
    --meta_path /mnt/petrelfs/wangweiyun/workspace_wwy/OpenRLHF-250228/datasets/internvl/meta/m3cot.json \
    --pretrain /mnt/petrelfs/share_data/wangweiyun/share_internvl_preview/InternVL2_5-8B \
    --ckpt_path ${OUTPUT_DIR}/ckpt \
    --save_path ${OUTPUT_DIR} \
    --save_steps 20 \
    --save_hf_ckpt \
    --disable_ds_ckpt \
    --micro_train_batch_size 2 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size 1024 \
    --num_episodes 10 \
    --max_epochs 1 \
    --prompt_max_len 8192 \
    --generate_max_len 8192 \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 2e-6 \
    --init_kl_coef 0.01 \
    --gamma 1.0 \
    --use_kl_loss \
    --use_kl_estimator_k3 \
    --input_key context_messages \
    --apply_chat_template \
    --normalize_reward \
    --flash_attn \
    --vllm_sync_backend nccl \
    --use_tensorboard ${OUTPUT_DIR} \
    --gradient_checkpointing"

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..."

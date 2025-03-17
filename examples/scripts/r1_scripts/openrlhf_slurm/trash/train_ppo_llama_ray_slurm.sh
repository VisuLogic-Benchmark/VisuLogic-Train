#!/bin/bash

#SBATCH -p Intern5
#SBATCH -J wwy_vl
#SBATCH --gres=gpu:8
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=2
#SBATCH --cpus-per-task=4
#SBATCH --quotatype=reserved
#SBATCH -o logs/text_only/train_ppo_llama_ray.log
#SBATCH -e logs/text_only/train_ppo_llama_ray.log
#SBATCH --overcommit               # needed for pytorch

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..."

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

mkdir -p ./outputs

# NOTE: working_dir中的所有文件都会上传到ray服务端，太大了会报错网络异常
# NOTE: 相对路径是相对服务端/tmp的路径，不是起脚本的路径了

# ===== submit ray job =====
# Job start
srun --overlap --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:0 -w "$node_1" bash -c \
"   ray job submit --address=http://localhost:8265 \
    --runtime-env-json='{\"working_dir\": \"./ray_working_dir\"}' \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 4 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 4 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 4 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 2 \
    --colocate_critic_reward \
    --colocate_actor_ref \
    --pretrain /mnt/petrelfs/share_data/wangweiyun/share_ckpt_hf/OpenRLHF/Llama-3-8b-sft-mixture \
    --reward_pretrain /mnt/petrelfs/share_data/wangweiyun/share_ckpt_hf/OpenRLHF/Llama-3-8b-rm-mixture \
    --save_path /mnt/petrelfs/wangweiyun/workspace_wwy/OpenRLHF-250228/outputs/examples/checkpoint/llama3-8b-rlhf \
    --micro_train_batch_size 8 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 16 \
    --rollout_batch_size 1024 \
    --max_samples 100000 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data /mnt/petrelfs/wangweiyun/workspace_wwy/OpenRLHF-250228/datasets/OpenRLHF/prompt-collection-v0.1 \
    --input_key context_messages \
    --apply_chat_template \
    --normalize_reward \
    --adam_offload \
    --flash_attn \
    --packing_samples \
    --vllm_sync_backend nccl \
    --use_tensorboard /mnt/petrelfs/wangweiyun/workspace_wwy/OpenRLHF-250228/outputs/examples/checkpoint/llama3-8b-rlhf \
    --gradient_checkpointing"

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..."

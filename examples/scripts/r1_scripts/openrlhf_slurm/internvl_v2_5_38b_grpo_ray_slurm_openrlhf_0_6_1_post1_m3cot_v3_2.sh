#!/bin/bash

#SBATCH -p Intern5
#SBATCH -J wwy_vl
#SBATCH --gres=gpu:8
#SBATCH --ntasks=256
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=32
#SBATCH --cpus-per-task=12
#SBATCH --quotatype=reserved
#SBATCH -o logs/internvl_v2_5/grpo/m3cot_38B_v3_2.log
#SBATCH -e logs/internvl_v2_5/grpo/m3cot_38B_v3_2.log
#SBATCH --overcommit               # needed for pytorch

export RAY_TMPDIR=/dev/shm/ray_wwy
rm -rf ${RAY_TMPDIR}
mkdir -p ${RAY_TMPDIR}

CURR_DIR="/mnt/petrelfs/wangweiyun/workspace_wwy/OpenRLHF-250228"
JOBLOG="${CURR_DIR}/logs/internvl_v2_5/grpo/m3cot_38B_v3_2.log"

echo "pwd=$(pwd)"
echo "JOBLOG=${JOBLOG}"
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..." &>> ${JOBLOG}

OUTPUT_DIR="${CURR_DIR}/outputs/internvl_v2_5/grpo_openrlhf_0_6_1_post1_m3cot_38B_v3_2"

# setup environment
export PYTHONPATH="/mnt/petrelfs/wangweiyun/workspace_cz/InternVL/internvl_chat_dev/petrel-oss-python-sdk"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TRITON_CACHE_DIR="/tmp/triton_wwy/"

export ACC_REWARD_PATH="${CURR_DIR}/logs/acc_reward.log"
export FORMAT_REWARD_PATH="${CURR_DIR}/logs/format_reward.log"


# launch ray daemon
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names
nodes_array=( $nodes )
node_1=${nodes_array[0]}
ip=$node_1

port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head" &>> ${JOBLOG}

echo "STARTING HEAD at $node_1" &>> ${JOBLOG}
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 -w "$node_1" bash -c \
    "ray start --head --node-ip-address=${ip} --port=${port} --num-gpus 8 --block" &>> ${JOBLOG} &

sleep 10s

worker_num=$((SLURM_JOB_NUM_NODES))
echo "worker_num: $worker_num" &>> ${JOBLOG}

for ((i = 1; i < worker_num; i++)); do
node_i=${nodes_array[$i]}
echo "STARTING WORKER $i at $node_i" &>> ${JOBLOG}
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 -w "$node_i" bash -c \
    "ray start --address ${ip_head} --num-gpus 8 --block" &>> ${JOBLOG} &

sleep 1s;
done

sleep 30s

mkdir -p ${OUTPUT_DIR}

# NOTE: working_dir中的所有文件都会上传到ray服务端，太大了会报错网络异常
# NOTE: 相对路径是相对服务端/tmp的路径，不是起脚本的路径了

echo "submit ray job" &>> ${JOBLOG}

# ===== submit ray job =====
# Job start
srun --overlap --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:0 -w "$node_1" bash -c \
"   ray job submit --address=http://localhost:8265 \
    --runtime-env-json='{\"working_dir\": \"${CURR_DIR}/ray_working_dir\"}' \
    -- python3 -u ${CURR_DIR}/internvl/train_ppo_ray_vl.py \
    --ref_num_nodes 8 \
    --ref_num_gpus_per_node 8 \
    --actor_num_nodes 8 \
    --actor_num_gpus_per_node 8 \
    --vllm_num_engines 64 \
    --vllm_tensor_parallel_size 2 \
    --vllm_gpu_memory_utilization 0.9 \
    --advantage_estimator group_norm \
    --n_samples_per_prompt 8 \
    --remote_rm_url ${CURR_DIR}/internvl/reward_fcuntions/reward_func.py \
    --meta_path ${CURR_DIR}/datasets/internvl/meta/m3cot.json \
    --pretrain /mnt/petrelfs/share_data/wangweiyun/share_internvl_preview/InternVL2_5-38B \
    --ckpt_path ${OUTPUT_DIR}/ckpt \
    --save_path ${OUTPUT_DIR} \
    --save_steps 20 \
    --save_hf_ckpt \
    --disable_ds_ckpt \
    --micro_train_batch_size 4 \
    --train_batch_size 256 \
    --micro_rollout_batch_size 4 \
    --rollout_batch_size 1024 \
    --num_episodes 10 \
    --max_epochs 1 \
    --prompt_max_len 8192 \
    --generate_max_len 4096 \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 1e-6 \
    --init_kl_coef 0.01 \
    --gamma 1.0 \
    --use_kl_loss \
    --kl_estimator k3 \
    --input_key context_messages \
    --apply_chat_template \
    --normalize_reward \
    --flash_attn \
    --vllm_sync_backend nccl \
    --use_tensorboard ${OUTPUT_DIR} \
    --gradient_checkpointing" &>> ${JOBLOG}

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..." &>> ${JOBLOG}


# for ((i = 0; i < worker_num; i++)); do
# node_i=${nodes_array[$i]}
# echo "STOPING WORKER $i at $node_i" &>> ${JOBLOG}
# srun --overlap --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:0 -w "$node_i" bash -c \
#     "ray stop" &>> ${JOBLOG} &

# sleep 1s;
# done


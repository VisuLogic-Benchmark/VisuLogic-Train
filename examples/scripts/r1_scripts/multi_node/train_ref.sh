set -x

set -x
T=`date +%Y%m%d_%H%M%S`

export NCCL_DEBUG=INFO


export PATH=/mnt/afs/wangjiahao/.conda/bin:$PATH
#export PATH=/mnt/afs1/luojiapeng/miniconda3/bin:$PATH
#export TORCH_EXTENSIONS_DIR=/mnt/afs/liangjinwei/.cache/torch_extensions
export TORCH_EXTENSIONS_DIR=/mnt/afs/wangjiahao/.cache/torch_extensions

cd
cp /mnt/afs1/tianhao2/aoss.conf ./

ROOT=/mnt/afs/wangjiahao/workspace/OpenRLHF/     #改成自己的openrlhf目录
cd $ROOT
# sleep 0.5d
source activate
conda activate /mnt/afs/wangjiahao/.conda/envs/internvl2_ljp
#conda activate /mnt/afs1/luojiapeng/miniconda3/envs/internvl2

export PYTHONPATH=$ROOT:$PYTHONPATH

OUTPUT_DIR=/mnt/afs/wangjiahao/workspace/dpo/model/dpo_openrlhf_check_1.2.4.1_0122   #修改成希望的输出目录
LOG_DIR=${OUTPUT_DIR}/logs
mkdir -p ${LOG_DIR}
LOG_FILE=${LOG_DIR}/node${RANK}_${T}.log

#meta_path=/mnt/afs/wangjiahao/workspace/OpenRLHF/data/testdata_1sample_3.json  #改成准备好的metepath
meta_path=/mnt/afs/lupeng1/myshare/workspace/checkpoints/DPO/vit0.3b_qwen2_5_14b_stage3_v8.3.2_stage4_dpo_v1.2/vit0.3b_qwen2_5_14b_stage3_v8.3.2_stage4_dpo_1.2.4.1_20241229/data/dpo_data_v1.2.4.1_logps.json
#model_name_or_path=/mnt/afs/wangjiahao/workspace/dpo/InternVL-dev2/internvl_chat_dev/workspace/debug/3000_hf
model_name_or_path=/mnt/afs/share_data/jiangtan/train_internvl/RUN/fusion/vit0.3b_qwen2_5_14b_stage3_v8.3.2_20241223/10000_hf   #改成自己的sft模型
#model_name_or_path=/mnt/afs/share_data/embodied/models/embodied_sft_20241023_vit0.3b_internlm7b_stage3_lr2e-5_car_general_embodied_full_stage3_new_config
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-10086}

GPUS=128    #数量和test_internvl_srun.sh中总卡数保持一致
GPUS_PER_NODE=8
RANK=${RANK:-0} # srun env node rank
WORLD_SIZE=${WORLD_SIZE:-1} # srun env node num
echo "nnodes=${WORLD_SIZE}, node_rank=${RANK}"

BATCH_SIZE=128  #目前只测试过bs=GPUS
PER_DEVICE_BATCH_SIZE=1  #目前只测试过bs=GPUS
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

# number of gpus: 512
# batch size per gpu: 1 (6)
# gradient accumulation steps: 1
# total batch size: 3k
# epoch: 1
# steps

echo $(which python)

export LAUNCHER=pytorch


read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
   --save_path ${OUTPUT_DIR} \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size ${BATCH_SIZE} \
   --micro_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
   --pretrain ${model_name_or_path} \
   --bf16 \
   --max_epochs 1 \
   --max_len 6264 \
   --zero_stage 3 \
   --learning_rate 1.0e-6 \
   --beta 0.1 \
   --dataset /mnt/afs/wangjiahao/workspace/OpenRLHF/data/testdata.json \
   --meta_path ${meta_path} \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --load_checkpoint \
   --max_samples 10000 \
   --lr_warmup_ratio 0.03 \
   --lr_scheduler_type cosine \
   --use_tensorboard ${OUTPUT_DIR}/runs \
   --internvl  \
   --force_image_size 448 \
   --conv_style internlm2-chat-v3 \
   --scale_threshold v3 \
   --min_dynamic_patch 1 \
   --max_dynamic_patch 12 \
   --lr_warmup_ratio 0.03 \
   --l2 0.05 \
   --adam_epsilon 1.0e-8 \
   --adam_betas 0.9 0.999 \
   --lr_scheduler_type cosine \
   --precompute_ref_log_probs \
   --seed 0

EOF
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --ipo [for IPO]
    # --label_smoothing 0.1 [for cDPO]
    # --ref_offload
    # --packing_samples
    # --nll_loss_coef (Regularization with NLL loss)
    #   --adam_offload

echo training_commands &>> ${LOG_FILE}
echo $training_commands &>> ${LOG_FILE}

torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$RANK \
  --master_port=$MASTER_PORT --master_addr=$MASTER_ADDR -m ${training_commands} 2>&1 | tee ${LOG_FILE}


SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

#############################
NNODES=6
GPU_PER_NODE=8
CPUS_PER_NODE=128
SCRIPT=$SCRIPT_DIR/train.sh
CONFIG_SCRIPTS=rloo_internvl2_5_38b.sh
#############################

export GPUS=$((GPU_PER_NODE * NNODES))

cd $(dirname $0)
mkdir -p $(dirname $0)/log
T=$(date +%Y%m%d%H%M)
LOG_PATH="log/srun_log_${T}.log"

srun -p Intern5 \
  --job-name=internvl_rl \
  --ntasks=${NNODES} \
  --ntasks-per-node=1 \
  --gres=gpu:${GPU_PER_NODE} \
  --cpus-per-task=${CPUS_PER_NODE} \
  --kill-on-bad-exit=1 \
  -o $LOG_PATH \
  -e $LOG_PATH \
  --quotatype=reserved \
  bash $SCRIPT $CONFIG_SCRIPTS
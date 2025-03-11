
PARTITION=cc4cdf6c-944a-4c46-a9de-3d508a06c4dd #AMP
#PARTITION=b7c081ea-ab5a-4278-ab4a-c51bc222de13
#PARTITION=amplarge
#PARTITION=r1-m1

WORKSPACE=a58d023b-de76-475f-89c2-7e50f7aa3c7a
CONTAINTER=registry.ms-sc-01.maoshanwangtech.com/studio-aicl/ubuntu20.04-py3.10-cuda11.8-cudnn8-transformer4.28.0:master-20230626-172512-32302
MOUNT=ce3b1174-f6eb-11ee-a372-82d352e10aed:/mnt/afs,1f29056c-c3f2-11ee-967e-2aea81fd34ba:/mnt/afs2,047443d2-c3f2-11ee-a5f9-9e29792dec2f:/mnt/afs1

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

#############################
JOBNAME="openrlhf_qwen2_5_vl_7b_2nodes"
GPU_PER_NODE=8
NNODES=2
SCRIPT=$SCRIPT_DIR/train.sh
#############################

if [ "$PARTITION" != "r1-m1" ]; then
    SPECS="N6lS.Iu.I80.$GPU_PER_NODE"
else
    SPECS="N6lS.Iq.I10.$GPU_PER_NODE"
fi

GPUS=$((GPU_PER_NODE * NNODES))

sco acp jobs create \
--workspace-name $WORKSPACE -p $PARTITION \
--container-image-url $CONTAINTER \
--storage-mount $MOUNT \
--worker-spec ${SPECS} \
-f pytorch -N ${NNODES} \
-j $JOBNAME \
--command="NNODES=$NNODES bash $SCRIPT"
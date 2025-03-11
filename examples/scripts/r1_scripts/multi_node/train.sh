set -ex

#export PATH=/mnt/afs1/luojiapeng/miniconda3/bin:$PATH
export PATH=/mnt/afs/wangjiahao/.conda/bin:$PATH

ROOT=/mnt/afs/wangjiahao/workspace/o1_r1/lmm-r1    #改成自己的openrlhf目录

cd $(dirname $0)
#cd $(ROOT)
source activate
conda activate /mnt/afs/wangjiahao/.conda/envs/openrlhf
# pip install --user -e /mnt/afs/lupeng1/myshare/o1_rl/expr/align_openrlhf/train_openrlhf/code
# pip install --user "antlr4-python3-runtime==4.7.2"


#cp /mnt/afs/lupeng1/aoss.conf ~/
# export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=200
export NCCL_IB_RETRY_CNT=20
export NCCL_IB_AR_THRESHOLD=0

echo $MASTER_ADDR 
echo $MASTER_PORT 
echo $WORLD_SIZE 
echo $RANK

export NCCL_P2P_LEVEL=NVL
export PYTHONPATH=$PWD:$PYTHONPATH
# export HF_HOME=$(realpath $PWD/../../cache/huggingface/)

num_nodes=$WORLD_SIZE
start_time=$(date +%Y%m%d%H%M)

# num_nodes has to be at least 1
if [ $num_nodes -lt 1 ]; then
    echo "Number of nodes must be at least 1"
    exit 1
fi

# if HOST contains "master", then this is the head node
if [[ $RANK -eq 0 ]]; then
    node_role="master"
else
    node_role="worker"
fi
head_node_ip=$MASTER_ADDR

script="vl_r1.sh"
mkdir -p $(dirname $0)/log

wait_time=60
if [ "$node_role" == "master" ]; then
    echo "Starting Ray head node..."
    # Start Ray on this node as the head node and extract its address
    # The `ray start --head` command outputs information that includes the address,
    # but here we're assuming it's known or statically assigned for simplicity.
    ray start --head --dashboard-host 0.0.0.0 --port=6379 --dashboard-port=8265 --resources '{"COMPUTE": 100000000000000.0, "HEAD": 100000000000000.0}'
    sleep $wait_time
elif [ "$node_role" == "worker" ]; then
    sleep $wait_time
    attempt=1
    echo "Starting Ray worker node and attempting to connect to the head node at $head_node_ip:6379"
    while true; do
        # Attempt to start Ray and connect to the head node
        ray start --address="$head_node_ip:6379" --dashboard-port=8265 --resources '{"COMPUTE": 100000000000000.0, "virtual_cluster_default": 100000000000000.0}'  && break || {
            if [ $attempt -le 5 ]; then
                echo "Ray worker start attempt $attempt failed. Retrying in $wait_time seconds..."
                ((attempt++))
                sleep $wait_time
            else
                echo "Failed to connect to the head node after $wait_time attempts. Exiting."
                exit 1
            fi
        }
    done
fi
# run the training script once Ray has been started on all nodes
sleep $wait_time
if [ "$node_role" == "master" ]; then
    num_active_ray_nodes=$(ray list nodes | grep ALIVE | wc -l)
    echo "Number of active Ray nodes: $num_active_ray_nodes"
    if [ $num_active_ray_nodes -lt $num_nodes ]; then
        echo "Waiting for all Ray nodes to start..."
        attempt=1
        while true; do
            num_active_ray_nodes=$(ray list nodes | grep ALIVE | wc -l)
            if [ $num_active_ray_nodes -eq $num_nodes ]; then
                break
            elif [ $attempt -le 5 ]; then
                echo "python command attempt $attempt failed. Retrying in $wait_time seconds..."
                ((attempt++))
                sleep $wait_time
            else
                echo "Failed to connect to the head node after $wait_time attempts. Exiting."
                exit 1
            fi
        done
    fi
    echo "End starting"
    # python examples/scripts/test_ray.py
    sh $(dirname $0)/${script} $RANK 2>&1 | tee log/vl_r1_${num_nodes}_${node_role}_${RANK}_${start_time}.log
else
    echo "End starting"
    sh $(dirname $0)/${script} $RANK 2>&1 | tee log/vl_r1_${num_nodes}_${node_role}_${RANK}_${start_time}.log
    sleep infinity
fi


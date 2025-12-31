master_addr=${master_addr:=$ARNOLD_WORKER_0_HOST}
master_port=${master_port:=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)}
nproc_per_node="${nproc_per_node:=$ARNOLD_WORKER_GPU}"
node_rank="${node_rank:=$ARNOLD_ID}"
nnodes="${nnodes:=$ARNOLD_WORKER_NUM}"

# export NVSHMEM_DEBUG=INFO
# export NVSHMEM_DEBUG_FILE=/tmp/nvshmem_%h_%p.log

torchrun \
  --node_rank=$node_rank \
  --nproc_per_node=1 \
  --nnodes=$nnodes \
  --rdzv_endpoint=${master_addr}:${master_port} \
  multi_node.py

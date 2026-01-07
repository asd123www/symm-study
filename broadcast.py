import os
import argparse

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import torch.distributed._symmetric_memory as symm_mem
import torch.distributed._symmetric_memory._nvshmem_triton as nvshmem
from torch.distributed._symmetric_memory._nvshmem_triton import requires_nvshmem


@requires_nvshmem
@triton.jit
def broadcast_int32_hybrid_kernel(
    buf_ptrs,
    local_buf_ptr,
    offset: tl.constexpr,
    root_rank: tl.constexpr,
    world_size: tl.constexpr,
    node_start: tl.constexpr,
    node_end: tl.constexpr,
):
    x = tl.load(local_buf_ptr + 0)
    for pe in tl.static_range(world_size):
        if (pe >= node_start) and (pe < node_end) and (pe != root_rank):
            peer_ptr = buf_ptrs[pe]
            peer_ptr = tl.cast(peer_ptr, tl.pointer_type(tl.int32))
            tl.store(peer_ptr + offset, x)
        else:
            nvshmem.put(local_buf_ptr + offset, local_buf_ptr + 0, 1, pe)

    nvshmem.quiet()


def main():
    # SymmMem with NVSHMEM backend
    symm_mem.set_backend("NVSHMEM")
    symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
    dist.barrier()

    buf = symm_mem.empty(args.symm_buffer_size, dtype=torch.int32, device=f"cuda:{local_rank}")
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD.group_name)
    dist.barrier()

    buf.fill_(1)

    root_rank = 0
    root_node = rank // local_world_size
    node_start = root_node * local_world_size
    node_end = min(node_start + local_world_size, world_size)

    if rank == root_rank:
        buf.fill_(args.value)
        buf_ptrs = tuple(hdl.buffer_ptrs)

        broadcast_int32_hybrid_kernel[(1,)](
            buf_ptrs,
            buf,
            offset=args.offset,
            root_rank=root_rank,
            world_size=world_size,
            node_start=node_start,
            node_end=node_end,
            num_warps=1,
        )
        torch.cuda.synchronize()

    dist.barrier()
    torch.cuda.synchronize()

    got = int(buf.cpu()[args.offset])
    expected = args.value
    for r in range(world_size):
        dist.barrier()
        if r == rank:
            print(f"[rank{rank}] buf[{args.offset}]={got} expected={expected}", flush=True)
        dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank0 broadcast 1 int32: direct intra-node + NVSHMEM inter-node, peer loop in kernel")
    parser.add_argument("--symm_buffer_size", type=int, default=1024, help="Number of int32 elements in symmetric buffer")
    parser.add_argument("--offset", type=int, default=10, help="Destination index in each rank's buffer")
    parser.add_argument("--value", type=int, default=238973473, help="Value broadcast from rank0 (stored at buf[0])")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    assert local_rank == rank % local_world_size

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=local_rank)

    if not (0 <= args.offset < args.symm_buffer_size):
        raise ValueError(f"--offset must be in [0, {args.symm_buffer_size-1}]")

    main()

    free, total = torch.cuda.mem_get_info()
    print(f"[rank{rank}] CUDA mem used={(total-free)/2**20:.1f} MiB free={free/2**20:.1f} MiB", flush=True)

    dist.destroy_process_group()

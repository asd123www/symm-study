import os
import argparse

import torch
import torch.distributed as dist
import triton
import triton.language as tl
import torch.distributed._symmetric_memory as symm_mem
import torch.distributed._symmetric_memory._nvshmem_triton as nvshmem
from torch.distributed._symmetric_memory._nvshmem_triton import requires_nvshmem


@triton.jit
def dma_put1_kernel(peer_dst, src, offset: tl.constexpr):
    x = tl.load(src)
    tl.store(peer_dst + offset, x)


def main():
    # use NVSHMEM backend
    symm_mem.set_backend("NVSHMEM")
    symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
    dist.barrier()

    buffer = symm_mem.empty(args.symm_buffer_size, dtype=torch.int32, device=f"cuda:{local_rank}")
    hdl = symm_mem.rendezvous(buffer, dist.group.WORLD.group_name)
    dist.barrier()

    if rank == 0:
        print("hdl.buffer_ptrs:", hdl.buffer_ptrs)

    if rank == 0:
        buffer.fill_(238973473)
    else:
        buffer.fill_(1)

    peer = 1 - rank
    offset = 10
    peer_buf = hdl.get_buffer(peer, buffer.shape, buffer.dtype)

    dma_put1_kernel[(1,)](peer_buf, buffer, offset=offset)
    torch.cuda.synchronize()
    dist.barrier()

    got = int(buffer.cpu()[offset])
    expected = 1 if rank == 0 else 238973473
    for i in range(world_size):
        if i == rank:
            print(f"rank{rank} got dst={got} expected={expected}", flush=True)
        dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Symmetric Memory All-gather')
    parser.add_argument('--symm_buffer_size', type=int, default=1024*1024*1024, help='Symmetric memory buffer size')
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    assert local_rank == rank % local_world_size
    torch.cuda.set_device(local_rank)

    dist.init_process_group("nccl", device_id=local_rank)

    main()

    # Check nvshmem memory usage
    free, total = torch.cuda.mem_get_info()
    print(f"[Mem Info]: used={(total-free)/2**20:.1f} MiB, free={free/2**20:.1f} MiB")

    dist.destroy_process_group()

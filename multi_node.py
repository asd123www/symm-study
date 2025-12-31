import os
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import torch.distributed._symmetric_memory as symm_mem
import torch.distributed._symmetric_memory._nvshmem_triton as nvshmem
from torch.distributed._symmetric_memory._nvshmem_triton import requires_nvshmem

@requires_nvshmem
@triton.jit
def put1_kernel(dst, src, pe: tl.constexpr):
    nvshmem.put(dst, src, 1, pe)
    nvshmem.quiet()  # force completion of GPU-initiated ops :contentReference[oaicite:1]{index=1}

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    dist.init_process_group("nccl", device_id=local_rank)

    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2

    symm_mem.set_backend("NVSHMEM")
    symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
    dist.barrier()

    dst = symm_mem.empty(1, dtype=torch.int32, device=f"cuda:{local_rank}")
    src = symm_mem.empty(1, dtype=torch.int32, device=f"cuda:{local_rank}")

    dst.fill_(-1)  # sentinel
    src.fill_(238973473 if rank == 0 else 1)

    # Safer: rendezvous both allocations (keeps symmetry/registration explicit)
    symm_mem.rendezvous(dst, dist.group.WORLD)
    symm_mem.rendezvous(src, dist.group.WORLD)
    dist.barrier()

    peer = 1 - rank
    put1_kernel[(1,)](dst, src, pe=peer)
    torch.cuda.synchronize()
    dist.barrier()

    got = int(dst.cpu()[0])
    expected = 1 if rank == 0 else 238973473
    print(f"rank{rank} got dst={got} expected={expected}", flush=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

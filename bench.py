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
def put_chunks_kernel(dst_ptr, src_ptr, n_elems, pe, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK
    # Grid is sized so start < n_elems always holds.
    n_this = tl.minimum(n_elems - start, BLOCK)
    nvshmem.put(dst_ptr + start, src_ptr + start, n_this, pe)


@requires_nvshmem
@triton.jit
def quiet_kernel():
    # Wait for completion of all outstanding GPU-initiated NVSHMEM ops
    nvshmem.quiet()


def sizeof_dtype(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bytes", type=int, default=256 * 1024 * 1024, help="message size in bytes (<= ~2GB recommended)")
    parser.add_argument("--iters", type=int, default=100, help="timed iterations")
    parser.add_argument("--warmup", type=int, default=20, help="warmup iterations (not timed)")
    parser.add_argument("--block-bytes", type=int, default=1 * 1024 * 1024, help="bytes per chunk per Triton program")
    parser.add_argument("--dtype", type=str, default="uint8", choices=["uint8", "float16", "float32", "int32"])
    args = parser.parse_args()

    dtype_map = {
        "uint8": torch.uint8,
        "float16": torch.float16,
        "float32": torch.float32,
        "int32": torch.int32,
    }
    dtype = dtype_map[args.dtype]
    elem_size = sizeof_dtype(dtype)

    if args.bytes % elem_size != 0:
        raise ValueError(f"--bytes must be a multiple of dtype size ({elem_size})")

    n_elems = args.bytes // elem_size

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    dist.init_process_group("nccl", device_id=local_rank)
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, "This benchmark expects world_size=2"

    # Enable symmetric memory NVSHMEM backend
    symm_mem.set_backend("NVSHMEM")
    symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
    dist.barrier()

    # Symmetric buffers
    dst = symm_mem.empty((n_elems,), dtype=dtype, device=f"cuda:{local_rank}")
    src = symm_mem.empty((n_elems,), dtype=dtype, device=f"cuda:{local_rank}")

    # Fill
    dst.zero_()
    # if dtype == torch.uint8:
    #     src.fill_(123)
    # elif dtype == torch.float16 or dtype == torch.float32:
    #     src.fill_(1.0)
    # else:
    #     src.fill_(7)

    # Explicit rendezvous for registrations/symmetry
    symm_mem.rendezvous(dst, dist.group.WORLD)
    symm_mem.rendezvous(src, dist.group.WORLD)
    dist.barrier()

    peer = 1 - rank

    # Kernel launch config
    BLOCK_ELEMS = args.block_bytes // elem_size
    if BLOCK_ELEMS <= 0:
        raise ValueError("block-bytes too small for chosen dtype")

    grid = (triton.cdiv(n_elems, BLOCK_ELEMS),)

    # Warmup (only rank0 sends; rank1 stays idle)
    dist.barrier()
    if rank == 0:
        for _ in range(args.warmup):
            put_chunks_kernel[grid](dst, src, n_elems, pe=peer, BLOCK=BLOCK_ELEMS)
            quiet_kernel[(1,)]()
        torch.cuda.synchronize()
    dist.barrier()

    # Timed region
    if rank == 0:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(args.iters):
            put_chunks_kernel[grid](dst, src, n_elems, pe=peer, BLOCK=BLOCK_ELEMS)
            quiet_kernel[(1,)]()
        end.record()

        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        sec = ms / 1e3

        total_bytes = args.bytes * args.iters
        gbps_dec = (total_bytes / 1e9) / sec
        gibps = (total_bytes / (1024**3)) / sec

        print(
            f"[rank0] size={args.bytes} bytes dtype={args.dtype} "
            f"iters={args.iters} time={sec:.6f}s  BW={gbps_dec:.2f} GB/s ({gibps:.2f} GiB/s)",
            flush=True,
        )

    # Optional correctness check after timing (sample a few elements on rank1)
    dist.barrier()
    if rank == 1:
        # ensure local stream is caught up (rank1 didn't launch kernels, but safe)
        torch.cuda.synchronize()
        sample = dst[:16].detach().cpu()
        print(f"[rank1] dst sample: {sample.tolist()}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

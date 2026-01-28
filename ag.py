import os
import argparse

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import torch.distributed._symmetric_memory as symm_mem
import torch.distributed._symmetric_memory._nvshmem_triton as nvshmem
from torch.distributed._symmetric_memory._nvshmem_triton import requires_nvshmem


NCCL_PAD_UNIT = 64


def print_per_rank(msg, flush=True):
    for i in range(world_size):
        if dist.get_rank() == i:
            print(msg, flush=flush)
        dist.barrier()


@requires_nvshmem
@triton.jit
def ring_all_gather_kernel(
    input_size,
    input_buffer,
    output_buffer,
    buf_ptrs,
    local_buf_ptr,
    flags,
    rank: tl.constexpr,
    node_start: tl.constexpr,
    node_end: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    nprog = tl.num_programs(axis=0)
    send_dst = (rank + 1) % world_size
    recv_src = (rank - 1 + world_size) % world_size

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input_size

    # Ring staging/receive layout inside the symmetric buffer.
    #
    # We use **world_size** contiguous buffers of size (nprog * BLOCK_SIZE):
    # - buffer 0: local input staging (padded) for step 0 send
    # - buffer (s+1): receive buffer for ring step s
    #
    # This avoids reuse/overwrite of the same receive buffer (and its signal word)
    # within a single kernel run, which can otherwise hang if some ranks start later
    # and miss an earlier signal that gets overwritten by a later step.
    buf_stride = nprog * BLOCK_SIZE

    # Load local input -> staging buffer0 (padded), and also write own rank into output
    tmp = tl.load(input_buffer + offsets, mask=mask, other=0)
    tl.store(local_buf_ptr + offsets, tmp, mask=mask)
    tl.store(output_buffer + rank * input_size + offsets, tmp, mask=mask)

    # Ring steps
    for s in tl.static_range(0, world_size - 1):
        send_base = s * buf_stride
        recv_base = (s + 1) * buf_stride
        recv_rank = (rank - 1 - s + world_size) % world_size

        # unique tag for nvshmem signals.
        send_tag = tl.full((), rank * world_size + (s + 1), tl.uint64)
        recv_tag = tl.full((), recv_src * world_size + (s + 1), tl.uint64)

        # --- Signal slot selection (per step + pid) ---
        sig_idx = (s + 1) * nprog + pid
        sig_ptr = flags + sig_idx  # symmetric address

        nvshmem.putmem_signal_block(
            dst = local_buf_ptr + recv_base + block_start,  # remote dst addr on send_dst
            src = local_buf_ptr + send_base + block_start,  # local src addr
            size_bytes = tl.full((), BLOCK_SIZE, tl.int64),
            signal = sig_ptr,  # remote signal word addr on send_dst
            sig_val = send_tag,      # signal value
            sig_op = 0,        # NVSHMEM_SIGNAL_SET=0
            pe = send_dst,
        )
        # print("before wait ", recv_tag)
        nvshmem.signal_wait_until(
            signal=sig_ptr,
            cmp=0,  # NVSHMEM_CMP_EQ = 0
            cmp_val=recv_tag,
        )
        # print("after wait ", recv_tag)

        # Now safe to read recv_base and store to output.
        #
        # Important: the recv buffer may have been previously accessed, so its lines can
        # be resident in (per-SM) caches. Remote NVSHMEM puts do not
        # necessarily invalidate those caches. Use a cache-bypassing load to reliably
        # observe remote updates after the signal wait.
        tmp = tl.load(local_buf_ptr + recv_base + offsets, mask=mask, cache_modifier=".cg")
        tl.store(output_buffer + recv_rank * input_size + offsets, tmp, mask=mask)


def main(args, rank, world_size, local_rank, local_world_size):
    # SymmMem with NVSHMEM backend
    symm_mem.set_backend("NVSHMEM")
    symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
    dist.barrier()

    buf = symm_mem.empty(args.symm_buffer_size, dtype=torch.int8, device=f"cuda:{local_rank}")
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD.group_name)
    dist.barrier()

    # --- flags ---
    # world_size * nChannels uint64 signal words (one set per ring step buffer, per pid), per rank
    flags = symm_mem.empty(world_size * args.nChannels, dtype=torch.uint64, device=f"cuda:{local_rank}")
    _flags_hdl = symm_mem.rendezvous(flags, dist.group.WORLD.group_name)
    # flags_ptrs = torch.tensor(_flags_hdl.buffer_ptrs, device=f"cuda:{local_rank}", dtype=torch.int64)
    dist.barrier()

    buf_ptrs = torch.tensor(hdl.buffer_ptrs, device=f"cuda:{local_rank}", dtype=torch.int64)

    root_node = rank // local_world_size
    node_start = root_node * local_world_size
    node_end = min(node_start + local_world_size, world_size)

    input_tensor = torch.randint(-128, 128, (args.input_size,), device=f"cuda:{local_rank}", dtype=torch.int8)
    expected = torch.empty((world_size * args.input_size,), dtype=torch.int8, device=f"cuda:{local_rank}")
    output_tensor = torch.empty_like(expected)
    dist.all_gather_into_tensor(expected, input_tensor)
    torch.cuda.synchronize()
    dist.barrier()

    BLOCK_SIZE = triton.cdiv(args.input_size, args.nChannels * NCCL_PAD_UNIT) * NCCL_PAD_UNIT
    assert BLOCK_SIZE < 1024 * 16, f"ERROR: {BLOCK_SIZE} > 1024 * 16, very slow to compile."
    need_bytes = world_size * args.nChannels * BLOCK_SIZE
    assert args.symm_buffer_size >= need_bytes, f"symm buffer too small: need >= {need_bytes} bytes"

    # flags must be reset before each kernel launch.
    # Otherwise `signal_wait_until(..., EQ, tag)` can spuriously succeed if the signal words
    # contain garbage (or tags from a previous launch).
    flags.zero_()
    # Make sure the memset is complete on all ranks before any peer starts
    # signaling into these words.
    torch.cuda.synchronize()
    dist.barrier()

    print("before the kernel launch")
    ring_all_gather_kernel[(args.nChannels,)](
        args.input_size,
        input_tensor.view(-1),
        output_tensor.view(-1),
        buf_ptrs,
        buf,
        flags,
        rank=rank,
        node_start=node_start,
        node_end=node_end,
        world_size=world_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=args.nWarps,
    )

    torch.cuda.synchronize()
    dist.barrier()

    print_per_rank(
        f"[rank{rank}] input_tensor[:16]   = {input_tensor[:16].detach().cpu().tolist()}\n"
        f"[rank{rank}] output_tensor[:16]  = {output_tensor[:16].detach().cpu().tolist()}\n"
        f"[rank{rank}] expected[:16]       = {expected[:16].detach().cpu().tolist()}"
    )
    if not torch.equal(output_tensor, expected):
        mism = (output_tensor != expected).nonzero(as_tuple=False)
        i = int(mism[0].item())
        got = int(output_tensor[i].item())
        exp = int(expected[i].item())
        raise AssertionError(
            f"output_tensor mismatch at flat index {i}: got {got}, expected {exp}. "
            f"(This index is in gathered rank {i // args.input_size}, "
            f"offset {i % args.input_size})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="All-gather kernel: load/store intra-node + NVSHMEM inter-node")
    parser.add_argument("--symm_buffer_size", type=int, default=1024 * 1024 * 1024, help="Number of bytes in symmetric buffer")
    parser.add_argument("--input_size", type=int, default=1024, help="Number of bytes in input tensor")
    parser.add_argument("--nChannels", type=int, default=1, help="Number of channels for all-gather")
    parser.add_argument("--nWarps", type=int, default=1, help="Number of warps inside each channel(thread_block)")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    assert local_rank == rank % local_world_size

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=local_rank)

    main(args, rank, world_size, local_rank, local_world_size)

    free, total = torch.cuda.mem_get_info()
    print_per_rank(f"[rank{rank}] CUDA mem used={(total - free) / 2**20:.1f} MiB free={free / 2**20:.1f} MiB")

    dist.destroy_process_group()

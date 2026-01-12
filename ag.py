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


@requires_nvshmem
@triton.jit
def all_gather_kernel(
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
    recv_src = (rank - 1 + world_size) % world_size  # (not used directly; just conceptual)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # ping-pong layout inside the symmetric buffer
    buf0_base = 0
    buf1_base = nprog * BLOCK_SIZE

    # Load local input -> buf0, and also write own rank into output
    mask = offsets < input_size
    tmp = tl.load(input_buffer + offsets, mask=mask, other=0)
    tl.store(local_buf_ptr + buf0_base + offsets, tmp, mask=mask)
    tl.store(output_buffer + rank * input_size + offsets, tmp, mask=mask)

    # Ring steps
    for s in tl.static_range(0, world_size - 1):
        send_base = buf0_base if (s & 1) == 0 else buf1_base
        recv_base = buf1_base if (s & 1) == 0 else buf0_base

        recv_rank = (rank - 1 - s + world_size) % world_size

        # --- Signal slot selection (per parity + pid) ---
        # recv_base parity: s=0 -> recv_base=buf1 -> parity=1; s=1 -> buf0 -> parity=0; ...
        recv_parity = (s + 1) & 1
        sig_idx = recv_parity * nprog + pid
        sig_ptr = flags + sig_idx   # symmetric address (valid for remote PEs too)

        # unique tag within this kernel launch (requires flags to be reset pre-launch)
        tag = tl.full((), s + 1, tl.uint64)

        # --- SEND (payload + signal) ---
        # Note: BLOCK_SIZE is bytes here because local_buf_ptr is int8
        nvshmem.putmem_signal_block(
            local_buf_ptr + recv_base + block_start,  # remote dst addr on send_dst
            local_buf_ptr + send_base + block_start,  # local src addr
            tl.full((), BLOCK_SIZE, tl.int64),
            sig_ptr,                                  # remote signal word addr on send_dst
            tag,                                      # signal value
            0, # NVSHMEM_SIGNAL_SET=0
            send_dst,
        )

        # --- WAIT for receive completion (from recv_src) ---
        # recv_src will write into *my* recv_base and set *my* sig_ptr to tag
        nvshmem.signal_wait_until(sig_ptr, 0, tag) # NVSHMEM_CMP_EQ = 0

        # Now safe to read recv_base and store to output
        tmp = tl.load(local_buf_ptr + recv_base + offsets, mask=mask, other=0)
        tl.store(output_buffer + recv_rank * input_size + offsets, tmp, mask=mask)


def main():
    # SymmMem with NVSHMEM backend
    symm_mem.set_backend("NVSHMEM")
    symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
    dist.barrier()

    buf = symm_mem.empty(args.symm_buffer_size, dtype=torch.int8, device=f"cuda:{local_rank}")
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD.group_name)
    dist.barrier()

   # --- NEW: flags ---
    flags = symm_mem.empty(2 * args.nChannels, dtype=torch.uint64, device=f"cuda:{local_rank}")
    flags_hdl = symm_mem.rendezvous(flags, dist.group.WORLD.group_name)
    dist.barrier()
    flags.zero_()

    buf_ptrs = torch.tensor(hdl.buffer_ptrs, device=f"cuda:{local_rank}", dtype=torch.int64)

    root_node = rank // local_world_size
    node_start = root_node * local_world_size
    node_end = min(node_start + local_world_size, world_size)

    input_tensor = torch.full((args.input_size,), rank, dtype=torch.int8, device=f"cuda:{local_rank}")
    output_tensor = torch.empty((world_size * args.input_size, ), dtype=torch.int8, device=f"cuda:{local_rank}")

    BLOCK_SIZE = triton.cdiv(args.input_size, args.nChannels * NCCL_PAD_UNIT) * NCCL_PAD_UNIT
    assert BLOCK_SIZE < 1024 * 16, f"ERROR: {BLOCK_SIZE} > 1024 * 16, very slow to compile."
    need_bytes = 2 * args.nChannels * BLOCK_SIZE
    assert args.symm_buffer_size >= need_bytes, f"symm buffer too small: need >= {need_bytes} bytes"
    all_gather_kernel[(args.nChannels,)](
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

    dist.barrier()
    torch.cuda.synchronize()

    print(f"rank{rank} output_tensor={output_tensor}")
    expected = (
        torch.arange(world_size, device=output_tensor.device, dtype=torch.int8)
        .repeat_interleave(args.input_size)
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
    parser.add_argument("--symm_buffer_size", type=int, default=1024 * 1024* 1024, help="Number of bytes in symmetric buffer")
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

    main()

    free, total = torch.cuda.mem_get_info()
    print(f"[rank{rank}] CUDA mem used={(total-free)/2**20:.1f} MiB free={free/2**20:.1f} MiB", flush=True)

    dist.destroy_process_group()

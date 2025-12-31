

### Environment Setup

Triton version mismatch error:
```
File "/usr/local/lib/python3.11/dist-packages/torch/distributed/_symmetric_memory/_nvshmem_triton.py", line 241, in requires_nvshmem
    triton.knobs.runtime.jit_post_compile_hook = _nvshmem_init_hook
    ^^^^^^^^^^^^
AttributeError: module 'triton' has no attribute 'knobs'"
```

Triton version too low, upgrade to 3.5.1. Also delete the local triton: `rm -rf /home/tiger/.local/lib/python3.11/site-packages/triton/`


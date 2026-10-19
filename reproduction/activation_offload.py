"""Lossless saved-tensor CPU offload. No batch splitting or gradient accumulation."""
from contextlib import contextmanager
import weakref
import torch


@contextmanager
def offload_saved_tensors(max_host_gib=40):
    cache = {}
    host_bytes = 0

    def pack(tensor):
        nonlocal host_bytes
        # Keep parameters and small tensors on-device; copy activations bit-for-bit.
        if tensor.device.type != 'cuda' or tensor.numel() * tensor.element_size() < 1024**2:
            return tensor
        if tensor.is_leaf and tensor.requires_grad:
            return tensor
        # A raw data_ptr can be reused by the CUDA allocator during this forward.
        # Verify Python identity with a weakref instead of treating reused storage
        # as an identical activation.
        key = (id(tensor), tensor._version)
        entry = cache.get(key)
        if entry is None or entry[0]() is not tensor:
            amount = tensor.numel() * tensor.element_size()
            if host_bytes + amount > max_host_gib * 2**30:
                raise MemoryError(f'Activation offload exceeded {max_host_gib} GiB host limit')
            import psutil
            if psutil.virtual_memory().available < amount + 6 * 2**30:
                raise MemoryError('Less than 6 GiB available host memory; stopping offload')
            host = torch.empty_like(tensor, device='cpu', pin_memory=True,
                                    memory_format=torch.preserve_format)
            host.copy_(tensor.detach())
            cache[key] = (weakref.ref(tensor), host)
            host_bytes += amount
        return (tensor.device, cache[key][1])

    def unpack(packed):
        if isinstance(packed, tuple):
            device, host = packed
            return host.to(device)
        return packed

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield


def enable_model_offload():
    from src.zoo.dfine.dfine import DFINE
    original = DFINE.forward

    def forward(self, *args, **kwargs):
        if self.training and torch.is_grad_enabled():
            with offload_saved_tensors():
                return original(self, *args, **kwargs)
        return original(self, *args, **kwargs)

    DFINE.forward = forward

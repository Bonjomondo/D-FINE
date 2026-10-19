"""Recompute individual blocks while preserving exactly one BatchNorm buffer update."""
from contextlib import contextmanager, nullcontext
import torch
from torch.utils.checkpoint import checkpoint


@contextmanager
def preserve_normalization_buffers(module):
    swapped = []
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            for name in ('running_mean', 'running_var', 'num_batches_tracked'):
                value = child._buffers.get(name)
                if value is not None:
                    swapped.append((child, name, value))
                    child._buffers[name] = value.clone()
    try:
        yield
    finally:
        for child, name, original in swapped:
            child._buffers[name] = original


def enable_activation_checkpointing():
    from src.nn.backbone.hgnetv2 import HG_Block
    from src.zoo.dfine.hybrid_encoder import RepNCSPELAN4, TransformerEncoderLayer
    from src.zoo.dfine.dfine_decoder import TransformerDecoderLayer
    originals = {}

    def wrap(original):
        def forward(self, *args, **kwargs):
            if self.training and torch.is_grad_enabled():
                return checkpoint(original, self, *args, use_reentrant=False,
                                  context_fn=lambda: (nullcontext(), preserve_normalization_buffers(self)),
                                  **kwargs)
            return original(self, *args, **kwargs)
        return forward

    for cls in (HG_Block, RepNCSPELAN4, TransformerEncoderLayer, TransformerDecoderLayer):
        originals[cls] = cls.forward
        cls.forward = wrap(cls.forward)

    def restore():
        for cls, original in originals.items():
            cls.forward = original
    return restore

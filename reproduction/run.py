"""Run upstream full-batch training with validated block activation recomputation."""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
mode = os.environ.get('DFINE_MEMORY_MODE', 'checkpoint')
if mode == 'checkpoint':
    from activation_checkpoint import enable_activation_checkpointing
    enable_activation_checkpointing()
elif mode == 'offload':
    from activation_offload import enable_model_offload
    enable_model_offload()
else:
    raise ValueError(mode)
print(f'Activation memory strategy: {mode}; batch size and optimizer unchanged.', flush=True)
runpy.run_path(str(ROOT / 'train.py'), run_name='__main__')

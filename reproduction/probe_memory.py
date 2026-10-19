"""Measure an actual AMP optimizer step; synthetic inputs are NOT AP reproduction."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from src.core import YAMLConfig
from src.misc.dist_utils import setup_seed

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--size', type=int, default=800)
parser.add_argument('--objects', type=int, default=20)
parser.add_argument('--model', default='l', choices=['s', 'm', 'l', 'x'])
parser.add_argument('--offload', action='store_true')
parser.add_argument('--checkpoint', action='store_true')
args = parser.parse_args()
assert not (args.offload and args.checkpoint)
if args.offload:
    from activation_offload import enable_model_offload
    enable_model_offload()
if args.checkpoint:
    from activation_checkpoint import enable_activation_checkpointing
    enable_activation_checkpointing()
setup_seed(0)
torch.set_num_threads(8)
cfg = YAMLConfig(str(ROOT / 'reproduction/configs' / f'paper_{args.model}_coco.yml'))
model = cfg.model.cuda().train()
criterion = cfg.criterion.cuda()
optimizer = cfg.optimizer
scaler = torch.amp.GradScaler('cuda')
samples = torch.rand(args.batch, 3, args.size, args.size, device='cuda')
targets = [{'labels': torch.randint(0, 80, (args.objects,), device='cuda'),
            'boxes': torch.cat([torch.rand(args.objects, 2, device='cuda') * .6 + .2,
                                torch.rand(args.objects, 2, device='cuda') * .15 + .02], dim=1)}
           for _ in range(args.batch)]
result = {'batch_size': args.batch, 'image_size': args.size, 'objects_per_image': args.objects,
          'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
          'synthetic_memory_probe_only': True}
result['activation_offload'] = args.offload
result['block_activation_checkpointing'] = args.checkpoint
start = time.monotonic()
try:
    with torch.autocast('cuda'):
        outputs = model(samples, targets=targets)
    losses = criterion(outputs, targets, epoch=0, step=0, global_step=0, epoch_step=3696)
    loss = sum(losses.values())
    assert torch.isfinite(loss), loss
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    result.update(status='passed', loss=float(loss.detach()))
except torch.cuda.OutOfMemoryError as exc:
    result.update(status='out_of_memory', error=str(exc))
except MemoryError as exc:
    result.update(status='host_memory_limit', error=str(exc))
result.update(seconds=time.monotonic()-start,
              peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
              peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
suffix = '_offload' if args.offload else '_checkpoint' if args.checkpoint else ''
path = ROOT / 'reproduction' / f'memory_{args.model}_b{args.batch}_s{args.size}{suffix}.json'
path.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
sys.exit(0 if result['status'] == 'passed' else 2)

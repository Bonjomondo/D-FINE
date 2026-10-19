"""One real-data training step on val images; never reported as COCO training/AP."""
import json
import argparse
import sys
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core import YAMLConfig
from src.misc.dist_utils import setup_seed
from src.solver.det_engine import train_one_epoch
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', action='store_true')
args = parser.parse_args()

torch.set_num_threads(8)
setup_seed(0)
if args.checkpoint:
    from activation_checkpoint import enable_activation_checkpointing
    enable_activation_checkpointing()
else:
    from activation_offload import enable_model_offload
    enable_model_offload()
cfg = YAMLConfig(str(ROOT / 'reproduction/configs/paper_l_coco.yml'), use_amp=True)
# Validation images are used only to check the official training engine before
# train2017 finishes downloading. The formal training uses train2017 exclusively.
data = cfg.yaml_cfg['train_dataloader']['dataset']
for key in ('img_folder', 'ann_file'):
    data[key] = cfg.yaml_cfg['val_dataloader']['dataset'][key]
cfg.yaml_cfg['train_dataloader']['num_workers'] = 2
model = cfg.model.cuda()
criterion = cfg.criterion.cuda()
batch = next(iter(cfg.train_dataloader))
assert batch[0].shape[0] == 32
start = time.monotonic()
initial_scale = cfg.scaler.get_scale()
for attempt in range(12):
    stats = train_one_epoch(model, criterion, [batch], cfg.optimizer, torch.device('cuda'),
                            0, False, max_norm=.1, scaler=cfg.scaler,
                            ema=cfg.ema.to('cuda'), lr_warmup_scheduler=cfg.lr_warmup_scheduler,
                            print_freq=1, epochs=74)
    # Native AMP may skip its first updates while adapting the loss scale.
    # Do not call a skipped update a completed optimizer-step verification.
    if cfg.optimizer.state:
        break
assert cfg.optimizer.state, 'AMP did not complete any finite optimizer update'
result = {'status': 'passed', 'scope': 'val_images_training_engine_smoke_only',
          'formal_training': False, 'batch_size': 32, 'image_size': list(batch[0].shape[-2:]),
          'seconds': time.monotonic()-start, 'stats': stats,
          'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30}
result.update(attempted_steps=attempt+1, initial_amp_scale=initial_scale,
              final_amp_scale=cfg.scaler.get_scale(), finite_optimizer_update_verified=True)
result['seconds_per_attempt'] = result['seconds'] / result['attempted_steps']
result['memory_strategy'] = 'block_activation_checkpointing' if args.checkpoint else 'activation_offload'
filename = 'checkpoint_real_batch_smoke.json' if args.checkpoint else 'real_batch_smoke.json'
(ROOT / 'reproduction' / filename).write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)

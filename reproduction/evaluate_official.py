"""Evaluate the author's L checkpoint on all 5,000 COCO val2017 images."""
import hashlib
import argparse
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core import YAMLConfig
from src.misc.dist_utils import setup_seed
from src.solver import TASKS
from src.solver.det_engine import evaluate

torch.set_num_threads(8)
setup_seed(0)
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=Path, default=ROOT / 'weight/official/dfine_l_coco.pth')
parser.add_argument('--self-trained', action='store_true')
parser.add_argument('--config', type=Path, default=ROOT / 'reproduction/configs/paper_l_coco.yml')
parser.add_argument('--dataset', choices=['coco2017', 'objects365'], default='coco2017')
parser.add_argument('--result-tag')
parser.add_argument('--paper-ap', type=float, default=54.0)
args = parser.parse_args()
checkpoint = args.checkpoint.resolve()
tag = 'self_trained' if args.self_trained else 'official'
result_tag = args.result_tag or f'{tag}_l_coco'
cfg = YAMLConfig(str(args.config.resolve()),
                 resume=str(checkpoint), device='cuda', use_amp=False,
                 output_dir=str(ROOT / f'output/{result_tag}_eval'))
cfg.yaml_cfg['HGNetv2']['pretrained'] = False
solver = TASKS[cfg.yaml_cfg['task']](cfg)
solver.eval()
assert len(solver.val_dataloader.dataset) == 5000
module = solver.ema.module if solver.ema else solver.model
stats, evaluator = evaluate(module, solver.criterion, solver.postprocessor,
                            solver.val_dataloader, solver.evaluator, solver.device,
                            epoch=-1, use_wandb=False)
names = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
         'AR_1', 'AR_10', 'AR_100', 'AR_small', 'AR_medium', 'AR_large']
result = {'scope': 'self_trained_checkpoint_evaluation' if args.self_trained else 'official_checkpoint_evaluation_only',
          'self_trained': args.self_trained,
          'images': 5000, 'model': 'D-FINE-L', 'checkpoint': str(checkpoint),
          'dataset': args.dataset,
          'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
          'paper_AP_percent': args.paper_ap if args.dataset == 'coco2017' else None,
          'metrics_percent': dict(zip(names, [v*100 for v in stats['coco_eval_bbox']])),
          'torch': torch.__version__, 'gpu': torch.cuda.get_device_name()}
(ROOT / f'reproduction/{result_tag}_metrics.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)

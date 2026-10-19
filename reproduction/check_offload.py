"""Compare original vs offloaded AMP loss, predictions, and every model gradient."""
import json
import argparse
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core import YAMLConfig
from src.misc.dist_utils import setup_seed
from activation_offload import offload_saved_tensors
from contextlib import nullcontext
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', action='store_true')
args = parser.parse_args()

torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
setup_seed(0)
cfg = YAMLConfig(str(ROOT / 'reproduction/configs/paper_l_coco.yml'))
model = cfg.model.cuda().train()
criterion = cfg.criterion.cuda()
state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
samples = torch.rand(2, 3, 640, 640, device='cuda')
targets = [{'labels': torch.randint(0, 80, (10,), device='cuda'),
            'boxes': torch.cat([torch.rand(10, 2, device='cuda')*.6+.2,
                                torch.rand(10, 2, device='cuda')*.15+.02], dim=1)} for _ in range(2)]
runs = []
for offload in (False, True, False):
    model.load_state_dict(state)
    model.zero_grad(set_to_none=True)
    setup_seed(123)
    restore = None
    if offload and args.checkpoint:
        from activation_checkpoint import enable_activation_checkpointing
        restore = enable_activation_checkpointing()
    with offload_saved_tensors() if offload and not args.checkpoint else nullcontext():
        with torch.autocast('cuda'):
            outputs = model(samples, targets=targets)
    if restore:
        restore()
    losses = criterion(outputs, targets, epoch=0, step=0, global_step=0, epoch_step=3696)
    loss = sum(losses.values())
    (loss * 128).backward()
    torch.cuda.synchronize()
    runs.append({'loss': loss.detach().cpu(),
                 'boxes': outputs['pred_boxes'].detach().cpu(),
                 'logits': outputs['pred_logits'].detach().cpu(),
                 'buffers': {k: v.detach().cpu().clone() for k, v in model.named_buffers()},
                 'gradients': {k: v.grad.detach().cpu().clone() / 128 for k, v in model.named_parameters()
                               if v.grad is not None}})
    del outputs, losses, loss
    torch.cuda.empty_cache()
for name in ('loss', 'boxes', 'logits'):
    torch.testing.assert_close(runs[0][name], runs[1][name], rtol=1e-5, atol=1e-6)
for name, buffer in runs[0]['buffers'].items():
    torch.testing.assert_close(buffer, runs[1]['buffers'][name], rtol=0, atol=0, msg=name)
assert runs[0]['gradients'].keys() == runs[1]['gradients'].keys()
max_gradient_error = 0.
baseline_error = 0.
error_sq = baseline_sq = reference_sq = 0.
failures = []
for name, gradient in runs[0]['gradients'].items():
    candidate = runs[1]['gradients'][name]
    baseline = runs[2]['gradients'][name]
    if not torch.allclose(gradient, candidate, rtol=1e-4, atol=1e-5):
        failures.append(name)
    max_gradient_error = max(max_gradient_error, (gradient-candidate).abs().max().item())
    baseline_error = max(baseline_error, (gradient-baseline).abs().max().item())
    error_sq += (gradient-candidate).double().square().sum().item()
    baseline_sq += (gradient-baseline).double().square().sum().item()
    reference_sq += gradient.double().square().sum().item()
relative_error = (error_sq/reference_sq)**.5
baseline_relative_error = (baseline_sq/reference_sq)**.5
# CUDA grid_sample backward is nondeterministic even in the upstream model.
# Require offload errors to remain below twice the measured original-repeat noise,
# with a hard 1% relative-L2 cap. Prediction/loss checks above remain strict.
within_repeat_noise = (relative_error <= max(1e-5, 2*baseline_relative_error)
                       and relative_error < .01
                       and max_gradient_error <= max(1e-5, 2*baseline_error))
result = {'status': 'passed' if within_repeat_noise else 'failed', 'amp': True, 'batch': 2, 'image_size': 640,
          'loss_original': runs[0]['loss'].item(), 'loss_offloaded': runs[1]['loss'].item(),
          'gradient_tensors_compared': len(runs[0]['gradients']),
          'max_absolute_gradient_error': max_gradient_error,
          'baseline_repeat_max_absolute_gradient_error': baseline_error,
          'gradient_relative_l2_error': relative_error,
          'baseline_repeat_gradient_relative_l2_error': baseline_relative_error,
          'acceptance': 'strict prediction/loss equality tolerance; gradient errors <= 2x upstream repeat noise and <1% relative L2',
          'nondeterministic_upstream_operator': 'grid_sampler_2d_backward_cuda',
          'gradient_tensors_outside_tolerance': failures,
          'gradient_tolerance': {'rtol': 1e-4, 'atol': 1e-5}}
result['memory_strategy'] = 'block_activation_checkpointing' if args.checkpoint else 'activation_offload'
result['normalization_buffers_identical'] = True
filename = 'checkpoint_equivalence.json' if args.checkpoint else 'offload_equivalence.json'
(ROOT / 'reproduction' / filename).write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
sys.exit(0 if result['status'] == 'passed' else 2)

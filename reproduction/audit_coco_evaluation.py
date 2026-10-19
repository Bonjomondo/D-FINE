"""Export full-val predictions and compare native COCO API against faster-coco-eval."""
import csv
import gzip
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core import YAMLConfig
from src.misc.dist_utils import setup_seed
from src.solver import TASKS
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from faster_coco_eval import COCO as FastCOCO, COCOeval_faster

OUT = ROOT / 'reproduction/evaluation_audit'
OUT.mkdir(exist_ok=True)
torch.set_num_threads(8)
setup_seed(0)
checkpoint = ROOT / 'output/paper_l_coco_seed0/best_stg2.pth'
cfg = YAMLConfig(str(ROOT / 'reproduction/configs/paper_l_coco.yml'), resume=str(checkpoint),
                 device='cuda', use_amp=False, output_dir=str(OUT))
cfg.yaml_cfg['HGNetv2']['pretrained'] = False
solver = TASKS[cfg.yaml_cfg['task']](cfg)
solver.eval()
assert solver.ema is not None
model = solver.ema.module.eval()
loader = solver.val_dataloader
assert len(loader.dataset) == 5000
records = []
image_ids = []
native = solver.evaluator
with torch.inference_mode():
    for step, (samples, targets) in enumerate(loader):
        sizes = torch.stack([t['orig_size'] for t in targets]).to('cuda')
        outputs = model(samples.to('cuda'))
        results = solver.postprocessor(outputs, sizes)
        batch = {int(t['image_id'].item()): result for t, result in zip(targets, results)}
        records.extend(native.prepare_for_coco_detection(batch))
        image_ids.extend(batch)
        if step % 50 == 0:
            print(f'Inference {step}/{len(loader)}; predictions={len(records)}', flush=True)
assert len(image_ids) == len(set(image_ids)) == 5000
predictions = OUT / 'self_trained_predictions.json.gz'
with gzip.open(predictions, 'wt', compresslevel=3) as stream:
    json.dump(records, stream, separators=(',', ':'))
annotation = Path(cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'])
gt = COCO(str(annotation))
standard = COCOeval(gt, gt.loadRes(records), 'bbox')
fast_gt = FastCOCO(str(annotation))
fast = COCOeval_faster(fast_gt, fast_gt.loadRes(records), 'bbox', separate_eval=True)
for evaluation in (standard, fast):
    evaluation.params.imgIds = sorted(image_ids)
for name in ('iouThrs', 'recThrs', 'maxDets', 'areaRng', 'catIds'):
    np.testing.assert_array_equal(getattr(standard.params, name), getattr(fast.params, name))
for label, evaluation in [('pycocotools', standard), ('faster-coco-eval', fast)]:
    print('Evaluating', label, flush=True)
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
names = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
         'AR_1', 'AR_10', 'AR_100', 'AR_small', 'AR_medium', 'AR_large']
max_delta = float(np.abs(standard.stats[:12]-fast.stats[:12]).max())
np.testing.assert_allclose(standard.stats[:12], fast.stats[:12], rtol=0, atol=1e-9)
stored = json.loads((ROOT / 'reproduction/self_trained_l_coco_metrics.json').read_text())
stored_stats = np.array([stored['metrics_percent'][name]/100 for name in names])
np.testing.assert_allclose(standard.stats[:12], stored_stats, rtol=0, atol=1e-5)
precision = standard.eval['precision']
by_iou = {}
for i, threshold in enumerate(standard.params.iouThrs):
    values = precision[i, :, :, 0, -1]
    by_iou[f'{threshold:.2f}'] = float(values[values >= 0].mean()*100)
with (OUT / 'per_category_ap.csv').open('w') as stream:
    writer = csv.writer(stream)
    writer.writerow(['category_id', 'category_name', 'AP', 'AP50', 'AP75'])
    for k, category_id in enumerate(standard.params.catIds):
        category = gt.cats[category_id]
        subsets = [precision[:, :, k, 0, -1], precision[0, :, k, 0, -1], precision[5, :, k, 0, -1]]
        values = [float(p[p >= 0].mean()*100) for p in subsets]
        writer.writerow([category_id, category['name'], *values])
result = dict(status='passed', images=5000, categories=80, predictions=len(records),
              checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              weights='EMA', inference_dtype='float32',
              input_size=[640,640], test_time_augmentation=False, nms=False, top_k=300,
              iou_thresholds=standard.params.iouThrs.tolist(), recall_points=len(standard.params.recThrs),
              max_detections=standard.params.maxDets, area_ranges=standard.params.areaRng,
              area_range_labels=standard.params.areaRngLbl,
              versions={p:importlib.metadata.version(p) for p in ['pycocotools','faster-coco-eval','torch','torchvision']},
              standard_metrics_percent=dict(zip(names, (standard.stats[:12]*100).tolist())),
              faster_metrics_percent=dict(zip(names, (fast.stats[:12]*100).tolist())),
              max_backend_difference_AP_points=max_delta*100,
              max_stored_result_difference_AP_points=float(np.abs(standard.stats[:12]-stored_stats).max()*100),
              AP_by_IoU_percent=by_iou, prediction_file=str(predictions))
(OUT / 'evaluation_audit.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)

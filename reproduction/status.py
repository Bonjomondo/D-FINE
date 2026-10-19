"""Print pipeline stage, verified data, current download progress and measured AP."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / 'reproduction'
DATA = ROOT.parent / 'datasets/coco2017'
if (REP / 'status.json').exists():
    payload = json.loads((REP / 'status.json').read_text())
    obj_status = REP / 'objects365/download_status.json'
    if obj_status.exists():
        payload['objects365'] = json.loads(obj_status.read_text())['stage']
    scope_file = REP / 'scope.json'
    if scope_file.exists():
        scope = json.loads(scope_file.read_text())
        payload['selected_experiment'] = scope['selected_experiment']
        if not scope.get('include_objects365', False):
            payload['objects365'] = 'deferred_by_user'
            payload.pop('remaining_paper_experiments', None)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
for name, size in [('annotations_trainval2017.zip', 252907541),
                   ('val2017.zip', 815585330), ('train2017.zip', 19336861798)]:
    verified = (DATA / (name + '.verified.json')).exists()
    archive = DATA / 'archives' / name
    parts = DATA / 'archives' / (name + '.parts16m')
    downloaded = archive.stat().st_size if archive.exists() else sum(p.stat().st_size for p in parts.glob('*') if p.is_file())
    print(f'{name}: {downloaded/size:.1%}, verified={verified}')
for filename in ('official_l_coco_metrics.json', 'self_trained_l_coco_metrics.json'):
    path = REP / filename
    if path.exists():
        result = json.loads(path.read_text())
        print(filename, 'AP=', result['metrics_percent']['AP'], 'self_trained=', result['self_trained'])

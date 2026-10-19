"""Check full official COCO split sizes and every referenced image path."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'datasets/coco2017'
result = {}
for split, expected in [('train2017', 118287), ('val2017', 5000)]:
    ann = json.loads((DATA / 'annotations' / f'instances_{split}.json').read_text())
    assert len(ann['images']) == expected, (split, len(ann['images']))
    assert len(ann['categories']) == 80
    missing = [im['file_name'] for im in ann['images'] if not (DATA / split / im['file_name']).is_file()]
    assert not missing, (split, missing[:10])
    result[split] = {'images': expected, 'annotations': len(ann['annotations']), 'missing_images': 0}
(ROOT / 'reproduction/data_verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))

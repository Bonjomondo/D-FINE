"""Compare streaming preprocessing against the original scripts, including image bytes."""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from PIL import Image
from preprocess_objects365 import preprocess

ROOT = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='dfine-objects365-check-', dir='/tmp') as temporary:
    work = Path(temporary)
    original, candidate = work / 'original', work / 'streaming'
    for split, first_id in [('train', 100), ('val', 200)]:
        images, annotations = [], []
        sizes = [(1001, 667), (400, 1200), (321, 233), (640, 640), (900, 710)]
        for offset, (width, height) in enumerate(sizes):
            image_id = first_id + offset
            name = f'images/v1/patch0/image_{image_id}.jpg'
            path = original / split / name
            path.parent.mkdir(parents=True, exist_ok=True)
            # One upstream missing training image exercises the original skip policy.
            if not (split == 'train' and offset == 4):
                Image.new('RGB', (width, height), (30+offset, 80, 120)).save(path)
            images.append(dict(id=image_id, file_name=name, width=width, height=height, license=5))
            annotations.append(dict(id=image_id, image_id=image_id, category_id=1, area=310.1,
                                    bbox=[3.1, 6.2, 17.3, 18.4], orig_size=[width, height],
                                    size=[width, height], iscrowd=0))
        data = dict(images=images, annotations=annotations,
                    categories=[dict(id=1, name='object'), dict(id=2, name='other')])
        (original / split / f'zhiyuan_objv2_{split}.json').write_text(json.dumps(data))
    shutil.copytree(original, candidate)
    shutil.copytree(original / 'val/images/v1', original / 'train/images_from_val/v1')
    subprocess.run([sys.executable, str(ROOT / 'tools/dataset/remap_obj365.py'),
                    '--base_dir', str(original), '--new_val_size', '2'],
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, str(ROOT / 'tools/dataset/resize_obj365.py'),
                    '--base_dir', str(original), '--num_workers', '2'],
                   check=True, stdout=subprocess.DEVNULL)
    report = preprocess(candidate, val_size=2, workers=2)
    image_count = 0
    for split in ('train', 'val'):
        name = f'{split}/new_zhiyuan_objv2_{split}_resized.json'
        expected = json.loads((original / name).read_text())
        actual = json.loads((candidate / name).read_text())
        assert actual == expected, split
        for image in actual['images']:
            filename = split + '/' + image['file_name']
            assert (original / filename).read_bytes() == (candidate / filename).read_bytes(), filename
            image_count += 1
    result = dict(status='passed', json_semantics_identical=True, image_bytes_identical=True,
                  checked_images=image_count, missing_image_skip_identical=True,
                  reference_scripts=['tools/dataset/remap_obj365.py', 'tools/dataset/resize_obj365.py'])
    (ROOT / 'reproduction/objects365/preprocessing_equivalence.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

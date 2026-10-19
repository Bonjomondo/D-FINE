"""Keep upstream configs intact; override paper schedules and local data paths."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'datasets' / 'coco2017'
OUT = ROOT / 'reproduction' / 'configs'
OUT.mkdir(parents=True, exist_ok=True)
for model, epochs in [('s', 124), ('m', 124), ('l', 74), ('x', 74)]:
    config = {
        '__include__': [f'../../configs/dfine/dfine_hgnetv2_{model}_coco.yml'],
        'epochs': epochs,
        'output_dir': str(ROOT / 'output' / f'paper_{model}_coco_seed0'),
        'HGNetv2': {'local_model_dir': str(ROOT / 'weight' / 'hgnetv2') + '/'},
        'train_dataloader': {'dataset': {
            'img_folder': str(DATA / 'train2017') + '/',
            'ann_file': str(DATA / 'annotations' / 'instances_train2017.json')},
            'total_batch_size': 32},
        'val_dataloader': {'dataset': {
            'img_folder': str(DATA / 'val2017') + '/',
            'ann_file': str(DATA / 'annotations' / 'instances_val2017.json')},
            # Does not change the set of evaluated images or metrics.
            'total_batch_size': 16},
    }
    (OUT / f'paper_{model}_coco.yml').write_text(yaml.safe_dump(config, sort_keys=False))

OBJECTS = ROOT.parent / 'datasets' / 'objects365'
for model, pretrain_epochs, finetune_epochs in [('s', 28, 58), ('m', 29, 49), ('l', 21, 32), ('x', 21, 31)]:
    for task, epochs in [('obj365', pretrain_epochs), ('obj2coco', finetune_epochs)]:
        config = {
            '__include__': [f'../../configs/dfine/objects365/dfine_hgnetv2_{model}_{task}.yml'],
            'epochs': epochs,
            'output_dir': str(ROOT / 'output' / f'paper_{model}_{task}_seed0'),
            'HGNetv2': {'local_model_dir': str(ROOT / 'weight/hgnetv2') + '/'},
            'train_dataloader': {'total_batch_size': 32, 'dataset': {}},
            'val_dataloader': {'total_batch_size': 16, 'dataset': {}},
        }
        for loader, split in [('train_dataloader', 'train'), ('val_dataloader', 'val')]:
            if task == 'obj365':
                dataset = {'img_folder': str(OBJECTS / split) + '/',
                           'ann_file': str(OBJECTS / split / f'new_zhiyuan_objv2_{split}_resized.json')}
            else:
                coco_split = split + '2017'
                dataset = {'img_folder': str(DATA / coco_split) + '/',
                           'ann_file': str(DATA / 'annotations' / f'instances_{coco_split}.json')}
            config[loader]['dataset'] = dataset
        (OUT / f'paper_{model}_{task}.yml').write_text(yaml.safe_dump(config, sort_keys=False))
print(OUT)

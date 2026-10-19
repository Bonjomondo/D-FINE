"""Prepare the complete original v2 data; queue paper L pretraining after COCO."""
import datetime
import fcntl
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / 'reproduction'
DATA = ROOT.parent / 'datasets/objects365'
STATE = REP / 'objects365/pipeline_status.json'
ENV = dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTHONUNBUFFERED='1',
           TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1', DFINE_MEMORY_MODE='checkpoint')
lock = (REP / 'objects365/pipeline.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def state(stage, **details):
    payload = dict(stage=stage, updated_at=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                   pid=os.getpid(), model='D-FINE-L', batch_size=32,
                   pretraining_epochs=21, finetuning_epochs=32, **details)
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(STATE)
    print(json.dumps(payload), flush=True)


def run(arguments, log_name):
    command = [str(ROOT / '.venv/bin/python'), '-u', *map(str, arguments)]
    print('Running:', command, flush=True)
    with (REP / 'logs' / log_name).open('a') as output:
        subprocess.run(command, cwd=ROOT, env=ENV, stdout=output, stderr=subprocess.STDOUT, check=True)


def train(task, tuning=None):
    output = ROOT / f'output/paper_l_{task}_seed0'
    command = ['reproduction/run.py', '-c', f'reproduction/configs/paper_l_{task}.yml',
               '--use-amp', '--seed=0', '-d', 'cuda']
    if (output / 'last.pth').exists():
        command.extend(['-r', output / 'last.pth'])
    elif tuning:
        command.extend(['-t', tuning])
    run(command, f'train_l_{task}.log')
    best = output / 'best_stg2.pth'
    if not best.exists():
        best = output / 'best_stg1.pth'
    assert best.exists(), f'No trained checkpoint in {output}'
    return best


try:
    scope_file = REP / 'scope.json'
    if scope_file.exists() and not json.loads(scope_file.read_text()).get('include_objects365', False):
        state('deferred_by_user', reason='Current selected scope is the COCO-only main experiment')
        raise SystemExit(0)
    assert json.loads((REP / 'objects365/preprocessing_equivalence.json').read_text())['status'] == 'passed'
    while not (DATA / 'download_complete.json').exists():
        status_file = REP / 'objects365/download_status.json'
        status = json.loads(status_file.read_text()) if status_file.exists() else {}
        if status.get('stage') == 'failed':
            raise RuntimeError(f'Objects365 download failed: {status.get("error")}')
        state('waiting_complete_original_data', download=status.get('current'),
              completed_files=status.get('completed_files'), total_files=97)
        time.sleep(30)
    if not (DATA / 'preprocess_complete.json').exists():
        state('preprocessing_full_dataset', validation_images=5000, max_edge=640)
        run(['reproduction/preprocess_objects365.py', '--workers', '4'], 'preprocess_objects365.log')
    ready = json.loads((DATA / 'preprocess_complete.json').read_text())
    assert ready['splits']['val']['images'] == 5000
    while not (REP / 'self_trained_l_coco_metrics.json').exists():
        state('waiting_coco_training_completion', dataset_ready=True)
        time.sleep(60)
    if not (REP / 'self_trained_l_obj365_metrics.json').exists():
        state('pretraining_l_objects365', training_images=ready['splits']['train']['images'], epochs=21)
        best = train('obj365')
        state('evaluating_objects365_pretraining', checkpoint=str(best))
        run(['reproduction/evaluate_official.py', '--checkpoint', best, '--self-trained',
             '--config', 'reproduction/configs/paper_l_obj365.yml', '--dataset', 'objects365',
             '--result-tag', 'self_trained_l_obj365'], 'self_trained_l_obj365_eval.log')
    pretrained = Path(json.loads((REP / 'self_trained_l_obj365_metrics.json').read_text())['checkpoint'])
    state('finetuning_l_obj2coco', epochs=32, initialization=str(pretrained))
    best = train('obj2coco', tuning=pretrained)
    state('evaluating_l_obj2coco', checkpoint=str(best), paper_AP_percent=57.1)
    run(['reproduction/evaluate_official.py', '--checkpoint', best, '--self-trained',
         '--config', 'reproduction/configs/paper_l_obj2coco.yml', '--result-tag', 'self_trained_l_obj2coco',
         '--paper-ap', '57.1'], 'self_trained_l_obj2coco_eval.log')
    state('complete_l_objects365_and_coco_finetuning',
          result=str(REP / 'self_trained_l_obj2coco_metrics.json'))
except Exception as exc:
    state('failed', error=str(exc), traceback=traceback.format_exc())
    raise

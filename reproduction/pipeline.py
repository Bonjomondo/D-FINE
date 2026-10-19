"""Persistent official evaluation -> full COCO L training -> independent evaluation."""
import datetime
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / 'reproduction'
DATA = ROOT.parent / 'datasets/coco2017'
PYTHON = str(ROOT / '.venv/bin/python')
STATE = REP / 'status.json'
ENV = dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTHONUNBUFFERED='1',
           TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
# Upstream checkpoints are trusted artifacts downloaded from the author's release.
# PyTorch >=2.6 otherwise defaults to weights_only=True for full resume states.
lock = (REP / 'pipeline.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def state(stage, **details):
    payload = dict(stage=stage, updated_at=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                   pid=os.getpid(), model='D-FINE-L', batch_size=32, epochs=74,
                   activation_offload=False, block_activation_checkpointing=True,
                   objects365='deferred_by_user', **details)
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(STATE)
    print(json.dumps(payload), flush=True)


def run(arguments, log_name):
    command = [PYTHON, '-u', *map(str, arguments)]
    print('Running:', command, flush=True)
    with (REP / 'logs' / log_name).open('a') as output:
        subprocess.run(command, cwd=ROOT, env=ENV, stdout=output, stderr=subprocess.STDOUT, check=True)


def wait_for(predicate, stage):
    while not predicate():
        state(stage)
        time.sleep(30)


try:
    for report in ['checkpoint_equivalence.json', 'checkpoint_real_batch_smoke.json']:
        assert json.loads((REP / report).read_text())['status'] == 'passed', report
    checkpoint = ROOT / 'weight/official/dfine_l_coco.pth'
    wait_for(lambda: (DATA / 'val2017.zip.verified.json').exists()
             and checkpoint.exists() and checkpoint.stat().st_size == 126059754,
             'waiting_official_checkpoint')
    if not (REP / 'official_l_coco_metrics.json').exists():
        state('evaluating_official_checkpoint', validation_images=5000)
        run(['reproduction/evaluate_official.py'], 'official_l_coco_eval.log')
    wait_for(lambda: (DATA / 'train2017.zip.verified.json').exists(), 'waiting_coco_train')
    state('verifying_coco_train')
    run(['reproduction/verify_data.py'], 'verify_data.log')
    out = ROOT / 'output/paper_l_coco_seed0'
    command = ['reproduction/run.py', '-c', 'reproduction/configs/paper_l_coco.yml',
               '--use-amp', '--seed=0', '-d', 'cuda']
    if (out / 'last.pth').exists():
        command.extend(['-r', out / 'last.pth'])
    state('training_l_coco', training_images=118287, validation_images=5000,
          training_log=str(REP / 'logs/train_l_coco.log'))
    run(command, 'train_l_coco.log')
    best = out / 'best_stg2.pth'
    if not best.exists():
        best = out / 'best_stg1.pth'
    assert best.exists(), 'No trained checkpoint was produced'
    state('evaluating_self_trained_checkpoint', checkpoint=str(best))
    run(['reproduction/evaluate_official.py', '--checkpoint', best, '--self-trained'],
        'self_trained_l_coco_eval.log')
    state('complete_l_coco', result=str(REP / 'self_trained_l_coco_metrics.json'),
          selected_scope_complete=True)
except Exception as exc:
    state('failed', error=str(exc), traceback=traceback.format_exc())
    raise

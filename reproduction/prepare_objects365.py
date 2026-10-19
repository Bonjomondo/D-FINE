"""Download original v2 train/val shards, verify, extract, then discard each archive."""
import datetime
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import tarfile
import traceback
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo
from prepare_coco import download

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / 'reproduction'
DATA = ROOT.parent / 'datasets/objects365'
META = DATA / 'verified'
MANIFEST = REP / 'objects365/download_manifest.json'
STATE = REP / 'objects365/download_status.json'


def state(stage, **details):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(stage=stage, updated_at=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                   pid=os.getpid(), **details)
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(payload, indent=2))
    temp.replace(STATE)
    print(json.dumps(payload), flush=True)


def verify_hash(path, item):
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    assert path.stat().st_size == item['size']
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8*1024*1024), b''):
            sha.update(chunk)
            md5.update(chunk)
    if item.get('sha256'):
        assert sha.hexdigest() == item['sha256'], f'SHA256 mismatch: {path}'
    if item.get('md5'):
        assert md5.hexdigest() == item['md5'], f'MD5 mismatch: {path}'
    return sha.hexdigest()


def extract(archive, item):
    count = 0
    with gzip.open(archive, 'rb') as stream:
        with tarfile.open(fileobj=stream, mode='r|') as tar:
            for member in tar:
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f'Unsupported tar member: {member.name}')
                parts = PurePosixPath(member.name).parts
                if '..' in parts or PurePosixPath(member.name).is_absolute():
                    raise ValueError(f'Unsafe archive path: {member.name}')
                if item['kind'] == 'annotation':
                    if parts[-1] != 'zhiyuan_objv2_train.json':
                        continue
                    destination = DATA / 'train' / parts[-1]
                else:
                    patch = f"patch{item['patch']}"
                    if patch not in parts:
                        raise ValueError(f'Unexpected shard structure: {member.name}')
                    relative = parts[parts.index(patch):]
                    destination = DATA / item['split'] / item['directory'] / Path(*relative)
                assert destination.resolve().is_relative_to(DATA.resolve())
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + '.extracting')
                with tar.extractfile(member) as source, temporary.open('wb') as output:
                    shutil.copyfileobj(source, output, 1024*1024)
                assert temporary.stat().st_size == member.size
                temporary.replace(destination)
                count += 1
                if count % 10000 == 0:
                    print(f"Extracted {item['split']}/{item['name']}: {count} files", flush=True)
        # Consume the gzip trailer to verify its CRC even after tar's EOF blocks.
        while stream.read(1024*1024):
            pass
    assert count > 0, item
    return count


def main():
    scope_file = REP / 'scope.json'
    if scope_file.exists() and not json.loads(scope_file.read_text()).get('include_objects365', False):
        state('deferred_by_user', reason='Current selected scope is the COCO-only main experiment')
        return
    DATA.mkdir(parents=True, exist_ok=True)
    META.mkdir(exist_ok=True)
    lock = (DATA / 'download.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads(MANIFEST.read_text())
    assert all(x['accessible_anonymously'] for x in manifest['files'])
    # Annotations and validation first; the complete original train split follows.
    items = sorted(manifest['files'], key=lambda x: (x['kind'] != 'annotation', x['split'] != 'val', x.get('patch', -1)))
    for index, item in enumerate(items):
        if scope_file.exists() and not json.loads(scope_file.read_text()).get('include_objects365', False):
            state('deferred_by_user', reason='Current selected scope is the COCO-only main experiment')
            return
        marker = META / f"{item['split']}_{item['name']}.json"
        if marker.exists():
            continue
        archive = DATA / 'archives' / item['split'] / item['name']
        required = 2*item['size'] + 40*2**30
        free = shutil.disk_usage(DATA).free
        if free < required:
            raise OSError(f'Insufficient free space: {free} bytes; need {required} for shard plus reserve')
        state('downloading', current=f"{item['split']}/{item['name']}", completed_files=index,
              total_files=len(items), archive_bytes=item['size'], url=item['url'])
        download(item['url'], archive, item['size'], workers=8)
        digest = verify_hash(archive, item)
        state('extracting', current=f"{item['split']}/{item['name']}", completed_files=index, total_files=len(items))
        if item['name'].endswith('.json'):
            destination = DATA / item['split'] / item['name']
            destination.parent.mkdir(parents=True, exist_ok=True)
            archive.replace(destination)
            count = 1
        else:
            count = extract(archive, item)
        marker.write_text(json.dumps(dict(item, actual_sha256=digest, extracted_files=count,
                                          gzip_crc_verified=item['name'].endswith('.gz')), indent=2))
        # These are task-owned, reproducible archives; original extracted images remain.
        if archive.exists():
            archive.unlink()
    ready = dict(files=len(items), total_archive_bytes=manifest['total_archive_bytes'],
                 source_manifest=str(MANIFEST), all_shards_verified=True)
    (DATA / 'download_complete.json').write_text(json.dumps(ready, indent=2))
    state('download_complete', **ready)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        state('failed', error=str(exc), traceback=traceback.format_exc())
        raise

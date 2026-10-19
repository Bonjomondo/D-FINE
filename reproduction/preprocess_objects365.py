"""Stream the official remap+resize procedure without loading all annotations into RAM."""
import argparse
import concurrent.futures
import datetime
import errno
import functools
import json
import os
import shutil
import sqlite3
import traceback
from pathlib import Path
from zoneinfo import ZoneInfo
import ijson
from PIL import Image


def items(path, field):
    with path.open('rb') as source:
        yield from ijson.items(source, field + '.item', use_float=True)


def process_image(base, source, position, image, val_size):
    image = dict(image)
    raw = base / source / image['file_name']
    assert raw.resolve().is_relative_to(base.resolve())
    target = 'val' if source == 'val' and position < val_size else 'train'
    original_name = image['file_name']
    scale = 1.
    try:
        with Image.open(raw) as im:
            width, height = im.size
            resized = max(width, height) > 640
            if resized:
                scale = 640 / max(width, height)
                width, height = int(width*scale), int(height*scale)
                destination = raw.with_name(raw.name.replace('.jpg', '_resized.jpg'))
                temporary = destination.with_name(destination.name + '.resizing')
                im = im.resize((width, height), Image.Resampling.LANCZOS)
                fmt = Image.registered_extensions()[destination.suffix.lower()]
                im.save(temporary, format=fmt)
                temporary.replace(destination)
                image['file_name'] = image['file_name'].replace('.jpg', '_resized.jpg')
            else:
                destination = raw
        image['width'], image['height'] = width, height
        if source == 'val' and target == 'train':
            # Same path remap as the official script, using hardlinks to share bytes.
            image['file_name'] = str(Path('images_from_val', *Path(image['file_name']).parts[1:]))
            linked = base / 'train' / image['file_name']
            linked.parent.mkdir(parents=True, exist_ok=True)
            if not linked.exists():
                os.link(destination, linked)
        error = None
    except (OSError, ValueError) as exc:
        if isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EDQUOT):
            raise
        error = f'{type(exc).__name__}: {exc}'
        width = height = 0
    return (source, image['id'], position, target, json.dumps(image, separators=(',', ':')),
            width, height, scale, int(error is None), original_name, error)


def preprocess(base, val_size=5000, workers=4, max_missing=10, progress=None):
    base = Path(base).resolve()
    database = base / 'preprocessing.sqlite3'
    conn = sqlite3.connect(database)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('''CREATE TABLE IF NOT EXISTS images (
        source TEXT, id INTEGER, position INTEGER, target TEXT, record TEXT,
        width INTEGER, height INTEGER, scale REAL, valid INTEGER, original_name TEXT, error TEXT,
        PRIMARY KEY(source,id))''')
    raw_paths = {split: base / split / f'zhiyuan_objv2_{split}.json' for split in ('train', 'val')}
    for source in ('val', 'train'):
        existing = {row[0] for row in conn.execute('SELECT id FROM images WHERE source=?', (source,))}
        indexed = enumerate(items(raw_paths[source], 'images'))
        count = len(existing)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            while True:
                batch = []
                for _ in range(256):
                    try:
                        position, image = next(indexed)
                    except StopIteration:
                        break
                    if image['id'] not in existing:
                        batch.append((position, image))
                if not batch:
                    # An empty batch may contain already processed rows during resume.
                    try:
                        position, image = next(indexed)
                    except StopIteration:
                        break
                    if image['id'] in existing:
                        continue
                    batch.append((position, image))
                futures = [pool.submit(process_image, base, source, pos, image, val_size) for pos, image in batch]
                conn.executemany('INSERT OR REPLACE INTO images VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                                 (f.result() for f in futures))
                conn.commit()
                count += len(batch)
                if count % 4096 < len(batch) and progress:
                    progress('resizing', source=source, processed_images=count)
    missing = [dict(source=r[0], id=r[1], file_name=r[2], error=r[3]) for r in conn.execute(
        'SELECT source,id,original_name,error FROM images WHERE valid=0')]
    (base / 'missing_images.json').write_text(json.dumps(missing, indent=2))
    # The official resizing script skips unreadable images. Refuse a large loss,
    # which could indicate incomplete extraction rather than upstream omissions.
    if len(missing) > max_missing:
        raise RuntimeError(f'{len(missing)} missing/unreadable images; refusing partial dataset')
    val_count = conn.execute("SELECT count(*) FROM images WHERE valid=1 AND target='val'").fetchone()[0]
    assert val_count == val_size, (val_count, val_size)
    overlap = conn.execute("""SELECT a.id FROM images a JOIN images b ON a.id=b.id
                              WHERE a.source='val' AND a.target='train' AND b.source='train' LIMIT 1""").fetchone()
    assert overlap is None, f'Train/validation IDs overlap: {overlap}'

    @functools.lru_cache(maxsize=100000)
    def metadata(source, image_id):
        return conn.execute('SELECT target,width,height,scale,valid FROM images WHERE source=? AND id=?',
                            (source, image_id)).fetchone()

    report = {'validation_size': val_size, 'unreadable_images': missing, 'splits': {}}
    for split in ('val', 'train'):
        destination = base / split / f'new_zhiyuan_objv2_{split}_resized.json'
        temporary = destination.with_suffix('.json.tmp')
        categories = list(items(raw_paths[split], 'categories'))
        image_count = annotation_count = 0
        if progress:
            progress('writing_annotations', split=split)
        with temporary.open('w') as output:
            output.write('{"images":[')
            rows = conn.execute("""SELECT record FROM images WHERE valid=1 AND target=?
                                   ORDER BY CASE WHEN source='val' THEN 0 ELSE 1 END,position""", (split,))
            for (record,) in rows:
                image = json.loads(record)
                assert (base / split / image['file_name']).is_file(), image['file_name']
                if image_count:
                    output.write(',')
                output.write(record)
                image_count += 1
            output.write('],"annotations":[')
            sources = ('val',) if split == 'val' else ('val', 'train')
            for source in sources:
                for annotation in items(raw_paths[source], 'annotations'):
                    meta = metadata(source, annotation['image_id'])
                    if not meta or meta[0] != split or not meta[4]:
                        continue
                    _, width, height, scale, _ = meta
                    if scale != 1:
                        annotation['area'] *= scale**2
                        annotation['bbox'] = [coord*scale for coord in annotation['bbox']]
                        for key in ('orig_size', 'size'):
                            if key in annotation:
                                annotation[key] = (width, height)
                    if annotation_count:
                        output.write(',')
                    json.dump(annotation, output, separators=(',', ':'))
                    annotation_count += 1
            output.write('],"categories":')
            json.dump(categories, output, separators=(',', ':'))
            output.write('}')
        temporary.replace(destination)
        report['splits'][split] = dict(images=image_count, annotations=annotation_count,
                                       categories=len(categories), path=str(destination))
    conn.close()
    (base / 'preprocess_complete.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'datasets/objects365')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    status_path = Path(__file__).resolve().parent / 'objects365/preprocess_status.json'

    def status(stage, **details):
        payload = dict(stage=stage, updated_at=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), **details)
        status_path.write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload), flush=True)

    try:
        assert (args.base_dir / 'download_complete.json').exists(), 'Full original dataset download is incomplete'
        if shutil.disk_usage(args.base_dir).free < 140*2**30:
            raise OSError('Need 140 GiB free for resized images, rewritten JSON and training reserve')
        result = preprocess(args.base_dir, workers=args.workers, progress=status)
        assert all(s['categories'] == 365 for s in result['splits'].values())
        status('complete', result=result)
    except Exception as exc:
        status('failed', error=str(exc), traceback=traceback.format_exc())
        raise

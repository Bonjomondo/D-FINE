"""Download official COCO2017 archives with resumable range requests and CRC checks."""
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "datasets" / "coco2017"
ARCHIVES = DATA / "archives"


def download(url, dest, size=None, workers=16):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return
    if size is None:
        subprocess.run(["curl", "-fLsS", "--retry", "8", "--retry-delay", "3",
                        "--connect-timeout", "30", "-C", "-", "-o", str(dest), url], check=True)
        return
    block = 16 * 1024 * 1024
    parts = dest.parent / (dest.name + ".parts16m")
    parts.mkdir(exist_ok=True)

    def part(index):
        start = index * block
        end = min(size, start + block) - 1
        path = parts / f"{index:05d}"
        expected = end - start + 1
        if path.exists() and path.stat().st_size == expected:
            return path
        for attempt in range(8):
            offset = path.stat().st_size if path.exists() else 0
            if offset > expected:
                path.unlink()
                offset = 0
            with path.open("ab") as output:
                result = subprocess.run(["curl", "-fLsS", "--connect-timeout", "30",
                                         "--max-time", "900", "--range",
                                         f"{start + offset}-{end}", url], stdout=output)
            if result.returncode == 0 and path.stat().st_size == expected:
                print(f"{dest.name}: part {index + 1}/{(size + block - 1)//block} complete", flush=True)
                return path
            time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"Could not download {url} range {start}-{end}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        paths = list(pool.map(part, range((size + block - 1) // block)))
    temporary = dest.with_suffix(dest.suffix + ".assembling")
    with temporary.open("wb") as output:
        for path in paths:
            with path.open("rb") as source:
                shutil.copyfileobj(source, output, 8 * 1024 * 1024)
    temporary.replace(dest)
    shutil.rmtree(parts)


def prepare(name, url, size, expected_count):
    archive = ARCHIVES / name
    marker = DATA / (name + ".verified.json")
    if marker.exists():
        return
    download(url, archive, size)
    with zipfile.ZipFile(archive) as z:
        files = [n for n in z.namelist() if not n.endswith("/")]
        if expected_count is not None:
            assert len(files) == expected_count, (name, len(files))
        # zipfile verifies the CRC while extracting each file.
        z.extractall(DATA)
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    marker.write_text(json.dumps({"url": url, "bytes": archive.stat().st_size,
                                  "sha256": digest.hexdigest(), "file_count": len(files),
                                  "crc_verified_on_extract": True}, indent=2))
    print(f"VERIFIED {name}: {len(files)} files", flush=True)


if __name__ == "__main__":
    # Validation data first, to allow checkpoint evaluation while train downloads.
    prepare("annotations_trainval2017.zip", "http://images.cocodataset.org/annotations/annotations_trainval2017.zip", 252907541, None)
    prepare("val2017.zip", "http://images.cocodataset.org/zips/val2017.zip", 815585330, 5000)
    prepare("train2017.zip", "http://images.cocodataset.org/zips/train2017.zip", 19336861798, 118287)
    print("COCO2017 download and extraction complete", flush=True)

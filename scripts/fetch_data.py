"""Download immutable public files and verify upstream Git LFS SHA-256 hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path: Path, source: dict) -> None:
    if path.stat().st_size != source["size"] or sha256(path) != source["sha256"]:
        raise ValueError(f"Size or SHA-256 mismatch: {path}")


def fetch_one(key: str, source: dict, data_dir: Path) -> dict:
    target = data_dir / key / Path(source["file"]).name
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/datasets/{source["repo"]}/resolve/{source["revision"]}/{source["file"]}'
    if target.exists():
        verify(target, source)
        print(f"Verified cached {key}", flush=True)
    else:
        partial = target.with_suffix(target.suffix + ".partial")
        print(f"Downloading {key} ({source['size'] / 1e6:.1f} MB)", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "pilot-data-audit/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        verify(partial, source)
        partial.replace(target)
        print(f"Verified downloaded {key}", flush=True)
    return {"key": key, **source, "url": url, "relative_path": str(target.relative_to(data_dir)).replace("\\", "/")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--sources", nargs="+", help="Default: all pinned sources")
    args = parser.parse_args()
    sources = json.loads((ROOT / "configs/data_sources.json").read_text(encoding="utf-8"))
    selected = args.sources or list(sources)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fetch_one, key, sources[key], args.data_dir) for key in selected]
        manifest = [future.result() for future in futures]
    (args.data_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Reconstruct pinned public source/model assets; no inference or account client.

Run with --sources and/or --model instructor|deberta. Downloads are serial,
unauthenticated HTTPS reads. Existing files must match their published hashes.
The audit manifest is a historical record; dynamic API responses are not fetched
or falsely asserted to be byte-identical on a later day.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/openai-followup-20261031'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def retrieve(url, path, expected=None):
    if not url.startswith(('https://raw.githubusercontent.com/', 'https://huggingface.co/')):
        raise ValueError('Public pinned asset host required')
    path = Path(path)
    if path.exists():
        actual = sha(path)
        if expected and actual != expected:
            raise ValueError('Existing asset checksum differs: ' + str(path))
        return actual
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.download-partial')
    request = urllib.request.Request(url, headers={'User-Agent': 'pilot-reproduction-public-assets/1'})
    # Exclusive creation preserves evidence and prevents two writers downloading
    # the same large file. Inspect an interrupted partial before another attempt.
    with partial.open('xb') as target, urllib.request.urlopen(request, timeout=60) as response:
        while chunk := response.read(1024 * 1024):
            target.write(chunk)
    actual = sha(partial)
    if expected and actual != expected:
        raise ValueError('Downloaded asset checksum differs; partial retained: ' + str(partial))
    partial.rename(path)
    return actual


def write_or_check(path, payload):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != payload:
            raise ValueError('Existing asset record differs: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', action='store_true')
    parser.add_argument('--model', choices=['instructor', 'deberta'])
    args = parser.parse_args()
    if not args.sources and not args.model:
        parser.error('Choose --sources and/or --model')
    config = json.loads((ROOT / 'configs/followup_assets.json').read_text(encoding='utf-8'))
    if args.sources:
        for record in config['source_files']:
            retrieve(record['url'], BASE / record['destination'], record['sha256'])
        target = BASE / 'sources/manifest.json'
        original = (ROOT / 'configs/followup_source_audit_manifest.json').read_bytes()
        if target.exists() and target.read_bytes() != original:
            raise ValueError('Existing historical source manifest differs')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with target.open('xb') as stream:
                stream.write(original)
        print(f'Verified {len(config["source_files"])} pinned public source files', flush=True)
    if args.model:
        spec = config['models'][args.model]
        folder = BASE / spec['cache'] / ('models--' + spec['repo'].replace('/', '--')) / 'snapshots' / spec['revision']
        files = {}
        for entry in spec['files']:
            url = f'https://huggingface.co/{spec["repo"]}/resolve/{spec["revision"]}/{entry["name"]}'
            files[entry['name']] = retrieve(url, folder / entry['name'], entry.get('sha256'))
            print(f'Verified {args.model} {entry["name"]}', flush=True)
        write_or_check(BASE / (args.model + '-asset-verification.json'), dict(
            model=spec['repo'], revision=spec['revision'], local_path=str(folder),
            files=files, public_download=True, inference_api_calls=0))
        if args.model == 'instructor':
            # Keep the original probe's input schema for a clean checkout.
            write_or_check(BASE / 'instructor-download.json', dict(local_path=str(folder),
                revision=spec['revision'], public_download=True, max_workers=1, inference_api_calls=0))
        print(str(folder), flush=True)


if __name__ == '__main__':
    main()

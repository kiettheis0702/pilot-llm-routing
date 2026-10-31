"""Build, verify or restore pinned offline inputs. No network or provider SDK."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
from uuid import uuid4
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'configs/openai-reproduction-files.json'
RELEASE = ROOT / 'configs/openai-reproduction-release.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def members(manifest):
    if (manifest.get('version') != 1
            or manifest.get('kind') != 'openai-offline-reproduction-inputs'
            or not isinstance(manifest.get('files'), list) or not manifest['files']):
        raise ValueError('Invalid reproduction manifest')
    result = {}
    folded = set()
    for entry in manifest['files']:
        name = entry.get('path')
        if not isinstance(name, str):
            raise ValueError('Invalid member path')
        parts = name.split('/')
        if (len(parts) < 2 or parts[0] not in {'data', 'outputs'}
                or PurePosixPath(name).is_absolute()
                or any(not p or p in {'.', '..'} or p.endswith((' ', '.'))
                       or any(ord(c) < 32 or c in '\\:<>"|?*' for c in p)
                       or p.lower().startswith(('.env', 'credentials', 'secrets'))
                       or re.fullmatch(r'(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?', p)
                       for p in parts)):
            raise ValueError('Unsafe or credential-like member path: ' + name)
        if (name.casefold() in folded or type(entry.get('size')) is not int
                or entry['size'] < 0 or not re.fullmatch('[0-9a-f]{64}', str(entry.get('sha256')))):
            raise ValueError('Invalid or duplicate manifest entry: ' + name)
        result[name] = entry
        folded.add(name.casefold())
    return result


def safe_path(root, name):
    root = Path(root).resolve()
    path = root / name
    if not path.resolve().is_relative_to(root):
        raise ValueError('Member escapes target root: ' + name)
    # Reject junctions/reparse points too, including ancestors inside the root.
    current = path
    while current != root:
        if current.is_symlink() or (current.exists() and
                getattr(current.lstat(), 'st_file_attributes', 0) & 0x400):
            raise ValueError('Linked member path: ' + name)
        current = current.parent
    return path


def check_file(path, entry):
    if not path.is_file() or path.stat().st_size != entry['size'] or digest(path) != entry['sha256']:
        raise ValueError('File checksum mismatch: ' + entry['path'])


def build(root, manifest, archive):
    entries = members(manifest)
    archive = Path(archive)
    if archive.exists():
        raise FileExistsError(archive)
    for name, entry in entries.items():
        check_file(safe_path(root, name), entry)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as handle:
        for name in sorted(entries):
            handle.write(safe_path(root, name), arcname=name)
    receipt = {'filename': archive.name, 'size': archive.stat().st_size, 'sha256': digest(archive)}
    verify_archive(archive, manifest, receipt['sha256'])
    return receipt


def verify_archive(archive, manifest, expected_sha256):
    entries = members(manifest)
    if not re.fullmatch('[0-9a-f]{64}', str(expected_sha256)) or digest(archive) != expected_sha256:
        raise ValueError('Archive checksum mismatch')
    with zipfile.ZipFile(archive) as handle:
        infos = handle.infolist()
        if len(infos) != len(entries) or {i.filename for i in infos} != set(entries):
            raise ValueError('Archive contains missing, duplicate or unexpected members')
        for info in infos:
            entry = entries[info.filename]
            if (info.file_size != entry['size'] or info.is_dir()
                    or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError('Archive member size/type mismatch: ' + info.filename)
            with handle.open(info) as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != entry['sha256']:
                    raise ValueError('Archive member checksum mismatch: ' + info.filename)
    return {'files': len(entries), 'uncompressed_bytes': sum(e['size'] for e in entries.values())}


def restore(archive, manifest, root, expected_sha256):
    verify_archive(archive, manifest, expected_sha256)
    entries = members(manifest)
    missing = []
    for name, entry in entries.items():
        target = safe_path(root, name)
        if target.exists():
            try:
                check_file(target, entry)
            except ValueError as exc:
                raise FileExistsError('Refusing to replace existing input: ' + str(target)) from exc
        else:
            if any(p.exists() and not p.is_dir() for p in target.parents):
                raise FileExistsError('Input parent is not a directory: ' + str(target))
            missing.append(name)
    # Validate the whole archive and every destination before the first write.
    with zipfile.ZipFile(archive) as handle:
        for name in missing:
            target = safe_path(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.restore-' + uuid4().hex)
            try:
                with handle.open(name) as source, temporary.open('xb') as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                check_file(temporary, entries[name])
                # Atomic publication without replacing a concurrently created file.
                os.link(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
    return {'restored': len(missing), 'reused': len(entries) - len(missing)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('build', 'verify', 'restore'))
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if args.stage == 'build':
        result = build(args.root, manifest, args.archive)
    else:
        release = json.loads(RELEASE.read_text(encoding='utf-8'))
        result = (restore(args.archive, manifest, args.root, release['sha256'])
                  if args.stage == 'restore'
                  else verify_archive(args.archive, manifest, release['sha256']))
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()

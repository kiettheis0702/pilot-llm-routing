"""A corrupt or conflicting transfer must never alter a research input."""
import hashlib
import json
import zipfile

import pytest


def fixture(tmp_path):
    source = tmp_path / 'source'
    files = {'data/processed/sample.bin': b'\x00\x01real input\xff',
             'outputs/cache/vectors.npy': b'cached vector fixture'}
    for name, content in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    manifest = {'version': 1, 'kind': 'openai-offline-reproduction-inputs',
                'files': [{'path': name, 'size': len(content),
                           'sha256': hashlib.sha256(content).hexdigest()}
                          for name, content in sorted(files.items())]}
    return source, files, manifest


def test_roundtrip_exact_bytes_and_idempotent_restore(tmp_path):
    from tools.reproduction_bundle import build, restore, verify_archive
    source, files, manifest = fixture(tmp_path)
    archive = tmp_path / 'inputs.zip'
    receipt = build(source, manifest, archive)
    assert receipt['sha256'] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert verify_archive(archive, manifest, receipt['sha256'])['files'] == 2
    target = tmp_path / 'fresh-checkout'
    assert restore(archive, manifest, target, receipt['sha256']) == {'restored': 2, 'reused': 0}
    for name, content in files.items():
        assert (target / name).read_bytes() == content
    assert restore(archive, manifest, target, receipt['sha256']) == {'restored': 0, 'reused': 2}


def test_archive_corruption_fails_before_creating_target(tmp_path):
    from tools.reproduction_bundle import build, restore
    source, _, manifest = fixture(tmp_path)
    archive = tmp_path / 'inputs.zip'
    receipt = build(source, manifest, archive)
    with archive.open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='checksum'):
        restore(archive, manifest, tmp_path / 'target', receipt['sha256'])
    assert not (tmp_path / 'target').exists()


@pytest.mark.parametrize('change', ['member', 'missing', 'extra', 'duplicate'])
def test_member_integrity_checked_even_with_matching_archive_digest(tmp_path, change):
    from tools.reproduction_bundle import restore
    _, files, manifest = fixture(tmp_path)
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as handle:
        for index, (name, content) in enumerate(files.items()):
            if change == 'missing' and index == 0:
                continue
            handle.writestr(name, b'wrong' if change == 'member' and index == 0 else content)
        if change == 'extra':
            handle.writestr('../outside.txt', 'escape')
        if change == 'duplicate':
            with pytest.warns(UserWarning):
                handle.writestr(next(iter(files)), b'duplicate')
    with pytest.raises(ValueError):
        restore(archive, manifest, tmp_path / 'target', hashlib.sha256(archive.read_bytes()).hexdigest())
    assert not (tmp_path / 'target').exists()
    assert not (tmp_path / 'outside.txt').exists()


def test_conflict_checked_for_all_destinations_before_any_write(tmp_path):
    from tools.reproduction_bundle import build, restore
    source, files, manifest = fixture(tmp_path)
    archive = tmp_path / 'inputs.zip'
    receipt = build(source, manifest, archive)
    target = tmp_path / 'target'
    conflict = target / 'outputs/cache/vectors.npy'
    conflict.parent.mkdir(parents=True)
    conflict.write_bytes(b'user work')
    with pytest.raises(FileExistsError):
        restore(archive, manifest, target, receipt['sha256'])
    assert conflict.read_bytes() == b'user work'
    assert not (target / 'data').exists()


@pytest.mark.parametrize('name', ['../outside', '/outside', 'C:/outside',
    'outputs/../.env', 'outputs/cache:stream', 'outputs/CON',
    'outputs/a.', 'outputs/a\\b', '.env.openai', 'outputs/.env.openai'])
def test_manifest_cannot_select_unsafe_paths_or_credentials(tmp_path, name):
    from tools.reproduction_bundle import build
    source, _, manifest = fixture(tmp_path)
    manifest['files'][0]['path'] = name
    with pytest.raises(ValueError):
        build(source, manifest, tmp_path / 'inputs.zip')
    assert not (tmp_path / 'inputs.zip').exists()


def test_build_rejects_modified_source_and_existing_archive(tmp_path):
    from tools.reproduction_bundle import build
    source, _, manifest = fixture(tmp_path)
    archive = tmp_path / 'inputs.zip'
    build(source, manifest, archive)
    original = archive.read_bytes()
    with pytest.raises(FileExistsError):
        build(source, manifest, archive)
    assert archive.read_bytes() == original
    (source / manifest['files'][0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        build(source, manifest, tmp_path / 'changed.zip')
    assert not (tmp_path / 'changed.zip').exists()


def test_interrupted_restore_can_reuse_already_verified_member(tmp_path):
    from tools.reproduction_bundle import build, restore
    source, files, manifest = fixture(tmp_path)
    archive = tmp_path / 'inputs.zip'
    receipt = build(source, manifest, archive)
    target = tmp_path / 'target'
    name = next(iter(files))
    path = target / name
    path.parent.mkdir(parents=True)
    path.write_bytes(files[name])
    assert restore(archive, manifest, target, receipt['sha256']) == {'restored': 1, 'reused': 1}

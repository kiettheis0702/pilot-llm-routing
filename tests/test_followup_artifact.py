"""Reject modified/missing evidence before claiming a remote artifact matches."""
import base64
import hashlib
from types import SimpleNamespace

import pytest

import tools.openai_followup_artifact as publication


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(publication,'ROOT',tmp_path)
    path=tmp_path/'scientific.csv';path.write_bytes(b'method,reward\npilot,0.5\n')
    raw=path.read_bytes()
    plan=dict(manifest_sha256='a'*64,total_bytes=len(raw),files=[dict(
        path='scientific.csv',artifact_path='report/scientific.csv',bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest())])
    remote=SimpleNamespace(id='server-artifact',version='v0',qualified_name='fixture:v0',
        url='https://example.test/artifact',manifest=SimpleNamespace(entries={
            'report/scientific.csv':SimpleNamespace(size=len(raw),
                digest=base64.b64encode(hashlib.md5(raw).digest()).decode())}))
    return path,plan,remote


def test_remote_artifact_requires_exact_inventory_size_and_digest(tmp_path,monkeypatch):
    path,plan,remote=fixture(tmp_path,monkeypatch)
    assert publication.verify_remote(None,remote,plan)['remote_manifest_verified']
    entry=remote.manifest.entries['report/scientific.csv']
    original=entry.digest;entry.digest='different-server-content'
    with pytest.raises(ValueError,match='digest'):
        publication.verify_remote(None,remote,plan)
    entry.digest=original;remote.manifest.entries['unexpected.csv']=entry
    with pytest.raises(ValueError,match='inventory'):
        publication.verify_remote(None,remote,plan)


def test_changed_local_source_cannot_be_verified_against_old_remote_manifest(tmp_path,monkeypatch):
    path,plan,remote=fixture(tmp_path,monkeypatch)
    path.write_bytes(path.read_bytes().replace(b'0.5',b'0.8'))
    with pytest.raises(ValueError,match='changed'):
        publication.verify_remote(None,remote,plan)

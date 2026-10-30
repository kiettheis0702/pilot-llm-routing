import hashlib
import json
from pathlib import Path

import pytest

from tools.course.prepare_inputs import ensure_sources


def test_source_preparation_is_serial_and_never_accepts_corrupted_cache(tmp_path):
    source = {'repo':'example/data', 'revision':'revision', 'file':'a.bin',
              'size':4, 'sha256':hashlib.sha256(b'data').hexdigest()}
    calls = []
    def downloader(key, spec, folder):
        calls.append(key)
        p = folder/key/spec['file']
        p.parent.mkdir(parents=True)
        p.write_bytes(b'data')
        return {'key':key}
    result = ensure_sources(tmp_path, {'one':source, 'two':source}, downloader=downloader)
    assert calls == ['one','two']
    assert len(result) == 2
    (tmp_path/'one/a.bin').write_bytes(b'evil')
    def must_not_download(*args): raise AssertionError('Corrupt caches must fail before fetch')
    with pytest.raises(ValueError, match='SHA-256'):
        ensure_sources(tmp_path, {'one':source}, downloader=must_not_download)

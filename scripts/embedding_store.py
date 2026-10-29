"""Validated float32 disk caches, compatible with the small pretraining core."""
from __future__ import annotations

from pathlib import Path
import os
from uuid import uuid4

import numpy as np

from scripts.fetch_data import sha256
from scripts.pretraining import _digest, _hashes, _names, _read_json, _recipe, _write_json


def validate_matrix(matrix, rows):
    if (matrix.ndim != 2 or matrix.shape[0] != rows or matrix.shape[1] < 1
            or matrix.dtype != np.dtype('float32')):
        raise ValueError('Invalid float32 vector dimensions or dtype')
    for start in range(0, rows, 256):
        if not np.isfinite(matrix[start:start + 256]).all():
            raise ValueError('Nonfinite embedding vectors')


def vector_file(path):
    path = Path(path)
    result = path / 'vectors.npy'
    if result.is_symlink() or not result.resolve().is_relative_to(path.resolve()):
        raise ValueError('Vector path escapes cache')
    return result


def _publish_manifest(path, payload):
    # The final name is created only after a complete, flushed write. Linking
    # within the same directory is atomic and refuses an existing destination.
    temporary = path / f'.manifest-{uuid4().hex}.tmp'
    try:
        _write_json(temporary, payload)
        with temporary.open('r+b') as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path / 'manifest.json')
    finally:
        temporary.unlink(missing_ok=True)


class DiskEmbeddingCache:
    @classmethod
    def create(cls, path, ids, prompt_hashes, vectors, provenance, *, metadata=None):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        ids, hashes = _names(ids, 'query IDs'), _hashes(prompt_hashes)
        recipe = _recipe(provenance)
        values = np.asarray(vectors)
        if values.dtype.kind not in 'iuf' or values.ndim != 2 or len(ids) != len(hashes):
            raise ValueError('Invalid embedding matrix or identities')
        values = np.asarray(values, dtype=np.float32)
        validate_matrix(values, len(ids))
        path.mkdir(parents=True, exist_ok=False)
        np.save(path / 'vectors.npy', values, allow_pickle=False)
        return cls.publish(path, ids, hashes, recipe, metadata=metadata)

    @classmethod
    def publish(cls, path, ids, prompt_hashes, provenance, *, metadata=None):
        path = Path(path)
        if (path / 'manifest.json').exists():
            raise FileExistsError(path / 'manifest.json')
        ids, hashes = _names(list(ids), 'query IDs'), _hashes(list(prompt_hashes))
        if len(ids) != len(hashes):
            raise ValueError('Prompt hash alignment mismatch')
        recipe = _recipe(provenance)
        vector_path = vector_file(path)
        matrix = np.load(vector_path, mmap_mode='r', allow_pickle=False)
        validate_matrix(matrix, len(ids))
        payload = dict(version=1, kind='disk-embedding-cache', query_ids=list(ids),
            prompt_hashes=list(hashes), dimension=matrix.shape[1], dtype='float32',
            vector_file='vectors.npy', vector_sha256=sha256(vector_path),
            provenance=recipe, metadata={} if metadata is None else metadata)
        _publish_manifest(path, payload)
        return cls.load(path)

    @classmethod
    def load(cls, path):
        path = Path(path)
        payload = _read_json(path / 'manifest.json')
        fields = {'version','kind','query_ids','prompt_hashes','dimension','dtype',
                  'vector_file','vector_sha256','provenance','metadata'}
        if (set(payload) != fields or type(payload['version']) is not int or payload['version'] != 1
                or payload['kind'] != 'disk-embedding-cache' or payload['vector_file'] != 'vectors.npy'
                or payload['dtype'] != 'float32' or type(payload['dimension']) is not int
                or not isinstance(payload['metadata'], dict)):
            raise ValueError('Invalid disk cache manifest')
        result = cls()
        result.ids = _names(payload['query_ids'], 'query IDs')
        result.prompt_hashes = _hashes(payload['prompt_hashes'])
        result._provenance = _recipe(payload['provenance'])
        if len(result.ids) != len(result.prompt_hashes):
            raise ValueError('Cache identity alignment mismatch')
        vector_path = vector_file(path)
        if sha256(vector_path) != payload['vector_sha256']:
            raise ValueError('Vector checksum mismatch')
        result._vectors = np.load(vector_path, mmap_mode='r', allow_pickle=False)
        validate_matrix(result._vectors, len(result.ids))
        if result._vectors.shape[1] != payload['dimension']:
            raise ValueError('Dimension mismatch')
        result.dimension = payload['dimension']
        result.metadata = payload['metadata']
        result._fingerprint = _digest(payload)
        result._index = {q:i for i,q in enumerate(result.ids)}
        return result

    @property
    def provenance(self):
        return dict(self._provenance)

    @property
    def vectors(self):
        return self._vectors.copy()

    def iter_vectors(self, batch_size=256):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('Invalid batch size')
        for start in range(0, len(self.ids), batch_size):
            yield start, np.array(self._vectors[start:start + batch_size], copy=True)

    def get(self, query_id):
        if query_id not in self._index:
            raise ValueError('Unknown embedding query ID')
        return self._vectors[self._index[query_id]].copy()

    def fingerprint(self):
        return self._fingerprint

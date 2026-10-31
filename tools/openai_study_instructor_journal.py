"""Append-only embedding commits, durable prefix resume and an actual OS lock."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np

from scripts.embedding_store import DiskEmbeddingCache
from scripts.pretraining import _digest,_hashes,_names,_recipe


def _publish(path,payload):
    temporary=path.with_name('.'+path.name+'-'+uuid4().hex+'.tmp')
    try:
        with temporary.open('x',encoding='utf-8') as stream:
            json.dump(dict(payload=payload,sha256=_digest(payload)),stream,sort_keys=True,allow_nan=False)
            stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def _read(path):
    if path.is_symlink():raise ValueError('Journal symlink is not allowed')
    value=json.loads(path.read_text(encoding='utf-8'))
    if set(value)!={'payload','sha256'} or _digest(value['payload'])!=value['sha256']:
        raise ValueError('Journal checksum mismatch')
    return value['payload']


def _vector_hash(values):return hashlib.sha256(np.ascontiguousarray(values,dtype=np.float32).tobytes()).hexdigest()


def _validate(values,dimension):
    raw=np.asarray(values)
    if raw.dtype.kind not in 'fiu' or raw.ndim!=2 or raw.shape[0]<1 or raw.shape[1]!=dimension:
        raise ValueError('Invalid vector batch shape or dtype')
    values=np.asarray(raw,dtype=np.float32)
    if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values.astype(float),axis=1),1.,rtol=1e-5,atol=1e-6):
        raise ValueError('Expected finite unit vectors')
    return values


class EmbeddingJournal:
    def __init__(self,path,*,ids,prompt_hashes,provenance,dimension,metadata):
        self.path=Path(path);self._lock=None;self._matrix=None;self._cache=None;self.next_row=0
        self.ids=_names(list(ids),'journal IDs');self.hashes=_hashes(list(prompt_hashes));self.recipe=_recipe(provenance)
        if type(dimension) is not int or dimension<1 or len(self.ids)!=len(self.hashes) or not isinstance(metadata,dict):
            raise ValueError('Invalid journal dimensions or identity')
        self.dimension=dimension;self.metadata=metadata
        self.state=dict(version=1,kind='append-only-embedding-journal',ids=list(self.ids),prompt_hashes=list(self.hashes),
            provenance=self.recipe,dimension=dimension,metadata=metadata)
        self.identity=_digest(self.state)
        if self.path.is_symlink():raise ValueError('Journal path must not be a symlink')
        self.path.mkdir(parents=True,exist_ok=True)
        try:
            self._acquire()
            state_path=self.path/'journal-state.json'
            if state_path.exists():
                if _read(state_path)!=self.state:raise ValueError('Journal input/recipe identity mismatch')
            else:
                if (self.path/'vectors.npy').exists() or list(self.path.glob('chunk-*.json')):
                    raise ValueError('Orphan vectors without journal identity')
                _publish(state_path,self.state)
            vector_path=self.path/'vectors.npy'
            if vector_path.is_symlink():raise ValueError('Vector symlink is not allowed')
            if vector_path.exists():self._matrix=np.load(vector_path,mmap_mode='r+',allow_pickle=False)
            else:self._matrix=np.lib.format.open_memmap(vector_path,mode='w+',dtype=np.float32,shape=(len(self.ids),dimension))
            if self._matrix.shape!=(len(self.ids),dimension) or self._matrix.dtype!=np.dtype('float32'):
                raise ValueError('Journal vector dimensions changed')
            for chunk in sorted(self.path.glob('chunk-*.json')):
                row=_read(chunk)
                if (set(row)!={'start','end','identity','vector_sha256'} or type(row['start']) is not int
                        or type(row['end']) is not int or row['start']!=self.next_row
                        or not row['start']<row['end']<=len(self.ids) or row['identity']!=self.identity
                        or chunk.name!=f'chunk-{row["start"]:08d}-{row["end"]:08d}.json'):
                    raise ValueError('Invalid contiguous journal commit')
                values=self._matrix[row['start']:row['end']]
                if _vector_hash(values)!=row['vector_sha256']:raise ValueError('Committed vector checksum mismatch')
                _validate(values,dimension);self.next_row=row['end']
            if (self.path/'manifest.json').exists():
                self._cache=DiskEmbeddingCache.load(self.path)
                if (self.next_row!=len(self.ids) or self._cache.ids!=self.ids or self._cache.prompt_hashes!=self.hashes
                        or self._cache.provenance!=self.recipe or self._cache.metadata!=self.metadata):
                    raise ValueError('Published cache/journal identity mismatch')
        except BaseException:self.close();raise

    def _acquire(self):
        lock_path=self.path/'writer.lock'
        if lock_path.is_symlink():raise ValueError('Lock symlink is not allowed')
        self._lock=lock_path.open('a+b')
        if self._lock.seek(0,2)==0:self._lock.write(b'0');self._lock.flush()
        self._lock.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self._lock.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self._lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self._lock.close();self._lock=None
            raise RuntimeError('Another writer owns this embedding journal') from None

    def commit(self,start,values):
        if self._lock is None:raise RuntimeError('Writer is closed')
        if self._cache is not None or (self.path/'manifest.json').exists():raise FileExistsError('Cache already published')
        if type(start) is not int or start!=self.next_row:raise ValueError('Embedding commits must be contiguous')
        values=_validate(values,self.dimension);end=start+len(values)
        if end>len(self.ids):raise ValueError('Commit exceeds declared vector count')
        self._matrix[start:end]=values;self._matrix.flush()
        with (self.path/'vectors.npy').open('r+b') as stream:os.fsync(stream.fileno())
        _publish(self.path/f'chunk-{start:08d}-{end:08d}.json',dict(start=start,end=end,identity=self.identity,vector_sha256=_vector_hash(values)))
        self.next_row=end

    def finish(self):
        if self._lock is None:raise RuntimeError('Writer is closed')
        if self.next_row!=len(self.ids):raise ValueError('Embedding journal incomplete')
        if self._cache is None:
            self._matrix.flush();self._matrix._mmap.close();self._matrix=None
            self._cache=DiskEmbeddingCache.publish(self.path,self.ids,self.hashes,self.recipe,metadata=self.metadata)
        return self._cache

    def close(self):
        if self._matrix is not None:self._matrix._mmap.close();self._matrix=None
        if self._lock is not None:self._lock.close();self._lock=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

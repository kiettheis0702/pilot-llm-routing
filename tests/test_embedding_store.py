import hashlib
import json

import numpy as np
import pytest

RECIPE = {'model':'fixture','version':'1','source':'test fixture','normalization':'none','synthetic':True}


def create(path, vectors=None):
    from scripts.embedding_store import DiskEmbeddingCache
    if vectors is None:vectors=np.array([[1.,2.],[3.,4.]],dtype=np.float32)
    ids=[f'q{i}' for i in range(len(vectors))]
    return DiskEmbeddingCache.create(path,ids,[hashlib.sha256(x.encode()).hexdigest() for x in ids],vectors,RECIPE)


def test_large_cache_readonly_mmap_and_integrity(tmp_path):
    from scripts.embedding_store import DiskEmbeddingCache
    x=np.full((4200,1024),.25,dtype=np.float32)
    cache=create(tmp_path/'cache',x)
    assert (tmp_path/'cache/vectors.npy').stat().st_size>16*1024*1024
    np.testing.assert_array_equal(cache.get('q4199'),np.full(1024,.25))
    loaded=DiskEmbeddingCache.load(tmp_path/'cache')
    assert loaded.fingerprint()==cache.fingerprint()
    assert loaded.dimension==1024
    assert not loaded._vectors.flags.writeable
    with pytest.raises(FileExistsError):create(tmp_path/'cache',x)
    with (tmp_path/'cache/vectors.npy').open('r+b') as f:f.seek(-1,2);f.write(b'X')
    with pytest.raises(ValueError,match='checksum'):DiskEmbeddingCache.load(tmp_path/'cache')


@pytest.mark.parametrize('value',[np.array([[np.nan,1.]]),np.array([[np.inf,2.]]),np.array([[True,False]]),np.array([1.,2.])])
def test_invalid_vectors_never_publish_complete_cache(tmp_path,value):
    with pytest.raises(ValueError):create(tmp_path/'bad',value)
    assert not (tmp_path/'bad/manifest.json').exists()


def test_metadata_tamper_and_incomplete_cache_rejected(tmp_path):
    from scripts.embedding_store import DiskEmbeddingCache
    create(tmp_path/'cache')
    path=tmp_path/'cache/manifest.json'
    x=json.loads(path.read_text());x['payload']['vector_file']='../outside.npy';path.write_text(json.dumps(x))
    with pytest.raises(ValueError):DiskEmbeddingCache.load(tmp_path/'cache')
    (tmp_path/'unfinished').mkdir()
    with pytest.raises((ValueError,FileNotFoundError)):DiskEmbeddingCache.load(tmp_path/'unfinished')

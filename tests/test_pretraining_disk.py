import hashlib
import tracemalloc

import numpy as np

from scripts import pretraining as pre
from scripts.embedding_store import DiskEmbeddingCache

RECIPE={'model':'fixture','version':'1','source':'fixture','normalization':'none','synthetic':True}


def inputs(n=4):
    ids=[f'q{i}' for i in range(2*n+2)]
    hashes=[hashlib.sha256(q.encode()).hexdigest() for q in ids]
    vectors=np.array([[1. if i<n or i==2*n else -1.,3.] for i in range(len(ids))])
    rows=[dict(battle_id=f'b{i:05d}',query_id=q,winner='large' if i<n or i==2*n else 'small',
               loser='small' if i<n or i==2*n else 'large',split='train' if i<2*n else 'holdout') for i,q in enumerate(ids)]
    return ids,hashes,vectors,rows


def test_disk_cache_trains_and_projects_like_small_cache(tmp_path):
    ids,hashes,vectors,rows=inputs()
    disk=DiskEmbeddingCache.create(tmp_path/'cache',ids,hashes,vectors,RECIPE)
    small=pre.EmbeddingCache(ids,hashes,vectors,RECIPE)
    kwargs=dict(dimension=2,projection_epochs=1,prior_epochs=2,seed=7)
    a=pre.fit_preference_prior(disk,rows,['small','large'],{'small':1.,'large':4.},'fixture',**kwargs)
    b=pre.fit_preference_prior(small,rows,['small','large'],{'small':1.,'large':4.},'fixture',**kwargs)
    np.testing.assert_allclose(a['W'],b['W'])
    np.testing.assert_allclose(pre.project_cache(a,disk),pre.project_cache(b,small))


def test_sampler_does_not_allocate_cartesian_candidates():
    ids,hashes,vectors,rows=inputs(200)
    cache=pre.EmbeddingCache(ids,hashes,vectors,RECIPE)
    tracemalloc.start()
    try:
        sampled=pre.sample_triplets(cache,rows,['small','large'],{'small':1.,'large':4.},'fixture')
        _,peak=tracemalloc.get_traced_memory()
    finally:tracemalloc.stop()
    assert sampled['eligible_anchors']==200
    assert peak<2*1024*1024, f'Candidate allocation grew to {peak} bytes'

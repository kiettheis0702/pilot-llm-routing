import importlib
import hashlib
import numpy as np
import pytest
from scripts.data_gate import query_hash
from scripts.embedding_store import DiskEmbeddingCache


def mod():return importlib.import_module('tools.openai_study_instructor_encode')
def row(q,text):return dict(query_id=q,text=text,prompt_hash=query_hash(text))
def raw_sha(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()


def test_length_sorted_compute_order_restores_original_role_order_and_exact_text_dedup():
    roles={'source-0':[row('r0','long question'),row('r1','x'),row('r2','x')],
           'source-1':[row('p0',' x ')]}
    lengths={raw_sha('long question'):50,raw_sha('x'):16,raw_sha(' x '):16}
    index=mod().build_index(roles,lengths)
    assert len(index['unique'])==3
    assert [r['token_length'] for r in index['unique']]==[16,16,50]
    assert [r['unique_index'] for r in index['roles']['source-0']]==[2,0,0]
    assert index['roles']['source-1'][0]['unique_index']==1
    assert [r['query_id'] for r in index['roles']['source-0']]==['r0','r1','r2']
    assert roles['source-0'][1]['prompt_hash']==roles['source-1'][0]['prompt_hash']


def test_index_refuses_wrong_text_hash_and_missing_length():
    roles={'source-0':[row('q','x')]};roles['source-0'][0]['prompt_hash']='0'*64
    with pytest.raises(ValueError):mod().build_index(roles,{raw_sha('x'):16})
    with pytest.raises(ValueError):mod().build_index({'source-0':[row('q','x')]},{})


def test_role_exports_align_ids_and_vectors_and_refuse_wrong_completed_unique_cache(tmp_path):
    roles={'source-0':[row('r0','long'),row('r1','short'),row('r2','short')],
           'source-1':[row('p0','long')]}
    index=mod().build_index(roles,{raw_sha('long'):50,raw_sha('short'):20})
    recipe=dict(model='synthetic-instructor',version='test',source='unit',normalization='L2',synthetic=True)
    unique=DiskEmbeddingCache.create(tmp_path/'unique',[r['query_id'] for r in index['unique']],
        [r['prompt_hash'] for r in index['unique']],np.array([[1.,0.],[0.,1.]],dtype=np.float32),recipe)
    result=mod().publish_role_caches(unique,index,tmp_path/'roles')
    assert result['source-0'].ids==('r0','r1','r2')
    np.testing.assert_array_equal(result['source-0'].get('r0'),[0.,1.])
    np.testing.assert_array_equal(result['source-0'].get('r1'),[1.,0.])
    np.testing.assert_array_equal(result['source-1'].get('p0'),[0.,1.])
    again=mod().publish_role_caches(unique,index,tmp_path/'roles')
    assert again['source-0'].fingerprint()==result['source-0'].fingerprint()
    wrong=DiskEmbeddingCache.create(tmp_path/'wrong',['wrong0','wrong1'],
        [r['prompt_hash'] for r in index['unique']],np.array([[1.,0.],[0.,1.]],dtype=np.float32),recipe)
    with pytest.raises(ValueError):mod().publish_role_caches(wrong,index,tmp_path/'must-not-exist')
    assert not (tmp_path/'must-not-exist').exists()

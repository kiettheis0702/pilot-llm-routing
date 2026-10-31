import importlib
import json
from pathlib import Path
import numpy as np
import pytest
from scripts.data_gate import query_hash
from scripts.embedding_store import DiskEmbeddingCache


def module():return importlib.import_module('tools.openai_study_instructor_study')


def test_instructor_declaration_keeps_original_full_core_inventory_and_excludes_ablations():
    from scripts.experiments import expected_runs
    source=json.loads(Path('configs/frozen_experiment_protocol.json').read_text(encoding='utf-8'))
    declared=module().instructor_declaration(source)
    conditions=expected_runs(declared)
    assert len(conditions)==858 and len(declared['arms'])==8
    assert {r['domain'] for r in conditions}=={'mmlu','routerbench'}
    assert len({r['method'] for r in conditions})==13
    assert {r['family'] for r in conditions}=={'primary'}
    assert declared['primary_plan']['pretraining']['dimension']==64
    assert declared['primary_plan']['pretraining']['projection_epochs']==30
    assert declared['primary_plan']['pretraining']['prior_epochs']==40
    assert 'Instructor' in declared['primary_plan']['embedding_recipe']


def test_study_checks_completed_cache_identity_before_using_vectors(tmp_path):
    recipe=dict(model='hkunlp/instructor-xl',version='ce48b213095e647a6c3536364b9fa00daf57f436',
        source='unit fixture only',normalization='L2',synthetic=False)
    x=np.zeros((1,768),dtype=np.float32);x[0,0]=1
    cache=DiskEmbeddingCache.create(tmp_path/'cache',['q'],[query_hash('question')],x,recipe)
    rows=[dict(query_id='q',prompt_hash=query_hash('question'),text='question')]
    completed=dict(complete=True,role_caches={'source-1':dict(rows=1,fingerprint=cache.fingerprint())})
    assert module().aligned_instructor(tmp_path/'cache',rows,'source-1',completed).dimension==768
    with pytest.raises(ValueError):module().aligned_instructor(tmp_path/'cache',[dict(query_id='other',prompt_hash=query_hash('question'))],'source-1',completed)
    completed['role_caches']['source-1']['fingerprint']='0'*64
    with pytest.raises(ValueError):module().aligned_instructor(tmp_path/'cache',rows,'source-1',completed)


def test_study_refuses_other_encoder_and_wrong_input_order(tmp_path):
    recipe=dict(model='text-embedding-3-small',version='wrong',source='unit fixture',normalization='L2',synthetic=False)
    cache=DiskEmbeddingCache.create(tmp_path/'cache',['q'],[query_hash('question')],np.ones((1,1536),dtype=np.float32),recipe)
    completed=dict(complete=True,role_caches={'source-0':dict(rows=1,fingerprint=cache.fingerprint())})
    with pytest.raises(ValueError):module().aligned_instructor(tmp_path/'cache',[dict(query_id='q',prompt_hash=query_hash('question'))],'source-0',completed)

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def project(tmp_path):
    from scripts.data_gate import make_splits, query_hash
    from scripts.fetch_data import sha256
    root, data = tmp_path / 'project', tmp_path / 'data'
    mappings = [{'arena': a, 'routerbench': a, 'status': 'exact_id'} for a in ('a', 'b')]
    write_json(root / 'configs/model_mapping.json', {'mappings': mappings})
    path = data / 'arena_55k/train.csv'
    path.parent.mkdir(parents=True)
    pd.DataFrame([dict(id=i, model_a='a', model_b='b',winner_model_a=int(i%2==0),
                      winner_model_b=int(i%2==1),winner_tie=0,prompt=json.dumps([f'p{i}']),
                      response_a='["answer a"]',response_b='["answer b"]') for i in range(10)]).to_csv(path,index=False)
    write_json(root / 'configs/data_sources.json', {'arena_55k': {'size':path.stat().st_size,'sha256':sha256(path)}})
    q = pd.DataFrame(dict(query_id=[f'q{i}' for i in range(12)],query=[f'query {i}' for i in range(12)],task='fixture'))
    q['query_hash']=q['query'].map(query_hash)
    prepared=data/'processed/routerbench_0shot';prepared.mkdir(parents=True)
    q.to_parquet(prepared/'queries.parquet',index=False)
    pd.DataFrame({'query_id':q.query_id.repeat(2),'arm_id':['a','b']*12,'reward':[0.,1.]*12,'actual_cost':.01}).to_parquet(prepared/'outcomes.parquet',index=False)
    make_splits(q,tuning_size=2).to_csv(prepared/'splits.csv',index=False)
    hashes={p.name:sha256(p) for p in prepared.iterdir()}
    write_json(root/'reports/data-gate.json',{'prepared':{'file_sha256':hashes}})
    return root,data


def test_preparation_preserves_sources_and_readbacks(tmp_path):
    from scripts.research_pipeline import prepare_data, load_prepared
    root,data=project(tmp_path)
    output=tmp_path/'output'
    report=prepare_data(data,output,root=root,rows=4,tuning_size=2)
    assert report['eligible_preferences']==10
    assert report['routing_queries']==4
    assert report['paper_equivalent'] is False
    loaded=load_prepared(output)
    assert len(loaded['comparisons'])==10
    assert {r['split'] for r in loaded['comparisons']}=={'train','holdout'}
    with pytest.raises(FileExistsError): prepare_data(data,output,root=root,rows=4,tuning_size=2)
    with (output/'comparisons.json').open('a') as f:f.write(' ')
    with pytest.raises(ValueError,match='checksum'):load_prepared(output)


def test_changed_source_and_full_pool_gate_precede_output(tmp_path):
    from scripts.research_pipeline import prepare_data
    root,data=project(tmp_path)
    path=root/'configs/model_mapping.json'
    config=json.loads(path.read_text());config['mappings'].append({'arena':'unknown','routerbench':'unknown-v1','status':'checkpoint_unverified'})
    write_json(path,config)
    with pytest.raises(ValueError,match='Unresolved'):
        prepare_data(data,tmp_path/'blocked',root=root,rows=4,tuning_size=2,require_all_arms=True)
    assert not (tmp_path/'blocked').exists()
    with (data/'arena_55k/train.csv').open('a') as f:f.write('bad')
    with pytest.raises(ValueError,match='mismatch'):
        prepare_data(data,tmp_path/'changed',root=root,rows=4,tuning_size=2)
    assert not (tmp_path/'changed').exists()


def test_fit_learn_stages_share_lineage_and_reject_changed_cache(tmp_path):
    from scripts.research_pipeline import prepare_data,load_prepared,fit_prepared,learn_prepared
    from scripts.embedding_store import DiskEmbeddingCache
    from scripts.pretraining import load_checkpoint
    root,data=project(tmp_path)
    prepared=tmp_path/'prepared'
    prepare_data(data,prepared,root=root,rows=4,tuning_size=2)
    inputs=load_prepared(prepared)
    recipe={'model':'analytic-fixture','version':'1','source':'unit test','normalization':'none','synthetic':True}
    pref=inputs['preference-inputs']
    # Explicitly marked fixture: favor A on even questions and B on odd ones.
    vectors=[[1. if int(r['query_id'].split(':')[-1])%2==0 else -1.,3.] for r in pref]
    DiskEmbeddingCache.create(tmp_path/'pref', [r['query_id'] for r in pref],
        [r['prompt_hash'] for r in pref],vectors,recipe)
    sizes={'sizes':{'a':4.,'b':1.},'provenance':'synthetic test only','source_urls':['https://example.com/fixture'],'synthetic':True}
    with pytest.raises(ValueError,match='synthetic'):
        fit_prepared(prepared,tmp_path/'pref',sizes,tmp_path/'refused')
    result=fit_prepared(prepared,tmp_path/'pref',sizes,tmp_path/'fit',dimension=2,
                        projection_epochs=20,prior_epochs=50,seed=7,allow_synthetic=True)
    assert result['prior_ready'] is True
    route=inputs['routing-inputs']
    DiskEmbeddingCache.create(tmp_path/'route',[r['query_id'] for r in route],
        [r['prompt_hash'] for r in route],[[1.,3.]]*len(route),recipe)
    report=learn_prepared(prepared,data,tmp_path/'route',tmp_path/'fit',tmp_path/'learning',root=root,tuning_size=2,allow_synthetic=True)
    assert set(report['methods'])=={'pilot','random','linucb','explore-only','epoch-greedy','fixed-00','fixed-01'}
    assert report['rows_per_method']==4
    assert report['paper_equivalent'] is False
    changed=dict(recipe,version='2')
    DiskEmbeddingCache.create(tmp_path/'wrong',[r['query_id'] for r in route],
        [r['prompt_hash'] for r in route],[[1.,3.]]*len(route),changed)
    with pytest.raises(ValueError,match='recipe'):
        learn_prepared(prepared,data,tmp_path/'wrong',tmp_path/'fit',tmp_path/'bad-learn',root=root,tuning_size=2,allow_synthetic=True)
    assert not (tmp_path/'bad-learn').exists()


@pytest.mark.parametrize('description',[None,'','  ',{},False])
def test_fit_rejects_empty_or_nonstring_cost_provenance(tmp_path,description):
    from scripts.research_pipeline import prepare_data,load_prepared,fit_prepared
    from scripts.embedding_store import DiskEmbeddingCache
    root,data=project(tmp_path)
    prepared=tmp_path/'prepared'
    prepare_data(data,prepared,root=root,rows=4,tuning_size=2)
    pref=load_prepared(prepared)['preference-inputs']
    provenance={'model':'fixture','version':'1','source':'test','normalization':'none','synthetic':True}
    DiskEmbeddingCache.create(tmp_path/'pref',[r['query_id'] for r in pref],
        [r['prompt_hash'] for r in pref],[[1.,3.]]*len(pref),provenance)
    sizes={'sizes':{'a':4.,'b':1.},'provenance':description,
           'source_urls':['https://example.com/fixture'],'synthetic':True}
    with pytest.raises(ValueError,match='provenance'):
        fit_prepared(prepared,tmp_path/'pref',sizes,tmp_path/'fit',dimension=2,allow_synthetic=True)
    assert not (tmp_path/'fit').exists()


def test_real_cache_must_match_verbatim_input_and_recipe(tmp_path):
    import numpy as np
    from scripts.data_gate import query_hash
    from scripts.local_embeddings import encode_rows
    from scripts.research_pipeline import _aligned_cache
    recipe=json.loads((Path(__file__).parents[1]/'configs/local_embedding.json').read_text())
    inputs=[dict(query_id='q1',text='code\n    indented',prompt_hash=query_hash('code\n    indented'))]
    changed=[dict(inputs[0],text='code indented')]
    assert query_hash(changed[0]['text'])==inputs[0]['prompt_hash']
    encoder=lambda texts:(np.full((len(texts),1024),1/32,dtype=np.float32),[4]*len(texts))
    encode_rows(changed,tmp_path/'cache',recipe,encoder)
    assert _aligned_cache(tmp_path/'cache',changed).dimension==1024
    with pytest.raises(ValueError,match='verbatim'):
        _aligned_cache(tmp_path/'cache',inputs)

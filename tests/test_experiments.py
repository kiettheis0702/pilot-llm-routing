"""Catch missing conditions, reset-at-prefix bugs, leakage and unsafe resume."""
import copy
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.bandit_contracts import ContextTable
from scripts.cost_policy import TokenPrices, TuningTokenCostEstimator
from scripts.deployment_adapter import FrozenDeploymentRouter, policy_state
from scripts.policies import LinearUCB
from scripts.splits import SplitSelection
from scripts.tuning import build_policy
from test_tuning import inputs


def api():
    assert importlib.util.find_spec('scripts.experiments') is not None, 'Full experiment runner is missing'
    from scripts import experiments
    return experiments


def small_inputs():
    ids=['l0','l1','l2','l3','l4','d0','d1','t0']
    queries=pd.DataFrame(dict(query_id=ids,query=['prompt '+q for q in ids],task='toy'))
    outcomes=pd.DataFrame([dict(query_id=q,arm_id=a,reward=float(a=='b'),actual_cost=.1)
                           for q in ids for a in ('a','b')])
    contexts=ContextTable({q:[1.,.5] for q in ids},label='synthetic test')
    selection=SplitSelection(tuple(ids[:5]),'learning','toy',42,1,10,5)
    return dict(queries=queries,outcomes=outcomes,contexts=contexts,selection=selection,
                seed=7,fractions=[.1,.25,.5,1.],lineage={'synthetic':True})


def test_inventory_contains_all_primary_and_paired_ablation_conditions():
    ex=api();protocol=json.loads(Path('configs/frozen_experiment_protocol.json').read_text())
    rows=ex.expected_runs(protocol)
    assert len(rows)==1098 and len({r['id'] for r in rows})==1098
    assert sum(r['family']=='primary' for r in rows)==858
    for domain,count in [('routerbench',32270),('mmlu',12369)]:
        for seed in [7,42,2026]:
            selected=[r for r in rows if r['domain']==domain and r['seed']==seed]
            assert len(selected)==183
            assert len({r['method'] for r in selected if r['family']=='primary'})==13
            half=[r for r in selected if r['family']=='primary' and r['method']=='pilot' and r['learning_fraction']==.5]
            assert len(half)==1 and half[0]['learning_count']==(count+1)//2
            assert half[0]['budget_fraction']==.5
            for variant in ['zero-prior-direction','unit-initialization-strength','prior-only',
                            'single-budget-bin','projection-dimension-8']:
                assert sum(r['variant']==variant for r in selected)==8


def test_prefix_snapshots_are_one_uninterrupted_learning_pass(tmp_path):
    ex=api();v=small_inputs();model=LinearUCB(('a','b'),2,alpha=1)
    result=ex.learn_stream(model=model,output_dir=tmp_path/'run',**v)
    assert result['metrics']['rounds']==5
    assert [s['rounds'] for s in result['snapshots'].values()]==[1,2,3,5]
    manual=LinearUCB(('a','b'),2,alpha=1);manual.reset(7)
    for i,q in enumerate(v['selection'].query_ids,1):
        x=v['contexts'].get(q);arm=manual.select(x);manual.observe(arm,x,float(arm=='b'))
        for snapshot in result['snapshots'].values():
            if snapshot['rounds']==i:
                loaded=FrozenDeploymentRouter.load_checkpoint(tmp_path/'run'/snapshot['file'])
                assert loaded.source_state()==policy_state(manual)
    trace=[json.loads(s) for s in (tmp_path/'run'/'trace.jsonl').read_text().splitlines()]
    assert [r['query_id'] for r in trace]==['l0','l1','l2','l3','l4']
    assert sum(r['reward'] for r in trace)==result['metrics']['total_reward']
    assert sum(1-r['reward'] for r in trace)==result['metrics']['cumulative_empirical_oracle_regret']


def test_learning_ignores_heldout_labels_and_text(tmp_path):
    ex=api();v=small_inputs()
    a=ex.learn_stream(model=LinearUCB(('a','b'),2),output_dir=tmp_path/'a',**v)
    v['outcomes'].loc[~v['outcomes'].query_id.isin(v['selection'].query_ids),'reward']=float('nan')
    v['queries'].loc[v['queries'].query_id=='d0','query']='different deployment text'
    b=ex.learn_stream(model=LinearUCB(('a','b'),2),output_dir=tmp_path/'b',**v)
    assert a==b
    for name in a['files']:
        assert (tmp_path/'a'/name).read_bytes()==(tmp_path/'b'/name).read_bytes()


def test_resume_verifies_identity_and_every_completed_file(tmp_path):
    ex=api();v=small_inputs();out=tmp_path/'run'
    first=ex.learn_stream(model=LinearUCB(('a','b'),2),output_dir=out,**v)
    before={p.name:p.read_bytes() for p in out.iterdir() if p.is_file()}
    assert ex.learn_stream(model=LinearUCB(('a','b'),2),output_dir=out,resume=True,**v)==first
    assert before=={p.name:p.read_bytes() for p in out.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):ex.learn_stream(model=LinearUCB(('a','b'),2),output_dir=out,**v)
    with pytest.raises(ValueError,match='identity'):
        ex.learn_stream(model=LinearUCB(('a','b'),2,alpha=2),output_dir=out,resume=True,**v)
    (out/'trace.jsonl').write_text('corrupt')
    with pytest.raises(ValueError,match='checksum'):
        ex.learn_stream(model=LinearUCB(('a','b'),2),output_dir=out,resume=True,**v)


@pytest.mark.parametrize('damage',['phase','duplicate','missing_arm','zero_context','bad_fraction','incomplete'])
def test_invalid_learning_cannot_publish_completion(tmp_path,damage):
    ex=api();v=small_inputs();out=tmp_path/'run'
    if damage=='phase':v['selection']=replace(v['selection'],split='deployment')
    elif damage=='duplicate':v['selection']=replace(v['selection'],query_ids=('l0','l0'))
    elif damage=='missing_arm':v['outcomes']=v['outcomes'].iloc[1:]
    elif damage=='zero_context':v['contexts']=ContextTable({'l0':[0.,0.]},label='bad')
    elif damage=='bad_fraction':v['fractions']=[.5,.5,1.]
    else:out.mkdir();(out/'trace.jsonl').write_text('partial')
    with pytest.raises((ValueError,FileExistsError)):
        ex.learn_stream(model=LinearUCB(('a','b'),2),output_dir=out,resume=True,**v)
    assert not (out/'manifest.json').exists()


def test_controlled_priors_only_change_declared_initialization():
    ex=api();prior=inputs()['prior'];original=copy.deepcopy(prior)
    main=build_policy(prior,'pilot',{'alpha':1})
    zero=ex.variant_policy(prior,'zero-prior-direction',{'alpha':1})
    unit=ex.variant_policy(prior,'unit-initialization-strength',{'alpha':1})
    np.testing.assert_array_equal(zero.A,main.A);np.testing.assert_array_equal(zero.b,0*main.b)
    np.testing.assert_array_equal(unit.A,np.repeat(np.eye(2)[None],2,axis=0))
    np.testing.assert_array_equal(unit.b,prior['theta'])
    for key in ('W','b','theta'):np.testing.assert_array_equal(prior[key],original[key])
    with pytest.raises(ValueError):ex.variant_policy(prior,'invented',{'alpha':1})


def test_prior_only_never_observes_learning_labels(tmp_path):
    ex=api();v=small_inputs();v['fractions']=[1.]
    v['outcomes']['reward']=float('nan')
    model=LinearUCB(('a','b'),2)
    result=ex.learn_stream(model=model,output_dir=tmp_path/'run',training_mode='none',**v)
    assert result['metrics']['rounds']==0 and result['training_mode']=='none'
    assert result['learning_selection_role']=='unused_disjoint_reference'
    assert (tmp_path/'run'/'trace.jsonl').read_text()==''
    snap=FrozenDeploymentRouter.load_checkpoint(tmp_path/'run'/result['snapshots']['1']['file'])
    assert snap.source_state()['rounds']==0


def test_deployment_manifest_preserves_budget_freeze_and_rejects_resume_changes(tmp_path):
    ex=api();v=small_inputs();model=LinearUCB(('a','b'),2);model.reset(7)
    router=FrozenDeploymentRouter(model,seed=7);before=router.source_state()
    selection=replace(v['selection'],query_ids=('d0','d1'),split='deployment',full_size=2)
    tuning=replace(v['selection'],query_ids=('t0',),split='tuning',full_size=1)
    estimator=TuningTokenCostEstimator.fit(tuning,{'t0':{'a':0,'b':0}},
        {a:TokenPrices(1_000_000,0) for a in ('a','b')},token_provenance='toy',price_provenance='toy')
    args=dict(router=router,queries=v['queries'],outcomes=v['outcomes'],contexts=v['contexts'],
        learning_selection=v['selection'],deployment_selection=selection,estimator=estimator,
        input_tokens={'d0':1,'d1':1},input_token_provenance='toy',budget=0.,bin_size=1,
        lower_bound=.1,upper_bound=10.,lineage={'synthetic':True},output_dir=tmp_path/'run')
    result=ex.deploy_run(**args)
    assert result['summary']['served_queries']==0
    assert result['summary']['metrics']['mean_reward_served'] is None
    assert result['summary']['metrics']['reward_per_requested_query']==0.
    assert router.source_state()==before
    assert ex.deploy_run(**args,resume=True)==result
    with pytest.raises(ValueError,match='identity'):ex.deploy_run(**dict(args,budget=1),resume=True)


def group_inputs():
    v=small_inputs();prior=inputs()['prior']
    # Translate the hand-sized environment into the prior's two-arm catalog.
    arms=prior['arms'];v['outcomes']['arm_id']=v['outcomes'].arm_id.map(dict(zip(('a','b'),arms)))
    p=json.loads(Path('configs/frozen_experiment_protocol.json').read_text())
    p['arms']=arms;p['primary_plan']['pretraining']['dimension']=2
    p['ablation_plan']['inventory']['primary_plus_ablation_deployment_runs']=702
    for domain in p['primary_plan']['datasets']:
        p['primary_plan']['frozen_counts'][domain]={'tuning':1,'learning':5,'deployment':2}
        p['tuning'][domain]['selected_parameters']={
            'pilot':{'alpha':1.},'linucb':{'alpha':5.},'epoch-greedy':{'window':10},
            'explore-only':{},'random':{},**{f'fixed-{i:02d}':{'arm_id':a} for i,a in enumerate(arms)}}
        p['tuning'][domain]['absolute_reference_budget']=2.
    tuning=replace(v['selection'],query_ids=('t0',),split='tuning',full_size=1)
    deployment=replace(v['selection'],query_ids=('d0','d1'),split='deployment',full_size=2)
    estimator=TuningTokenCostEstimator.fit(tuning,{'t0':dict.fromkeys(arms,0)},
        {a:TokenPrices(1_000_000,0) for a in arms},token_provenance='toy',price_provenance='toy')
    return dict(protocol=p,prior=prior,domain='routerbench',seed=7,queries=v['queries'],outcomes=v['outcomes'],
        contexts=v['contexts'],selections={'learning':v['selection'],'deployment':deployment,'tuning':tuning},
        estimator=estimator,input_counts={'d0':1,'d1':1},input_token_provenance='toy',lineage={'synthetic':True})


def test_group_executes_full_budget_grid_and_reuses_main_for_single_bin(tmp_path):
    ex=api();assert hasattr(ex,'execute_group'),'Group orchestration is missing'
    bundle=group_inputs();gate=lambda: {'available_gib':16,'heavy_jobs':[]}
    primary=ex.execute_group(bundle,tmp_path,method='pilot',variant='main',preflight=gate)
    assert len(primary)==11
    full=[r for r in primary if r['request']['lineage']['condition']['learning_fraction']==1]
    assert len(full)==8
    assert sorted(r['request']['budget'] for r in full)==[.1,.2,.4,.6,1.,1.5,2.,2.5]
    main_source={r['request']['lineage']['learning_checkpoint_sha256'] for r in full}
    single=ex.execute_group(bundle,tmp_path,method='pilot',variant='single-budget-bin',preflight=gate)
    assert len(single)==8 and {r['request']['bin_size'] for r in single}=={2}
    assert {r['request']['lineage']['learning_checkpoint_sha256'] for r in single}==main_source
    initial=ex.execute_group(bundle,tmp_path,method='pilot',variant='prior-only',preflight=gate)
    assert len(initial)==8
    assert all(r['request']['lineage']['learning_rounds']==0 for r in initial)
    assert all(r['request']['lineage']['learning_selection_role']=='unused_disjoint_reference' for r in initial)
    assert ex.execute_group(bundle,tmp_path,method='pilot',variant='main',preflight=gate,resume=True)==primary


def test_single_bin_refuses_missing_main_and_resource_failure_starts_nothing(tmp_path):
    ex=api();assert hasattr(ex,'execute_group'),'Group orchestration is missing'
    with pytest.raises(ValueError,match='main'):
        ex.execute_group(group_inputs(),tmp_path/'a',method='pilot',variant='single-budget-bin',preflight=lambda: {})
    def blocked():raise RuntimeError('RAM below gate')
    with pytest.raises(RuntimeError,match='RAM'):
        ex.execute_group(group_inputs(),tmp_path/'b',method='pilot',variant='main',preflight=blocked)
    assert not (tmp_path/'b').exists()


def test_pin_loader_rejects_changed_file_and_parent_escape(tmp_path):
    ex=api();assert hasattr(ex,'verify_pins'),'Pinned lineage loader is missing'
    from scripts.fetch_data import sha256
    (tmp_path/'input.json').write_text('original')
    value={'nested':[{'path':'input.json','sha256':sha256(tmp_path/'input.json')}]}
    ex.verify_pins(value,tmp_path)
    (tmp_path/'input.json').write_text('modified')
    with pytest.raises(ValueError,match='checksum'):ex.verify_pins(value,tmp_path)
    with pytest.raises(ValueError,match='path'):ex.verify_pins({'path':'../outside','sha256':'a'*64},tmp_path)


def test_real_loader_refuses_tampered_protocol_before_data_access(tmp_path):
    ex=api();assert hasattr(ex,'load_real_bundle'),'Real data loader is missing'
    p=json.loads(Path('configs/frozen_experiment_protocol.json').read_text());p['arms'].pop()
    (tmp_path/'protocol.json').write_text(json.dumps(p))
    with pytest.raises(ValueError,match='checksum'):
        ex.load_real_bundle(tmp_path/'protocol.json',tmp_path/'absent-index.json',domain='routerbench',seed=7,root=tmp_path)


def test_real_loader_rejects_synthetic_prior_index(tmp_path):
    ex=api();assert hasattr(ex,'load_real_bundle'),'Real data loader is missing'
    from scripts.pretraining import _digest
    p={'kind':'frozen-local-experiment-protocol','version':1,'synthetic':False}
    p['manifest_sha256']=_digest(p)
    index={'kind':'verified-research-prior-index','version':1,'synthetic':True}
    index['manifest_sha256']=_digest(index)
    (tmp_path/'protocol.json').write_text(json.dumps(p));(tmp_path/'index.json').write_text(json.dumps(index))
    with pytest.raises(ValueError,match='synthetic'):
        ex.load_real_bundle(tmp_path/'protocol.json',tmp_path/'index.json',domain='routerbench',seed=7,root=tmp_path)


def test_resource_preflight_detects_experiment_process():
    import os,subprocess,sys,time
    if os.name!='nt':pytest.skip('Windows resource preflight')
    from scripts.resources import resource_snapshot
    # Sleeping sentinel has no training workload; exercise real process discovery.
    with subprocess.Popen([sys.executable,'-c','import time; time.sleep(4)','scripts.experiments'],
                          stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL) as child:
        time.sleep(.1)
        snapshot=resource_snapshot()
        assert child.pid in {p['pid'] for p in snapshot['heavy_jobs']}

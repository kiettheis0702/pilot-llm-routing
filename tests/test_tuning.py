import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.bandit_contracts import ContextTable
from scripts.cost_policy import TokenPrices, TuningTokenCostEstimator
from scripts.pretraining import fit_preference_prior
from scripts.splits import SplitSelection
from test_pretraining import fixture,ARMS,SIZES,SOURCE


def inputs():
    cache,comparisons=fixture()
    prior=fit_preference_prior(cache,comparisons,ARMS,SIZES,SOURCE,dimension=2,
        projection_epochs=30,prior_epochs=40,margin=.5,seed=7)
    ids=['t0','t1','t2','t3','learning','deployment']
    queries=pd.DataFrame(dict(query_id=ids,query=['prompt '+q for q in ids],task='toy'))
    outcomes=pd.DataFrame([dict(query_id=q,arm_id=a,reward=float(a=='large'),actual_cost=.1)
        for q in ids for a in ARMS])
    contexts=ContextTable({q:[1.,.2] for q in ids},label='synthetic fixture')
    selection=SplitSelection(tuple(ids[:4]),'tuning','toy-sha256',42,4,10,4)
    grid=dict(version=1,seed=42,alpha_grid=[1.,2.],window_grid=[2,4])
    return dict(prior=prior,queries=queries,outcomes=outcomes,contexts=contexts,
                selection=selection,grid=grid,allow_synthetic=True)


def test_full_candidate_inventory_scores_ties_and_reload(tmp_path):
    from scripts.tuning import run_search,load_search
    values=inputs();values['outcomes']['reward']=0.
    result=run_search(**values,output_dir=tmp_path/'run')
    assert result['phase']=='tuning' and len(result['candidates'])==10
    assert len(result['selected'])==7
    assert result['selected']['pilot']['parameters']=={'alpha':1.}
    assert result['selected']['epoch-greedy']['parameters']=={'window':2}
    assert result['score_rule']=='maximum_prequential_total_reward; first_grid_entry_on_exact_tie'
    restored=load_search(tmp_path/'run')
    assert restored==result
    for candidate in result['candidates']:
        assert candidate['rounds']==4 and candidate['total_reward']==0.
        trace=[json.loads(line) for line in (tmp_path/'run'/candidate['trace']).read_text().splitlines()]
        assert [row['query_id'] for row in trace]==list(values['selection'].query_ids)
        assert all(set(row['predictions'])==set(ARMS) for row in trace)
    with pytest.raises(FileExistsError):run_search(**values,output_dir=tmp_path/'run')


def test_deployment_mutation_cannot_change_any_tuning_artifact(tmp_path):
    from scripts.tuning import run_search
    values=inputs()
    first=run_search(**values,output_dir=tmp_path/'a')
    modified=values['outcomes'].copy()
    mask=~modified.query_id.isin(values['selection'].query_ids)
    modified.loc[mask,'reward']=float('nan');modified.loc[mask,'actual_cost']=-5
    values['outcomes']=modified
    values['queries'].loc[values['queries'].query_id=='deployment','query']='changed held-out text'
    second=run_search(**values,output_dir=tmp_path/'b')
    assert first==second
    files=[str(p.relative_to(tmp_path/'a')) for p in (tmp_path/'a').rglob('*') if p.is_file()]
    for name in files: assert (tmp_path/'a'/name).read_bytes()==(tmp_path/'b'/name).read_bytes()


@pytest.mark.parametrize('damage',['phase','duplicate','missing_query','missing_arm','bad_reward','bad_context','grid_duplicate','grid_bool','grid_empty','not_ready','synthetic'])
def test_invalid_inputs_reject_before_output(tmp_path,damage):
    from scripts.tuning import run_search
    values=inputs()
    if damage=='phase':values['selection']=replace(values['selection'],split='deployment')
    elif damage=='duplicate':values['selection']=replace(values['selection'],query_ids=('t0','t0'))
    elif damage=='missing_query':values['queries']=values['queries'][values['queries'].query_id!='t1']
    elif damage=='missing_arm':values['outcomes']=values['outcomes'].iloc[1:]
    elif damage=='bad_reward':values['outcomes'].loc[0,'reward']=float('nan')
    elif damage=='bad_context':values['contexts']=ContextTable({'t0':[1.,0.]},label='partial')
    elif damage=='grid_duplicate':values['grid']['alpha_grid']=[1,1]
    elif damage=='grid_bool':values['grid']['window_grid']=[True]
    elif damage=='grid_empty':values['grid']['alpha_grid']=[]
    elif damage=='not_ready':values['prior']['metrics']['validation']['all_lambdas_ready']=False
    else:values['allow_synthetic']=False
    with pytest.raises(ValueError):run_search(**values,output_dir=tmp_path/'run')
    assert not (tmp_path/'run').exists()


def test_tuning_only_observes_selected_arm_and_predictions_precede_feedback(tmp_path,monkeypatch):
    from scripts.tuning import run_search
    from scripts.pilot import PILOT
    events=[];original=PILOT.observe
    def observe(self,arm,context,reward):
        events.append((arm,float(reward)))
        return original(self,arm,context,reward)
    monkeypatch.setattr(PILOT,'observe',observe)
    values=inputs();result=run_search(**values,output_dir=tmp_path/'run')
    assert len(events)==8
    for candidate in result['candidates']:
        if candidate['method']!='pilot':continue
        trace=[json.loads(line) for line in (tmp_path/'run'/candidate['trace']).read_text().splitlines()]
        assert [(r['arm_id'],r['reward']) for r in trace]==events[:4];del events[:4]
        assert trace[0]['round']==1 and trace[0]['rounds_before_feedback']==0
    assert not events


def test_loader_detects_missing_or_changed_trace(tmp_path):
    from scripts.tuning import run_search,load_search
    values=inputs();result=run_search(**values,output_dir=tmp_path/'run')
    target=tmp_path/'run'/result['candidates'][0]['trace']
    target.write_text(target.read_text()+'\n')
    with pytest.raises(ValueError,match='checksum'):load_search(tmp_path/'run')


def test_budget_calibration_uses_tuning_predictions_and_estimates_only(tmp_path):
    from scripts.tuning import run_search,calibrate_budget
    values=inputs();result=run_search(**values,output_dir=tmp_path/'run')
    ids=values['selection'].query_ids
    estimator=TuningTokenCostEstimator.fit(values['selection'],{q:dict.fromkeys(ARMS,2) for q in ids},
        {a:TokenPrices(1_000_000,0) for a in ARMS},token_provenance='toy',price_provenance='toy')
    counts=dict.fromkeys(ids,2)
    calibrated=calibrate_budget(tmp_path/'run',counts,estimator,reference_arm='large')
    positive=[]
    for chosen in result['selected'].values():
        for line in (tmp_path/'run'/chosen['trace']).read_text().splitlines():
            row=json.loads(line)
            positive.extend(v/2 for v in row['predictions'].values() if v>0)
    assert calibrated['lower_bound']==pytest.approx(min(positive)*.5)
    assert calibrated['upper_bound']==pytest.approx(max(positive)*2)
    assert calibrated['reference_estimated_cost_per_query']==2.
    assert calibrated['tuning_query_count']==4
    assert calibrated['positive_ratio_count']==len(positive)
    with pytest.raises(ValueError):calibrate_budget(tmp_path/'run',dict(counts,deployment=1),estimator,reference_arm='large')


def test_trial_parameters_are_not_carried_to_main_learning(tmp_path):
    from scripts.tuning import run_search,build_policy
    values=inputs();result=run_search(**values,output_dir=tmp_path/'run')
    chosen=result['selected']['pilot']
    fresh=build_policy(values['prior'],chosen['method'],chosen['parameters'])
    assert fresh.rounds==0
    fresh.reset(42)
    assert fresh.rounds==0

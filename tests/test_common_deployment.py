import copy
import hashlib
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from scripts.bandit_contracts import ContextTable
from scripts.cost_policy import BinnedCostPolicy, TokenPrices, TuningTokenCostEstimator
from scripts.deployment import run_deployment
from scripts.pilot import PILOT
from scripts.policies import LinearUCB, ExploreOnly, EpochGreedy, RandomPolicy, FixedPolicy
from scripts.splits import SplitSelection


def budget(**overrides):
    settings = dict(arm_catalog=('a','b'),budget=10,query_count=2,bin_size=2,
                    lower_bound=.01,upper_bound=1)
    settings.update(overrides)
    return BinnedCostPolicy(**settings)


def models():
    return [PILOT(('a','b'),2,priors={'a':[1,0],'b':[0,1]},lambdas={'a':2,'b':1},
                  provenance='synthetic prior',alpha=100),
            LinearUCB(('a','b'),2,alpha=100), ExploreOnly(('a','b'),2),
            EpochGreedy(('a','b'),2,window=2), RandomPolicy(('a','b'),2),
            FixedPolicy(('a','b'),2,arm_id='b')]


def inputs():
    def selection(phase,ids):
        return SplitSelection(tuple(ids),phase,'synthetic-manifest',42,1,10,len(ids))
    ids=['t','l','d1','d2','d3']
    queries=pd.DataFrame({'query_id':ids,'query':['query '+q for q in ids],'task':'toy'})
    outcomes=pd.DataFrame([dict(query_id=q,arm_id=a,reward=.8 if a=='a' else .4,
                               actual_cost=.25 if a=='a' else .5) for q in ids for a in ('a','b')])
    estimator=TuningTokenCostEstimator.fit(selection('tuning',['t']),
        {'t':{'a':1,'b':1}},{a:TokenPrices(1_000_000,0) for a in ('a','b')},
        token_provenance='synthetic tuning',price_provenance='synthetic prices')
    return dict(queries=queries,outcomes=outcomes,
        contexts=ContextTable({'d1':[1,0],'d2':[0,1],'d3':[1,1]},label='synthetic'),
        learning_selection=selection('learning',['l']),
        deployment_selection=selection('deployment',['d1','d2','d3']),
        estimator=estimator,input_tokens={'d1':1,'d2':1,'d3':1},
        input_token_provenance='synthetic query-only counts',budget=6,bin_size=2,
        lower_bound=.01,upper_bound=1)


def test_linear_score_domain_preserves_strict_order_without_clipping():
    policy=budget(score_domain='linear')
    first=policy.select({'a':2.,'b':3.},{'a':1.,'b':1.})
    assert first.arm_id=='b' and first.predicted_reward==3.
    second=policy.select({'a':-3.,'b':-2.},{'a':1.,'b':1.})
    assert second.arm_id=='b' and second.branch=='fallback'
    with pytest.raises(ValueError,match='cosine'):
        budget().select({'a':2.,'b':3.},{'a':1.,'b':1.})


def test_explicit_ties_and_fixed_allowed_arm_apply_in_both_branches():
    policy=budget()
    decision=policy.select({'a':1.,'b':1.},{'a':1.,'b':1.},tie_break_order=('b','a'))
    assert decision.arm_id=='b'
    decision=policy.select({'a':.9,'b':.8},{'a':1.,'b':1.},tie_break_order=('b','a'))
    assert decision.arm_id=='a'
    for rewards in ({'a':1.,'b':1.},{'a':-.5,'b':-.1}):
        decision=budget(budget=1).select(rewards,{'a':.1,'b':2.},allowed_arms=('b',))
        assert decision.status=='insufficient_budget' and decision.eligible_arms==()


@pytest.mark.parametrize('options',[
    {'allowed_arms':()}, {'allowed_arms':('a','a')}, {'allowed_arms':('missing',)},
    {'allowed_arms':'a'}, {'tie_break_order':('a',)},
    {'tie_break_order':('b','b')}, {'tie_break_order':('a','unknown')},
])
def test_invalid_selection_options_do_not_reserve_budget(options):
    policy=budget();before=policy.report()
    with pytest.raises(ValueError):
        policy.select({'a':1.,'b':1.},{'a':1.,'b':1.},**options)
    assert policy.report()==before and policy.decisions==()


def test_unknown_domain_and_nonfinite_linear_scores_fail():
    with pytest.raises(ValueError): budget(score_domain='ucb')
    policy=budget(score_domain='linear');before=policy.report()
    with pytest.raises(ValueError): policy.select({'a':float('inf'),'b':0.},{'a':1.,'b':1.})
    assert policy.report()==before


@pytest.mark.parametrize('index',range(6))
def test_snapshot_matches_frozen_exploitation_and_is_independent(index,tmp_path):
    from scripts.deployment_adapter import FrozenDeploymentRouter, policy_state
    model=models()[index]
    model.reset(23);model.observe('a',[1,0],.75);model.observe('b',[0,1],.5)
    before=policy_state(model)
    frozen=FrozenDeploymentRouter(model,seed=17)
    context=np.array([2.,1.])
    expected=model.predict_rewards(context)
    assert frozen.predict_rewards(context)==pytest.approx(expected)
    assert policy_state(model)==before
    frozen.save_checkpoint(tmp_path/'frozen.json')
    restored=FrozenDeploymentRouter.load_checkpoint(tmp_path/'frozen.json')
    assert restored.metadata()==frozen.metadata()
    assert restored.predict_rewards(context)==pytest.approx(expected)
    assert restored.selection_options('q1')==frozen.selection_options('q1')
    state_copy=restored.source_state();state_copy['rounds']=999
    model.reset(99)
    assert restored.source_state()['rounds']==2
    assert frozen.predict_rewards(context)==pytest.approx(expected)
    assert frozen.metadata()==restored.metadata()
    with pytest.raises(FileExistsError): frozen.save_checkpoint(tmp_path/'frozen.json')


def test_snapshot_rejects_unsupported_models_and_invalid_state(tmp_path):
    from scripts.deployment_adapter import FrozenDeploymentRouter
    with pytest.raises(ValueError): FrozenDeploymentRouter(object())
    with pytest.raises(ValueError): FrozenDeploymentRouter(models()[0],seed=True)
    model=models()[1];model._A[0,0,0]=-1
    with pytest.raises(ValueError): FrozenDeploymentRouter(model)
    frozen=FrozenDeploymentRouter(models()[1]);path=tmp_path/'frozen.json'
    frozen.save_checkpoint(path)
    document=json.loads(path.read_text());document['payload']['seed']=123
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError,match='checksum'): FrozenDeploymentRouter.load_checkpoint(path)


def test_random_per_query_permutation_is_repeatable_and_uses_no_learned_rng():
    from scripts.deployment_adapter import FrozenDeploymentRouter, policy_state
    model=models()[4];before=policy_state(model)
    router=FrozenDeploymentRouter(model,seed=42)
    orders=[router.selection_options('q'+str(i))['tie_break_order'] for i in range(40)]
    assert set(orders)=={('a','b'),('b','a')}
    assert orders==[router.selection_options('q'+str(i))['tie_break_order'] for i in range(40)]
    assert policy_state(model)==before
    assert router.predict_rewards([1,0])=={'a':.5,'b':.5}
    restricted=budget(budget=1,query_count=1)
    decision=restricted.select(router.predict_rewards([1,0]),{'a':.5,'b':2},
                                **router.selection_options('q0'))
    assert decision.arm_id=='a'


def test_common_pilot_matches_legacy_decisions_and_reports_coverage():
    from scripts.deployment import run_policy_deployment
    model=models()[0]
    first=run_deployment(model,**inputs())
    second=run_policy_deployment(model,**inputs(),deployment_seed=42)
    assert first.trace==second.trace
    assert second.summary['metrics']['served_fraction']==1
    assert second.summary['metrics']['reward_per_requested_query']==pytest.approx(2/3)
    assert second.summary['deployment_adapter']['score_domain']=='cosine'
    assert second.summary['deployment_adapter']['source_model_frozen'] is True


def test_linear_runner_does_not_clip_scores_or_apply_exploration_bonus():
    from scripts.deployment import run_policy_deployment
    model=models()[1]
    model.observe('a',[1,0],1.)
    for _ in range(4): model.observe('b',[1,0],1.)
    values=inputs();values['contexts']=ContextTable(dict.fromkeys(('d1','d2','d3'),[10,0]),label='synthetic high norm')
    result=run_policy_deployment(model,**values)
    assert [r.decision.arm_id for r in result.trace]==['b']*3
    assert [r.decision.predicted_reward for r in result.trace]==pytest.approx([8.]*3)
    assert result.summary['deployment_adapter']['score_domain']=='linear'
    assert result.summary['prediction_diagnostics']['above_one']==6
    with pytest.raises(ValueError): run_deployment(model,**values)


@pytest.mark.parametrize('index',range(6))
def test_common_runner_outcomes_do_not_change_decisions_or_model(index,monkeypatch):
    from scripts.deployment import run_policy_deployment
    from scripts.deployment_adapter import policy_state
    model=models()[index]
    model.observe('a',[1,0],.8);model.observe('b',[0,1],.4)
    before=policy_state(model)
    def forbidden(*args,**kwargs): raise AssertionError('deployment touched learning method')
    for method in ('select','observe','reset','predict_rewards'):
        monkeypatch.setattr(model,method,forbidden)
    first=run_policy_deployment(model,**inputs(),deployment_seed=42)
    values=inputs();values['outcomes']['reward']=.01;values['outcomes']['actual_cost']=100
    second=run_policy_deployment(model,**values,deployment_seed=42)
    assert [r.decision for r in first.trace]==[r.decision for r in second.trace]
    assert second.summary['budget']['actual_budget_violation_observed'] is True
    assert policy_state(model)==before
    assert first.summary['model']['sha256']==second.summary['model']['sha256']
    assert first.summary['context']==second.summary['context']
    json.dumps(asdict(second),allow_nan=False)


def test_all_to_one_never_falls_back_to_another_model_and_reports_unserved():
    from scripts.deployment import run_policy_deployment
    values=inputs();values['budget']=1
    values['input_tokens']=dict.fromkeys(('d1','d2','d3'),{'a':0,'b':2})
    result=run_policy_deployment(models()[5],**values)
    assert result.trace==()
    assert result.summary['metrics']['mean_reward_served'] is None
    assert result.summary['metrics']['reward_per_requested_query']==0
    assert result.summary['metrics']['served_fraction']==0
    assert result.summary['terminal_decision']['query_id']=='d1'


def test_invalid_future_linear_prediction_fails_before_reservation(monkeypatch):
    from scripts.deployment import run_policy_deployment
    model=models()[1];model._b[:]=10
    values=inputs();values['contexts']=ContextTable({'d1':[1,0],'d2':[0,1],'d3':[1e308,1e308]},label='overflow')
    def forbidden(*args,**kwargs): raise AssertionError('reserved before complete preflight')
    monkeypatch.setattr(BinnedCostPolicy,'select',forbidden)
    with pytest.raises(ValueError,match='finite'): run_policy_deployment(model,**values)

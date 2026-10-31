"""Free local paper extensions: exact source recipes and leakage-safe views."""
import copy
import numpy as np
import pytest


def test_instructor_masks_instruction_then_applies_dense_and_normalizes():
    from tools.openai_study_instructor import pool_dense_normalize
    tokens=np.array([[[1000.,0.],[1.,1.],[1.,3.]]])
    weight=np.array([[2.,0.],[0.,1.]])
    actual=pool_dense_normalize(tokens,np.array([[1,1,1]]),[1],weight)
    np.testing.assert_allclose(actual,[[1/np.sqrt(2),1/np.sqrt(2)]])
    actual=pool_dense_normalize(tokens,np.array([[1,1,0]]),[1],weight)
    np.testing.assert_allclose(actual,[[2/np.sqrt(5),1/np.sqrt(5)]])


@pytest.mark.parametrize('damage',['empty','nonfinite','dimension'])
def test_instructor_rejects_invalid_vectors_instead_of_publishing_cache(damage):
    from tools.openai_study_instructor import pool_dense_normalize
    tokens=np.ones((1,2,2));mask=np.ones((1,2));weight=np.eye(2)
    if damage=='empty':mask[:]=0
    elif damage=='nonfinite':tokens[0,1,0]=np.nan
    else:weight=np.ones((2,3))
    with pytest.raises(ValueError):pool_dense_normalize(tokens,mask,[0],weight)


def test_instructor_text_recipe_matches_official_pair_tokenizer():
    from tools.openai_study_instructor import format_instructor_input
    assert format_instructor_input(' Represent: ',' Question ')=='Represent:Question'
    with pytest.raises(ValueError):format_instructor_input('Represent:','  ')


def test_routellm_mf_uses_normalized_model_vectors_and_unnormalized_query_projection():
    from tools.openai_study_routellm import mf_scores
    models=np.array([[3.,0.],[0.,4.]])
    projection=np.array([[1.,0.,0.],[0.,1.,0.]])
    classifier=np.array([.5,2.]);query=np.array([1.,2.,10.])
    np.testing.assert_allclose(mf_scores(query,models,projection,classifier),[.5,4.])
    np.testing.assert_allclose(mf_scores(3*query,models,projection,classifier),[1.5,12.])


def test_routellm_mf_rejects_invalid_shapes_and_zero_model_direction():
    from tools.openai_study_routellm import mf_scores
    with pytest.raises(ValueError):mf_scores(np.ones(3),np.zeros((2,2)),np.ones((2,3)),np.ones(2))
    with pytest.raises(ValueError):mf_scores(np.ones(4),np.ones((2,2)),np.ones((2,3)),np.ones(2))


def prior_fixture():
    return dict(arms=['a','b','c'],W=np.eye(2),b=np.zeros(2),theta=np.array([[1.,0.],[0.,1.],[1.,1.]]),
        metrics=dict(validation=dict(per_arm={a:dict(support=4,correct=2,accuracy=.5,**{'lambda':2.}) for a in ('a','b','c')},all_lambdas_ready=True,blockers={})))


def test_binary_view_preserves_shared_projection_and_selects_only_verified_arms():
    from tools.openai_study_binary import restrict_prior
    parent=prior_fixture();before=copy.deepcopy(parent)
    view=restrict_prior(parent,['c','a'])
    assert view['arms']==['c','a']
    np.testing.assert_array_equal(view['theta'],parent['theta'][[2,0]])
    np.testing.assert_array_equal(view['W'],parent['W'])
    assert set(view['metrics']['validation']['per_arm'])=={'c','a'}
    for field in ('theta','W','b'):np.testing.assert_array_equal(parent[field],before[field])
    with pytest.raises(ValueError):restrict_prior(parent,['a','unverified'])
    with pytest.raises(ValueError):restrict_prior(parent,['a','a'])


def test_noise_is_uniform_replacement_probability_not_forced_flip_probability():
    from tools.openai_study_robustness import apply_noise
    same=apply_noise(1.,.05,.03,1)
    assert same==dict(feedback=1.,replaced=True,changed=False)
    flipped=apply_noise(1.,.05,.03,0)
    assert flipped==dict(feedback=0.,replaced=True,changed=True)
    assert apply_noise(1.,0.,0.,0)==dict(feedback=1.,replaced=False,changed=False)
    assert apply_noise(1.,.01,.03,0)['feedback']==1.


@pytest.mark.parametrize('args',[(.5,.05,.01,0),(1.,-.1,.03,0),(1.,.1,1.,0),(1.,.1,.03,2)])
def test_noise_rejects_nonbinary_feedback_and_invalid_probabilities(args):
    from tools.openai_study_robustness import apply_noise
    with pytest.raises(ValueError):apply_noise(*args)


def test_drift_has_disjoint_warmup_before_and_after_with_source_order_preserved():
    from tools.openai_study_robustness import make_drift_stream
    ids=['m0','g0','m1','g1','m2','g2','m3']
    tasks={q:'grade-school-math' if q[0]=='g' else 'mmlu-law' for q in ids}
    stream=make_drift_stream(ids,tasks,warm_count=2,before_count=2,after_count=3)
    assert stream==dict(warm=['m0','m1'],before=['m2','m3'],after=['g0','g1','g2'])
    with pytest.raises(ValueError):make_drift_stream(ids,tasks,warm_count=3,before_count=3,after_count=3)

"""Selected noisy feedback must not replace true evaluation reward."""
import importlib
import numpy as np
import pytest
from scripts.pilot import PILOT
from scripts.bandit_contracts import ContextTable


def test_uniform_replacement_updates_only_selected_arm_but_scores_true_reward():
    noise=importlib.import_module('tools.openai_study_noise')
    model=PILOT(['a','b'],2,priors={'a':[1.,0.],'b':[0.,1.]},
        lambdas={'a':1.,'b':1.},provenance='synthetic test',alpha=0.)
    model.reset(42)
    context=ContextTable({'q':np.array([2.,0.])},label='synthetic')
    def feedback(q,a):
        assert (q,a)==('q','a')
        return 1.,.25
    result,trace=noise.learn_noise(model,['q'],context,feedback,{'q':1.},epsilon=1.,
        draws={'q':(0.,0)})
    assert trace[0]['feedback']==0. and trace[0]['reward']==1.
    assert result['total_reward']==1. and result['cumulative_empirical_oracle_regret']==0.
    assert result['noise_replaced']==1 and result['noise_changed']==1
    np.testing.assert_array_equal(model.A[0],[[2.,0.],[0.,1.]])
    np.testing.assert_array_equal(model.A[1],np.eye(2))
    np.testing.assert_array_equal(model.b,[[1.,0.],[0.,1.]])


def test_noise_draw_is_query_keyed_and_independent_of_stream_order():
    noise=importlib.import_module('tools.openai_study_noise')
    before={q:noise.draw_for_query(42,q) for q in ['q1','q2']}
    after={q:noise.draw_for_query(42,q) for q in ['q2','q1']}
    assert before==after
    assert before!= {q:noise.draw_for_query(7,q) for q in ['q1','q2']}
    assert all(0<=u<1 and bit in (0,1) for u,bit in before.values())


def test_zero_noise_keeps_feedback_and_never_reports_forced_flip():
    noise=importlib.import_module('tools.openai_study_noise')
    model=PILOT(['a'],1,priors={'a':[1.]},lambdas={'a':1.},provenance='synthetic',alpha=0.)
    context=ContextTable({'q':np.ones(1)},label='synthetic')
    result,trace=noise.learn_noise(model,['q'],context,lambda q,a:(1.,.1),{'q':1.},
        epsilon=0.,draws={'q':(0.,0)})
    assert trace[0]['feedback']==1.
    assert result['noise_replaced']==result['noise_changed']==0
    with pytest.raises(ValueError):noise.draw_for_query(-1,'q')

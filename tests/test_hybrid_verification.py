import numpy as np
import pytest

from tools.openai_study_hybrid_verify import cheap_probabilities, native_front


def test_native_front_uses_same_query_gpt4_totals_and_full_coverage():
    ids = ['q0', 'q1']
    truth = {('q0','cheap'):(1.,1.), ('q0','strong'):(1.,10.),
             ('q1','cheap'):(0.,1.), ('q1','strong'):(1.,10.)}
    logits = np.log(np.asarray([[.8,.2],[.2,.8]]))
    front = native_front(ids, logits, truth, ('cheap','strong'), [0., .5, 1.])
    assert [r['quality_fraction_gpt4'] for r in front] == [.5, 1., 1.]
    assert [r['cost_fraction_gpt4'] for r in front] == [.1, .55, 1.]
    assert all(r['requested'] == r['served'] == 2 and r['coverage'] == 1 for r in front)
    del truth['q1','cheap']
    with pytest.raises(ValueError, match='outcome'):
        native_front(ids, logits, truth, ('cheap','strong'), [0.])


def test_classifier_probabilities_are_stable_and_reject_nonfinite_predictions():
    np.testing.assert_allclose(cheap_probabilities([[1000.,1000.],[1000.,-1000.]]), [.5,1.])
    with pytest.raises(ValueError):
        cheap_probabilities([[float('nan'),0.]])

import importlib
import numpy as np
import pytest


def test_binary_verifier_selects_parent_rows_by_names_without_changing_projection():
    module=importlib.import_module('tools.openai_study_binary_verify')
    source="theta=None\nif True:\n    theta=np.array(prior['theta'])\nW=np.array(prior['W'])\n"
    adapted=module.adapt_shared_prior(source)
    namespace=dict(np=np,prior={'arms':['outside','large','small'],'theta':[[9,9],[1,0],[0,1]],'W':[[2,3],[4,5]]},arms=['small','large'])
    exec(adapted,namespace)
    np.testing.assert_array_equal(namespace['theta'],[[0,1],[1,0]])
    np.testing.assert_array_equal(namespace['W'],[[2,3],[4,5]])
    with pytest.raises(ValueError):module.adapt_shared_prior(source+source)
    with pytest.raises(ValueError):module.adapt_shared_prior('pass')

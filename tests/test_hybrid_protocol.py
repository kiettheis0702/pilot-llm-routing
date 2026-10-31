import math

import pytest

from tools.openai_study_hybrid import (
    deterministic_labels,
    make_prompt_disjoint_stream,
    partition_classifier_training,
    select_validation_checkpoint,
    threshold_actions,
)


def test_hybrid_deterministic_loss_labels_keep_ties_with_cheap_model():
    assert deterministic_labels([1, 0, .5, 0], [0, 1, .5, 0]) == [0, 1, 0, 0]
    with pytest.raises(ValueError):
        deterministic_labels([float('nan')], [1])


def test_drift_removes_shared_prompt_groups_without_reordering_task_cohorts():
    ids = ['m0', 'm1', 'm2', 'g0', 'm3', 'g1', 'g2']
    tasks = {q: ('grade-school-math' if q.startswith('g') else 'mmlu-test') for q in ids}
    groups = dict(zip(ids, ['a', 'a', 'b', 'c', 'd', 'e', 'f']))
    stream = make_prompt_disjoint_stream(ids, tasks, groups, warm_count=1, before_count=2, after_count=2)
    assert stream == {'warm': ['m0'], 'before': ['m2', 'm3'], 'after': ['g0', 'g1']}
    with pytest.raises(ValueError, match='enough'):
        make_prompt_disjoint_stream(ids, tasks, groups, warm_count=2, before_count=2, after_count=2)


def test_supervised_partition_rejects_group_leakage_even_when_ids_differ():
    groups = {'a': 'same', 'b': 'same', 'c': 'other'}
    with pytest.raises(ValueError, match='prompt'):
        partition_classifier_training(['a'], ['b'], ['c'], groups)
    assert partition_classifier_training(['a'], ['c'], [], groups)['train'] == ['a']


def test_checkpoint_selection_reads_only_finite_validation_losses_and_ties_earlier():
    assert select_validation_checkpoint([.7, .4, .4, .5]) == 2
    with pytest.raises(ValueError):
        select_validation_checkpoint([.4, math.inf])


def test_threshold_front_uses_probability_of_cheap_matching_and_includes_boundary():
    assert threshold_actions([0, .5, 1], .5) == [1, 0, 0]
    assert threshold_actions([0, 1], 0) == [0, 0]
    with pytest.raises(ValueError):
        threshold_actions([1.1], .5)

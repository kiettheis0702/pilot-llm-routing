"""Prospective HybridLLM/data-shift contracts; no provider API client."""
from __future__ import annotations

import math


def deterministic_labels(cheap_rewards, strong_rewards):
    cheap, strong = list(cheap_rewards), list(strong_rewards)
    if (not cheap or len(cheap) != len(strong)
            or any(not math.isfinite(x) or not 0 <= x <= 1 for x in cheap + strong)):
        raise ValueError('Finite aligned rewards in [0,1] required')
    # Microsoft NClassReranker DET_2CLS at match_t=0: ties prefer cheap.
    return [0 if a >= b else 1 for a, b in zip(cheap, strong)]


def make_prompt_disjoint_stream(ordered_ids, tasks, groups, *, warm_count, before_count, after_count):
    ids = list(ordered_ids)
    if (len(set(ids)) != len(ids) or any(q not in tasks or q not in groups for q in ids)
            or any(type(n) is not int or n < 1 for n in (warm_count, before_count, after_count))):
        raise ValueError('Invalid stream identities/counts')
    seen = set()
    mmlu, gsm = [], []
    for q in ids:
        if not (tasks[q].startswith('mmlu-') or tasks[q] == 'grade-school-math'):
            continue
        if groups[q] in seen:
            continue
        seen.add(groups[q])
        (mmlu if tasks[q].startswith('mmlu-') else gsm).append(q)
    if len(mmlu) < warm_count + before_count or len(gsm) < after_count:
        raise ValueError('Not enough distinct prompt groups for declared drift stream')
    return dict(warm=mmlu[:warm_count], before=mmlu[warm_count:warm_count + before_count], after=gsm[:after_count])


def partition_classifier_training(train, validation, test, groups):
    partitions = dict(train=list(train), validation=list(validation), test=list(test))
    seen_ids, seen_groups = set(), set()
    for ids in partitions.values():
        if len(set(ids)) != len(ids) or any(q not in groups for q in ids) or set(ids) & seen_ids:
            raise ValueError('Missing/overlapping classifier query identities')
        selected = {groups[q] for q in ids}
        if selected & seen_groups:
            raise ValueError('Classifier partition prompt group leakage')
        seen_ids.update(ids)
        seen_groups.update(selected)
    return partitions


def select_validation_checkpoint(losses):
    values = list(losses)
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Finite nonnegative validation losses required')
    return min(range(len(values)), key=lambda i: (values[i], i)) + 1


def threshold_actions(probabilities, threshold):
    values = list(probabilities)
    if (not math.isfinite(threshold) or not 0 <= threshold <= 1
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values)):
        raise ValueError('Valid probabilities and threshold required')
    return [0 if probability >= threshold else 1 for probability in values]

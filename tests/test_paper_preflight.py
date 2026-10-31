"""Independent fixtures for the free paper-preparation tools."""
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from scripts.cost_policy import ActualCostLedger, BinnedCostPolicy
from scripts.data_gate import query_hash


@pytest.mark.skipif(os.name != 'nt', reason='Live Windows process inventory')
@pytest.mark.parametrize('workload', ['tools.paper_preflight.compare',
    'tools.paper_preflight.audit', 'tools.paper_preflight.embedding_plan',
    'tools.paper_preflight.import_embeddings', 'tools.paper_preflight.train',
    'tools.openai_study', 'tools.openai_study_report', 'tools/research/verify_runs.py'])
def test_resource_guard_detects_another_preparation_process(workload):
    from scripts.resources import resource_snapshot, require_resources
    # This tiny owned probe waits for EOF; it never starts a real heavy job.
    child = subprocess.Popen([sys.executable, '-c',
        "import sys; print('ready', flush=True); sys.stdin.read()", workload],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True)
    try:
        assert child.stdout.readline().strip() == 'ready'
        snapshot = resource_snapshot()
        assert any(row['pid'] == child.pid or row['parent_pid'] == child.pid
                   for row in snapshot['heavy_jobs'])
        with pytest.raises(RuntimeError, match='Another heavy job'):
            require_resources(snapshot=dict(snapshot, available_gib=16))
    finally:
        child.communicate('', timeout=10)
        assert child.returncode == 0


def controller(**kwargs):
    return BinnedCostPolicy(arm_catalog=('a', 'b'), budget=1, query_count=2,
        bin_size=1, lower_bound=1, upper_bound=1, **kwargs)


def test_literal_threshold_can_overspend_and_actual_accounting_stays_separate():
    guarded = controller()
    literal = controller(affordability_mode='paper_literal')
    scores, costs = {'a': .9, 'b': .2}, {'a': .8, 'b': .1}
    assert guarded.select(scores, costs).arm_id == 'b'
    decision = literal.select(scores, costs)
    assert decision.arm_id == 'a'
    assert decision.available_after == pytest.approx(-.3)
    ledger = ActualCostLedger(literal)
    ledger.record(decision, .85)
    assert ledger.report()['actual_total'] == .85
    assert ledger.report()['estimated_total'] == .8
    assert literal.report()['affordability_mode'] == 'paper_literal'
    # The next bin allocates another .5, with the prior deficit carried forward.
    next_decision = literal.select(scores, costs)
    assert next_decision.available_before == pytest.approx(.2)
    assert next_decision.arm_id == 'a'
    assert literal.report()['estimated_budget_violation'] is True


@pytest.mark.parametrize('mode', ['unknown', None, True])
def test_invalid_budget_interpretation_is_rejected(mode):
    with pytest.raises(ValueError, match='affordability'):
        controller(affordability_mode=mode)


def test_literal_fallback_keeps_the_average_remaining_budget_rule():
    policy = controller(affordability_mode='paper_literal')
    decision = policy.select({'a': -.1, 'b': -.2}, {'a': .8, 'b': .1})
    assert decision.arm_id == 'b'
    assert decision.branch == 'fallback'


def test_relative_metrics_keep_all_requested_queries_after_early_termination():
    from tools.paper_preflight.compare import relative_metrics
    result = relative_metrics(total_reward=1, actual_cost=2, served=1,
        requested=4, gpt4_total_reward=2, gpt4_total_cost=8)
    assert result['quality_fraction_gpt4'] == .5
    assert result['cost_fraction_gpt4'] == .25
    assert result['service_utility'] == .25
    assert result['coverage'] == .25
    assert relative_metrics(total_reward=0, actual_cost=0, served=0,
        requested=4, gpt4_total_reward=0, gpt4_total_cost=0)['quality_fraction_gpt4'] is None


def test_budget_stream_terminates_without_shrinking_reference_or_reading_feedback_for_selection():
    from tools.paper_preflight.compare import simulate_budget_stream
    ids = ['q1', 'q2']
    predictions = {q: {'a': .9, 'b': .2} for q in ids}
    estimates = {'q1': {'a': .3, 'b': .1}, 'q2': {'a': .5, 'b': .5}}
    outcomes = {('q1', 'a'): (1., 1.), ('q1', 'b'): (.5, .1),
                ('q2', 'a'): (.5, 1.), ('q2', 'b'): (.25, .1)}
    args = dict(query_ids=ids, arms=['a', 'b'], predictions=predictions,
        estimates=estimates, outcomes=outcomes, reference_arm='a', budget=.4,
        bin_size=2, lower_bound=10, upper_bound=10, affordability_mode='guarded')
    result, trace = simulate_budget_stream(**args)
    assert result['served'] == 1
    assert result['status'] == 'insufficient_budget'
    assert result['relative']['quality_fraction_gpt4'] == pytest.approx(1/3)
    assert result['relative']['cost_fraction_gpt4'] == .05
    assert [r['arm_id'] for r in trace] == ['b']
    outcomes[('q1', 'a')] = (0., 10.)
    changed, changed_trace = simulate_budget_stream(**args)
    assert [r['arm_id'] for r in changed_trace] == ['b']
    assert changed['relative']['gpt4_total_reward'] == .5


def test_budget_stream_rejects_missing_reference_and_duplicate_stream_ids():
    from tools.paper_preflight.compare import simulate_budget_stream
    args = dict(query_ids=['q'], arms=['a', 'b'], predictions={'q': {'a': 1, 'b': 0}},
        estimates={'q': {'a': .1, 'b': .1}}, outcomes={('q', 'a'): (1., .1)},
        reference_arm='a', budget=1, bin_size=1, lower_bound=1, upper_bound=1,
        affordability_mode='guarded')
    with pytest.raises(ValueError):
        simulate_budget_stream(**args)
    args['query_ids'] = ['q', 'q']
    with pytest.raises(ValueError):
        simulate_budget_stream(**args)


@pytest.mark.parametrize('overrides', [dict(actual_cost=-1), dict(served=5),
    dict(total_reward=float('nan')), dict(requested=0), dict(gpt4_total_cost=-1)])
def test_relative_metrics_reject_invalid_accounting(overrides):
    from tools.paper_preflight.compare import relative_metrics
    args = dict(total_reward=1, actual_cost=2, served=1, requested=4,
                gpt4_total_reward=2, gpt4_total_cost=8)
    args.update(overrides)
    with pytest.raises(ValueError):
        relative_metrics(**args)


def preference_fixture():
    inputs, comparisons = [], []
    # q0 and q1 are separate battles about the same normalized prompt.
    for i, text in enumerate(['repeat', ' repeat ', 'two', 'three', 'four', 'five', 'six', 'seven']):
        inputs.append(dict(query_id=f'q{i}', text=text, prompt_hash=query_hash(text)))
        comparisons.append(dict(battle_id=str(i), query_id=f'q{i}',
            winner='a', loser='b', split='train'))
    return comparisons, inputs


def test_three_way_split_keeps_prompt_groups_and_exact_test_size():
    from tools.paper_preflight.audit import split_preferences_three_way
    comparisons, inputs = preference_fixture()
    rows = split_preferences_three_way(comparisons, inputs, test_pairs=3, seed=42)
    assert len([r for r in rows if r['split'] == 'test']) == 3
    by_id = {r['query_id']: r['split'] for r in rows}
    assert by_id['q0'] == by_id['q1']
    assert {r['split'] for r in rows} == {'train', 'calibration', 'test'}
    assert rows == split_preferences_three_way(comparisons[::-1], inputs[::-1], test_pairs=3, seed=42)
    assert all(r['split'] == 'train' for r in comparisons)  # Source never mutated.


def test_three_way_split_rejects_impossible_group_size_and_wrong_prompt_hash():
    from tools.paper_preflight.audit import split_preferences_three_way
    comparisons, inputs = preference_fixture()
    with pytest.raises(ValueError):
        split_preferences_three_way(comparisons[:2], inputs[:2], test_pairs=1)
    inputs[0]['prompt_hash'] = 'incorrect'
    with pytest.raises(ValueError, match='hash'):
        split_preferences_three_way(comparisons, inputs, test_pairs=3)


def test_stream_audit_detects_cross_split_prompt_overlap_and_changed_order():
    from tools.paper_preflight.audit import summarize_stream
    queries = pd.DataFrame([dict(query_id=q, query=text, query_hash=query_hash(text), task=task)
        for q, text, task in [('q1', 'same', 'mmlu-one'), ('q2', 'same', 'mmlu-two'),
                              ('q3', 'other', 'hellaswag'), ('q4', 'fourth', 'hellaswag')]])
    splits = pd.DataFrame([dict(query_id=q, query_hash=h, task=t, split=s, stream_position=i)
        for i, (q, h, t, s) in enumerate(zip(queries.query_id, queries.query_hash, queries.task,
            ['learning', 'deployment', 'tuning', 'learning']))])
    report = summarize_stream(queries, splits)
    assert report['raw_task_count'] == 3
    assert report['cross_split_prompt_groups'] == 1
    changed = splits.copy()
    changed.loc[[0, 3], 'stream_position'] = [3, 0]
    other = summarize_stream(queries, changed)
    assert report['streams']['learning']['order_sha256'] != other['streams']['learning']['order_sha256']
    assert report['streams']['deployment'] == other['streams']['deployment']


def write_inputs(path, rows):
    path.write_text(json.dumps(rows), encoding='utf-8')
    return path


def test_embedding_plan_deduplicates_text_and_batches_without_sending(tmp_path):
    from tools.paper_preflight.embedding_plan import build_plan
    source = write_inputs(tmp_path/'input.json', [
        dict(query_id='q1', text='hello', prompt_hash=query_hash('hello')),
        dict(query_id='q2', text='hello', prompt_hash=query_hash('hello')),
        dict(query_id='q3', text='world', prompt_hash=query_hash('world'))])
    output = tmp_path/'plan'
    result = build_plan([source], output, batch_size=1)
    assert result['source_rows'] == 3
    assert result['unique_texts'] == 2
    assert result['tokens'] == 2
    assert result['estimated_standard_usd'] == pytest.approx(.00000004)
    assert result['requests'] == 2
    assert result['paid_api_calls'] == 0
    assert result['approved'] is False
    requests = [json.loads(line) for line in (output/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['body']['input'] for r in requests] == [['hello'], ['world']]
    assert all(r['body']['model'] == 'text-embedding-3-small' for r in requests)
    assert all('dimensions' not in r['body'] for r in requests)
    mapping = json.loads((output/'id-map.json').read_text(encoding='utf-8'))
    assert mapping['q1'] == mapping['q2']
    assert mapping['q1'] != mapping['q3']
    with pytest.raises(FileExistsError):
        build_plan([source], output)


@pytest.mark.parametrize('bad', [dict(batch_size=0), dict(batch_size=True),
    dict(max_batch_tokens=0), dict(price_per_million=-1), dict(price_per_million=float('nan'))])
def test_embedding_plan_rejects_invalid_configuration_before_output(tmp_path, bad):
    from tools.paper_preflight.embedding_plan import build_plan
    source = write_inputs(tmp_path/'input.json', [dict(query_id='q', text='hello')])
    output = tmp_path/'plan'
    with pytest.raises(ValueError):
        build_plan([source], output, **bad)
    assert not output.exists()


def test_embedding_plan_rejects_duplicate_id_and_oversized_input(tmp_path):
    from tools.paper_preflight.embedding_plan import build_plan
    source = write_inputs(tmp_path/'input.json', [dict(query_id='q', text='a'), dict(query_id='q', text='b')])
    with pytest.raises(ValueError, match='identity'):
        build_plan([source], tmp_path/'duplicate')
    write_inputs(source, [dict(query_id='q', text=' x' * 8200)])
    with pytest.raises(ValueError, match='token'):
        build_plan([source], tmp_path/'oversized')
    assert not (tmp_path/'oversized').exists()


def test_embedding_plan_honors_total_request_tokens_without_truncating(tmp_path):
    from tools.paper_preflight.embedding_plan import build_plan
    source = write_inputs(tmp_path/'input.json', [dict(query_id='a', text='hello world'),
                                                 dict(query_id='b', text='hello there')])
    output = tmp_path/'plan'
    manifest = build_plan([source], output, max_batch_tokens=2)
    assert manifest['requests'] == 2
    requests = [json.loads(line) for line in (output/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['body']['input'] for r in requests] == [['hello world'], ['hello there']]

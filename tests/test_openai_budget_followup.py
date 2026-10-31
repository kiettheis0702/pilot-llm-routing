"""Exhaustion/allocation variants must count skipped queries and preserve budget."""
import pytest


def stream(cost=.3,n=4,budget=1.,bin_size=2):
    ids=[f'q{i}' for i in range(n)];arms=['gpt4']
    return dict(query_ids=ids,arms=arms,predictions={q:{'gpt4':1.} for q in ids},
        estimates={q:{'gpt4':cost} for q in ids},outcomes={(q,'gpt4'):(1.,cost) for q in ids},
        reference_arm='gpt4',budget=budget,bin_size=bin_size,lower_bound=.1,upper_bound=10.)


def test_guarded_stop_matches_existing_controller_and_keeps_unserved_denominator():
    from tools.openai_study_budget import simulate_allocation
    from tools.paper_preflight.compare import simulate_budget_stream
    values=stream()
    reference,_=simulate_budget_stream(**values,affordability_mode='guarded')
    result,trace=simulate_allocation(**values,policy='guarded-stop')
    assert result['served']==reference['served']==1
    assert result['relative']['quality_fraction_gpt4']==.25
    assert result['requested']==4 and len(trace)==1


def test_skipping_a_query_advances_time_to_later_bins_without_borrowing():
    from tools.openai_study_budget import simulate_allocation
    result,trace=simulate_allocation(**stream(),policy='guarded-skip')
    assert result['served']==3 and result['requested']==4
    assert result['estimated_cost']==pytest.approx(.9)
    assert result['relative']['quality_fraction_gpt4']==.75
    assert [r['query_id'] for r in trace if r['arm_id'] is None]==['q1']
    assert [r['query_id'] for r in trace if r['arm_id'] is not None]==['q0','q2','q3']
    result,trace=simulate_allocation(**stream(cost=.8,n=2,bin_size=1),policy='guarded-skip')
    assert trace[0]['arm_id'] is None and trace[1]['arm_id']=='gpt4'
    assert result['estimated_cost']<=result['budget']


def test_fixed_per_query_does_not_spend_another_queries_share():
    from tools.openai_study_budget import simulate_allocation
    result,trace=simulate_allocation(**stream(),policy='fixed-per-query')
    assert result['served']==0 and result['estimated_cost']==0.
    assert len(trace)==4 and result['relative']['coverage']==0.


def test_literal_variant_reports_overallocation_instead_of_clamping_it():
    from tools.openai_study_budget import simulate_allocation
    result,_=simulate_allocation(**stream(cost=.8,n=2,bin_size=1),policy='paper-literal-stop')
    assert result['served']==2 and result['estimated_cost']==pytest.approx(1.6)
    assert result['estimated_overspend']==pytest.approx(.6)


def test_retrospective_lagrange_uses_predictions_costs_and_can_abstain():
    from tools.openai_study_budget import simulate_allocation
    values=stream();values['predictions']={q:{'gpt4':float(4-i)} for i,q in enumerate(values['query_ids'])}
    first,trace=simulate_allocation(**values,policy='retrospective-lagrange')
    assert first['served']==3 and first['estimated_cost']<=1.
    assert [r['query_id'] for r in trace if r['arm_id'] is not None]==['q0','q1','q2']
    values['outcomes']={key:(0.,.7) for key in values['outcomes']}
    second,other=simulate_allocation(**values,policy='retrospective-lagrange')
    assert [r['arm_id'] for r in trace]==[r['arm_id'] for r in other]
    assert second['total_reward']==0. and second['actual_cost']>second['budget']


def test_invalid_policy_is_rejected():
    from tools.openai_study_budget import simulate_allocation
    with pytest.raises(ValueError):simulate_allocation(**stream(),policy='unknown')


def test_budget_inventory_has_all_policies_seeds_domains_and_frozen_budgets():
    import json
    from pathlib import Path
    from tools.openai_study_budget import condition_inventory, POLICIES
    parent=json.loads(Path('configs/frozen_experiment_protocol.json').read_text(encoding='utf-8'))
    rows=condition_inventory(parent)
    assert len(rows)==240 and len({r['id'] for r in rows})==240
    assert {r['policy'] for r in rows}==set(POLICIES)
    assert len({(r['domain'],r['seed']) for r in rows})==6
    assert all(r['budget_fraction'] in parent['primary_plan']['budget_fractions'] for r in rows)

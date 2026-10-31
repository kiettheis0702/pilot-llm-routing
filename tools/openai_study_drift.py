"""Prompt-disjoint MMLU-to-GSM8K selected-feedback stream; no paid service."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/openai-followup-20261031/drift'


def learn_shift(model, stream, context_for_query, selected_outcome, oracle_for_query):
    rows = []
    for phase in ('warm', 'before', 'after'):
        for position, query in enumerate(stream[phase], 1):
            context = context_for_query(query)
            scores = model.scores(context)
            arm = model.select(context)
            scale = np.abs(context).max()
            normalized = (context / scale) / np.linalg.norm(context / scale)
            index = model.arms.index(arm)
            uncertainty = math.sqrt(max(float(normalized @ np.linalg.solve(model.A[index], normalized)), 0.))
            reward, cost = selected_outcome(query, arm)
            row = dict(round=len(rows) + 1, phase=phase, phase_position=position,
                query_id=query, arm_id=arm, reward=reward, actual_cost=cost,
                rounds_before_feedback=model.rounds, scores=scores,
                selected_uncertainty=uncertainty, oracle_reward=oracle_for_query(query),
                empirical_oracle_regret=oracle_for_query(query) - reward)
            # The only feedback call in the online path contains the selected arm.
            model.observe(arm, context, reward)
            rows.append(row)
    return rows


def summarize_windows(rows, *, size=1000):
    if type(size) is not int or size < 1:
        raise ValueError('Positive drift window size required')
    before = [r for r in rows if r['phase'] == 'before']
    after = [r for r in rows if r['phase'] == 'after']
    if not before or not after:
        raise ValueError('Both sides of the task shift are required')
    groups = {'all-evaluation':before + after, 'all-before':before, 'all-after':after,
              'last-before':before[-size:], 'first-after':after[:size], 'last-after':after[-size:]}
    result = {}
    for name, group in groups.items():
        reward = math.fsum(r['reward'] for r in group)
        cost = math.fsum(r['actual_cost'] for r in group)
        reference_reward = math.fsum(r['gpt4_reward'] for r in group)
        reference_cost = math.fsum(r['gpt4_cost'] for r in group)
        if reference_reward <= 0 or reference_cost <= 0:
            raise ValueError('Positive same-window GPT4 references required')
        result[name] = dict(requested=len(group), served=len(group), coverage=1.,
            total_reward=reward, service_utility=reward / len(group), actual_recorded_cost=cost,
            gpt4_total_reward=reference_reward, gpt4_total_cost=reference_cost,
            quality_fraction_gpt4=reward / reference_reward, cost_fraction_gpt4=cost / reference_cost)
    return result


def declare():
    from scripts.experiments import verify_pins
    from tools.openai_study import DEFAULT_BASE as CORE, checked_pin, once, sealed
    from tools.paper_preflight.common import digest, read_json
    hybrid = ROOT / 'outputs/openai-followup-20261031/hybrid/drift/protocol.json'
    parent = sealed(CORE / 'protocol.json'); verify_pins(parent, ROOT)
    prospective = read_json(hybrid)
    protocol = dict(version=1, kind='openai-verified-binary-online-task-shift',
        parent=checked_pin(CORE / 'protocol.json'), hybrid_cohort=checked_pin(hybrid),
        stream=prospective['stream'], seed=42,
        arms=['gpt-4-1106-preview', 'meta/llama-2-70b-chat'],
        alpha=parent['tuning']['mmlu']['selected_parameters']['pilot']['alpha'],
        feedback='selected historical arm only, original shared8-arm human-preference prior',
        budget_controller=False, coverage='all evaluation queries served', window=1000,
        sources=[checked_pin(Path(__file__)), checked_pin(ROOT / 'tools/openai_study_hybrid.py'),
                 checked_pin(ROOT / 'tools/openai_study_drift_verify.py'),
                 checked_pin(ROOT / 'tools/openai_study_binary_run.py')],
        differences='local verified Llama2 pair, label-free unique prompt cohorts; original author stream/windows unknown',
        paper_equivalent=False, synthetic=False, paid_api_calls=0)
    protocol['manifest_sha256'] = digest(protocol); once(BASE / 'protocol.json', protocol)


def run():
    from scripts.deployment_adapter import FrozenDeploymentRouter
    from scripts.experiments import verify_pins
    from scripts.resources import require_resources
    from tools.openai_study import DEFAULT_BASE as CORE, load_data, once, sealed
    from tools.openai_study_binary_run import binary_policy
    from tools.paper_preflight.common import digest, sha256
    require_resources(); protocol = sealed(BASE / 'protocol.json'); verify_pins(protocol, ROOT)
    if (BASE / 'manifest.json').exists():
        raise ValueError('Completed drift stream exists; verify rather than retrain')
    bundle = load_data(CORE, 'routerbench', protocol['seed'])
    model = binary_policy(bundle['prior'], 'pilot', {'alpha':protocol['alpha']}); model.reset(protocol['seed'])
    truth = {(r.query_id, r.arm_id):(float(r.reward), float(r.actual_cost))
             for r in bundle['outcomes'].itertuples(index=False) if r.arm_id in protocol['arms']}
    selected = [q for phase in ('warm','before','after') for q in protocol['stream'][phase]]
    oracle = {q:max(truth[q,a][0] for a in protocol['arms']) for q in selected}
    rows = learn_shift(model, protocol['stream'], bundle['contexts'].get, lambda q,a:truth[q,a], oracle.__getitem__)
    for row in rows:
        row['gpt4_reward'], row['gpt4_cost'] = truth[row['query_id'], 'gpt-4-1106-preview']
    with (BASE / 'trace.jsonl').open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
    FrozenDeploymentRouter(model, seed=protocol['seed']).save_checkpoint(BASE / 'checkpoint.json')
    once(BASE / 'windows.json', summarize_windows(rows, size=protocol['window']))
    artifact = dict(version=1, kind='selected-feedback-pilot-task-shift', protocol_sha256=sha256(BASE / 'protocol.json'),
        lineage=bundle['lineage'], rounds=len(rows), evaluation_queries=10000, warm_queries=5000,
        files={name:sha256(BASE / name) for name in ('trace.jsonl','checkpoint.json','windows.json')},
        synthetic=False, paper_equivalent=False)
    artifact['manifest_sha256'] = digest(artifact); once(BASE / 'manifest.json', artifact)
    print(json.dumps(dict(complete=True, rounds=len(rows), evaluation_queries=10000)), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['declare','run'])
    globals()[parser.parse_args().action]()


if __name__ == '__main__':
    main()

"""Shared GPT-4 denominators and paired historical budget-policy comparisons."""
from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

from tools.paper_preflight.common import ROOT, digest, read_json, sha256, write_json


def relative_metrics(*, total_reward, actual_cost, served, requested,
                     gpt4_total_reward, gpt4_total_cost):
    if (type(requested) is not int or requested < 1 or type(served) is not int
            or not 0 <= served <= requested):
        raise ValueError('Invalid requested/served counts')
    for value in (total_reward, actual_cost, gpt4_total_reward, gpt4_total_cost):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Metrics must be finite and nonnegative')
    if total_reward > served or gpt4_total_reward > requested:
        raise ValueError('Reward exceeds the benchmark [0,1] support')
    return dict(quality_fraction_gpt4=total_reward/gpt4_total_reward if gpt4_total_reward else None,
        cost_fraction_gpt4=actual_cost/gpt4_total_cost if gpt4_total_cost else None,
        service_utility=total_reward/requested, coverage=served/requested,
        gpt4_total_reward=gpt4_total_reward, gpt4_total_cost=gpt4_total_cost,
        denominator='GPT-4 historical outcomes on ALL requested queries, including unserved queries')


def simulate_budget_stream(*, query_ids, arms, predictions, estimates, outcomes,
                           reference_arm, budget, bin_size, lower_bound, upper_bound,
                           affordability_mode):
    """Select using predictions/estimated costs only, then reveal chosen outcomes.

    The complete benchmark grid is evaluator data, used separately for reference
    totals. No oracle score or realized cost enters the selection function.
    """
    from scripts.cost_policy import ActualCostLedger, BinnedCostPolicy
    ids, catalog = tuple(query_ids), tuple(arms)
    if (not ids or len(set(ids)) != len(ids) or reference_arm not in catalog
            or set(predictions) != set(ids) or set(estimates) != set(ids)):
        raise ValueError('Invalid comparison stream identities')
    if set(outcomes) != {(q, a) for q in ids for a in catalog}:
        raise ValueError('Historical outcomes must cover the complete model/query grid')
    for value in outcomes.values():
        if (not isinstance(value, (tuple, list)) or len(value) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) or v < 0 for v in value)
                or value[0] > 1):
            raise ValueError('Invalid historical reward or cost')
    policy = BinnedCostPolicy(arm_catalog=catalog, budget=budget, query_count=len(ids),
        bin_size=bin_size, lower_bound=lower_bound, upper_bound=upper_bound,
        affordability_mode=affordability_mode)
    ledger, trace, total_reward = ActualCostLedger(policy), [], 0.
    terminal = None
    for q in ids:
        decision = policy.select(predictions[q], estimates[q])
        if decision.status != 'selected':
            terminal = dict(query_id=q, decision=asdict(decision))
            break
        reward, cost = outcomes[q, decision.arm_id]
        ledger.record(decision, cost)
        total_reward += reward
        trace.append(dict(query_id=q, arm_id=decision.arm_id, reward=reward,
                          actual_cost=cost, decision=asdict(decision)))
    accounting = ledger.report()
    reference_reward = sum(outcomes[q, reference_arm][0] for q in ids)
    reference_cost = sum(outcomes[q, reference_arm][1] for q in ids)
    return dict(status=accounting['status'], served=len(trace), requested=len(ids),
        total_reward=total_reward, actual_cost=accounting['actual_total'], budget=budget,
        affordability_mode=affordability_mode, terminal=terminal, accounting=accounting,
        relative=relative_metrics(total_reward=total_reward, actual_cost=accounting['actual_total'],
            served=len(trace), requested=len(ids), gpt4_total_reward=reference_reward,
            gpt4_total_cost=reference_cost)), trace


def execute(output_dir, *, root=ROOT):
    import argparse
    import gc
    import pandas as pd
    from scripts.deployment_adapter import FrozenDeploymentRouter
    from scripts.experiments import load_artifact, load_real_bundle
    from scripts.resources import require_resources
    from tools.course.analyze import verify_package

    root, output = Path(root), Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    preflight = require_resources()
    package = root/'reports/full-experiments-2026-10-29'
    verify_package(package)
    aggregate = pd.read_csv(package/'aggregate/per-seed.csv')
    outcomes = pd.read_parquet(root/'data/processed/routerbench_0shot/outcomes.parquet')
    splits = pd.read_csv(root/'data/processed/routerbench_0shot/splits.csv')
    reference_arm = 'gpt-4-1106-preview'
    reference = {}
    for domain in ('routerbench', 'mmlu'):
        frame = splits.loc[(splits.split == 'deployment') &
            (True if domain == 'routerbench' else splits.task.str.startswith('mmlu-'))].sort_values('stream_position')
        ids = frame.query_id.tolist()
        values = outcomes.loc[(outcomes.arm_id == reference_arm) & outcomes.query_id.isin(ids)].set_index('query_id').loc[ids]
        reference[domain] = dict(query_count=len(ids), order_sha256=digest(ids),
            total_reward=float(values.reward.sum()), total_cost=float(values.actual_cost.sum()))
    normalized = []
    for row in aggregate.to_dict('records'):
        ref = reference[row['domain']]
        if row['requested'] != ref['query_count']:
            raise ValueError('Historical aggregate evaluation population mismatch')
        result = relative_metrics(total_reward=row['total_reward'], actual_cost=row['actual_recorded_cost'],
            served=int(row['served']), requested=int(row['requested']),
            gpt4_total_reward=ref['total_reward'], gpt4_total_cost=ref['total_cost'])
        normalized.append(dict(row, **result, allocated_budget_fraction_gpt4=row['budget']/ref['total_cost']))
    output.mkdir(parents=True)
    pd.DataFrame(normalized).to_csv(output/'historical-gpt4-relative.csv', index=False, lineterminator='\n')
    del outcomes
    rows, pairs, inventories = [], [], []
    for domain in ('routerbench', 'mmlu'):
        for seed in (7, 42, 2026):
            require_resources()
            bundle = load_real_bundle(root/'configs/frozen_experiment_protocol.json',
                root/'configs/research_prior_index.json', domain=domain, seed=seed, root=root)
            protocol = bundle['protocol']
            learning_dir = root/f'outputs/research-experiments-20261029/learning/{domain}/seed-{seed}/main/pilot'
            learned = load_artifact(learning_dir)
            snapshot = learned['snapshots']['1']
            checkpoint = learning_dir/snapshot['file']
            router = FrozenDeploymentRouter.load_checkpoint(checkpoint)
            if (router.arms != tuple(protocol['arms'])
                    or router.source_state()['rounds'] != len(bundle['selections']['learning'].query_ids)
                    or router.metadata()['source_state_sha256'] != snapshot['source_state_sha256']):
                raise ValueError('Frozen router identity or learning count mismatch')
            ids = bundle['selections']['deployment'].query_ids
            predictions = {q: router.predict_rewards(bundle['contexts'].get(q)) for q in ids}
            estimates = {q: bundle['estimator'].estimate(bundle['input_counts'][q]) for q in ids}
            selected = bundle['outcomes'].loc[bundle['outcomes'].query_id.isin(ids)]
            grid = {(r.query_id, r.arm_id): (float(r.reward), float(r.actual_cost))
                    for r in selected.itertuples(index=False)}
            group = output/domain/f'seed-{seed}'
            group.mkdir(parents=True)
            evidence = dict(query_ids=list(ids), arms=list(router.arms), predictions=predictions,
                estimates=estimates, outcomes=selected.to_dict('records'), reference_arm=reference_arm,
                bounds=protocol['tuning'][domain]['bounds'], bin_size=protocol['primary_plan']['bin_size'],
                checkpoint_sha256=sha256(checkpoint), learning_manifest_sha256=sha256(learning_dir/'manifest.json'),
                source_state_sha256=router.metadata()['source_state_sha256'], lineage=bundle['lineage'])
            write_json(group/'inputs.json', evidence)
            tuning = protocol['tuning'][domain]
            for fraction in protocol['primary_plan']['budget_fractions']:
                results, traces = {}, {}
                for mode in ('guarded', 'paper_literal'):
                    result, trace = simulate_budget_stream(query_ids=ids, arms=router.arms,
                        predictions=predictions, estimates=estimates, outcomes=grid,
                        reference_arm=reference_arm, budget=tuning['absolute_reference_budget']*fraction,
                        bin_size=evidence['bin_size'], **tuning['bounds'], affordability_mode=mode)
                    results[mode], traces[mode] = result, trace
                    key = f'budget-{fraction:g}-{mode}'
                    with (group/(key+'.jsonl')).open('w', encoding='utf-8', newline='\n') as handle:
                        for item in trace:
                            import json
                            handle.write(json.dumps(item, allow_nan=False) + '\n')
                    write_json(group/(key+'.json'), result)
                    rows.append(dict(domain=domain, seed=seed, budget_fraction=fraction, mode=mode,
                        requested=len(ids), served=result['served'], total_reward=result['total_reward'],
                        budget=result['budget'], actual_cost=result['actual_cost'],
                        estimated_cost=result['accounting']['estimated_total'],
                        actual_overspend=result['accounting']['actual_budget_excess_observed'],
                        estimated_overspend=max(0., result['accounting']['estimated_total']-result['budget']),
                        **result['relative']))
                old = aggregate.loc[(aggregate.domain == domain) & (aggregate.seed == seed) &
                    (aggregate.method == 'pilot') & (aggregate.variant == 'main') &
                    (aggregate.learning_fraction == 1) & (aggregate.budget_fraction == fraction)].iloc[0]
                guarded = results['guarded']
                if (guarded['served'] != int(old.served)
                        or not math.isclose(guarded['total_reward'], old.total_reward, abs_tol=1e-9)
                        or not math.isclose(guarded['actual_cost'], old.actual_recorded_cost, abs_tol=1e-9)):
                    raise ValueError('Default controller failed historical replay')
                guarded_actions = {r['query_id']: r['arm_id'] for r in traces['guarded']}
                literal_actions = {r['query_id']: r['arm_id'] for r in traces['paper_literal']}
                common = set(guarded_actions) & set(literal_actions)
                pairs.append(dict(domain=domain, seed=seed, budget_fraction=fraction,
                    common_served=len(common), changed_common_actions=sum(guarded_actions[q] != literal_actions[q] for q in common),
                    literal_only_served=len(set(literal_actions)-set(guarded_actions)),
                    guarded_only_served=len(set(guarded_actions)-set(literal_actions)),
                    quality_delta=results['paper_literal']['relative']['quality_fraction_gpt4']-guarded['relative']['quality_fraction_gpt4'],
                    cost_delta=results['paper_literal']['relative']['cost_fraction_gpt4']-guarded['relative']['cost_fraction_gpt4']))
            if router.metadata()['source_state_sha256'] != evidence['source_state_sha256']:
                raise ValueError('Comparison changed frozen router')
            print(f'Completed {domain} seed {seed}: 16 offline conditions', flush=True)
            inventories.append(dict(domain=domain, seed=seed, inputs_sha256=sha256(group/'inputs.json')))
            del bundle, router, evidence, grid, predictions, estimates, selected
            gc.collect()
    pd.DataFrame(rows).to_csv(output/'budget-comparison.csv', index=False, lineterminator='\n')
    pd.DataFrame(pairs).to_csv(output/'paired-differences.csv', index=False, lineterminator='\n')
    manifest = dict(version=1, kind='historical-bge-budget-interpretation-comparison',
        conditions=len(rows), pairs=len(pairs), normalized_historical_conditions=len(normalized),
        reference=reference, group_inputs=inventories, preflight=preflight, paid_api_calls=0,
        embedding='existing BGE-M3, not OpenAI', model_count=8, paper_equivalent=False,
        only_changed_factor='threshold affordability intersection; all other budget formulas unchanged',
        caveat='GPT-4-relative denominators align units, not model pool/data/preprocessing with authors',
        code_sha256={str(p.relative_to(root)).replace('\\', '/'): sha256(p)
            for p in (root/'scripts/cost_policy.py', Path(__file__))},
        files={str(p.relative_to(output)).replace('\\', '/'): sha256(p)
               for p in sorted(output.rglob('*')) if p.is_file()})
    write_json(output/'manifest.json', manifest)
    return manifest


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    result = execute(args.output_dir)
    print({k: result[k] for k in ('conditions', 'pairs', 'normalized_historical_conditions', 'paid_api_calls')})


if __name__ == '__main__':
    main()

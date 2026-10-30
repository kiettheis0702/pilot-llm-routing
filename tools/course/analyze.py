"""Course-specific, provenance-preserving analysis of the frozen local results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.pilot import PILOT
from scripts.policies import LinearUCB

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    """Return the SHA-256 of a file without loading its content into memory."""
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    """Write a new strict JSON file, refusing to replace existing evidence."""
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def verify_package(path):
    """Check every manifest member and reject paths escaping the package root."""
    root = Path(path).resolve()
    files = json.loads((root/'package-files.json').read_text(encoding='utf-8'))['files']
    if not files:
        raise ValueError('Empty package inventory')
    for name, expected in files.items():
        member = (root/name).resolve()
        if not member.is_relative_to(root):
            raise ValueError('Package member escapes root')
        if not member.is_file() or sha256(member) != expected:
            raise ValueError('Package checksum mismatch: '+name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and p != root/'package-files.json'}
    if actual != set(files):
        raise ValueError('Package inventory does not cover exactly the available evidence')
    return dict(file_count=len(files), manifest_sha256=sha256(root/'package-files.json'))


def read_rows(package):
    """Read verified aggregate rows, preserving IDs and finite numeric values."""
    verify_package(package)
    with (Path(package)/'aggregate/per-seed.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    strings = {'id', 'domain', 'variant', 'method'}
    for row in rows:
        for key in row.keys()-strings:
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError('Nonfinite aggregate value')
    manifest = json.loads((Path(package)/'aggregate/manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('complete') or len(rows) != manifest['primary_runs']+manifest['ablation_runs']:
        raise ValueError('Incomplete aggregate inventory')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate condition ID')
    if set(manifest['run_manifest_sha256']) != {r['id'] for r in rows}:
        raise ValueError('Condition inventory differs from run provenance')
    return rows


def magnitude_stress(rounds, alpha, rewards):
    """Compare cosine and linear exploitation under identical priors and feedback.

    Args: positive integer rounds, nonnegative finite alpha, two increasing
    positive rewards <=1. Returns summaries and selected-only per-round traces.
    This deliberately violates the unit-direction reward assumption.
    """
    if type(rounds) is not int or rounds < 1:
        raise ValueError('rounds must be a positive integer')
    if isinstance(alpha, bool) or not math.isfinite(alpha) or alpha < 0:
        raise ValueError('alpha must be nonnegative and finite')
    if len(rewards) != 2 or not all(math.isfinite(r) for r in rewards) or not 0 < rewards[0] < rewards[1] <= 1:
        raise ValueError('Expected two strictly increasing positive rewards <=1')
    arms = ('low', 'high')
    priors = {a: [float(r), 0.] for a, r in zip(arms, rewards)}
    cosine = PILOT(arms, 2, priors=priors, lambdas={a: 1. for a in arms},
                   provenance='synthetic oracle prior for controlled magnitude stress', alpha=alpha)
    linear = LinearUCB(arms, 2, alpha=alpha, ridge=1.)
    # Explicit experimental control: identical A and b initialization to PILOT.
    # This is not the zero-prior LinUCB baseline used in the main experiment.
    linear.reset(42)
    linear._b = np.array([priors[a] for a in arms], dtype=float)
    cosine.reset(42)
    result = []
    for name, model in [('pilot-cosine', cosine), ('linear-prior-control', linear)]:
        trace = []
        total = regret = 0.
        counts = dict.fromkeys(arms, 0)
        for t in range(rounds):
            arm = model.select([1., 0.])
            reward = float(rewards[arms.index(arm)])
            model.observe(arm, [1., 0.], reward)
            counts[arm] += 1
            total += reward
            regret += rewards[1]-reward
            trace.append(dict(round=t+1, arm=arm, reward=reward, regret=regret))
        result.append(dict(method=name, low_count=counts['low'], high_count=counts['high'],
            total_reward=total, regret=regret, mean_reward=total/rounds,
            update_count=model.rounds, trace=trace))
    return dict(synthetic=True, rounds=rounds, alpha=alpha, seed=42,
                rewards=list(rewards), policies=result)


def cost_sensitivity(rows, factors):
    """Summarize cost shocks on fixed selected decisions, returning no fake reruns."""
    if not rows:
        raise ValueError('Empty cost population')
    for r in rows:
        if not all(math.isfinite(r[k]) and r[k] >= 0 for k in ('budget', 'actual_recorded_cost')):
            raise ValueError('Invalid cost or budget')
    result = []
    for factor in factors:
        if isinstance(factor, bool) or not math.isfinite(factor) or factor <= 0:
            raise ValueError('Invalid cost factor')
        excess = [max(0., factor*r['actual_recorded_cost']-r['budget']) for r in rows]
        result.append(dict(factor=factor, conditions=len(rows),
            over_budget_count=sum(v > 1e-9 for v in excess),
            maximum_excess=max(excess), total_excess=sum(excess),
            counterfactual=factor != 1., choices_unchanged=True))
    return result


def comparison(rows, protocol):
    """Build a descriptive comparison without asserting matched paper conditions."""
    values = []
    for domain in ('routerbench', 'mmlu'):
        group = [r for r in rows if r['domain'] == domain and r['variant'] == 'main'
                 and r['learning_fraction'] == 1 and r['budget_fraction'] == .5]
        means = {m: float(np.mean([r['service_utility'] for r in group if r['method'] == m]))
                 for m in sorted({r['method'] for r in group})}
        best_bandit = max((m for m in means if not m.startswith('fixed-')), key=means.get)
        values.append(dict(domain=domain, local_budget_fraction=.5, local_method_means=means,
            best_bandit=best_bandit, pilot_relative_to_best=means['pilot']/means[best_bandit]-1,
            published_headline=protocol['paper_comparison'][domain+'_headline'],
            matched_reproduction=False,
            paper_relative_error=None,
            reason='Different arm pool, embedding, preparation, budgets and reference-cost definition'))
    return values


def figures(output, rows, stress, costs):
    """Render scientific figures from saved numerical evidence, without paper images."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    folder = output/'figures'
    folder.mkdir()
    selected = next(s for s in stress if s['rounds'] == 2000 and s['alpha'] == 1.)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    for p, color in zip(selected['policies'], ('#C44B35', '#146C78')):
        ax.plot([r['round'] for r in p['trace']], [r['regret'] for r in p['trace']], label=p['method'], color=color, lw=2.5)
    ax.set(xlabel='Learning rounds', ylabel='Cumulative oracle regret',
           title='Synthetic magnitude stress: identical context and priors')
    ax.legend(frameon=False)
    fig.savefig(folder/'magnitude-stress.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    for label, group in costs.items():
        ax.plot([r['factor'] for r in group], [100*r['over_budget_count']/r['conditions'] for r in group],
                'o-', lw=2, label=label)
    ax.set(xlabel='Multiplier on recorded selected costs', ylabel='Conditions over budget (%)',
           title='Counterfactual accounting with unchanged routing decisions')
    ax.legend(frameon=False)
    fig.savefig(folder/'cost-sensitivity.png', dpi=180)
    plt.close(fig)


def validate_protocol(protocol):
    """Reject declared stress settings that this controlled implementation cannot execute."""
    spec=protocol['synthetic_magnitude_stress']
    rewards=spec['rewards']
    if len(rewards)!=2 or not all(math.isfinite(r) for r in rewards) or not 0<rewards[0]<rewards[1]<=1:
        raise ValueError('Invalid protocol rewards')
    expected={'context':[1.,0.], 'arms':['low','high'], 'lambda':1., 'seed':42,
              'prior_vectors':[[rewards[0],0.],[rewards[1],0.]]}
    for key,value in expected.items():
        if spec.get(key)!=value:
            raise ValueError('Unsupported stress protocol parameter: '+key)
    if 2000 not in spec['horizons'] or 1. not in spec['alphas']:
        raise ValueError('Stress figures require the 2000-round alpha=1 case')
    # Validate every declared case before creating any output directory.
    for n in spec['horizons']:
        if type(n) is not int or n<1:raise ValueError('Invalid protocol horizon')
    for alpha in spec['alphas']:
        if isinstance(alpha,bool) or not math.isfinite(alpha) or alpha<0:
            raise ValueError('Invalid protocol alpha')
    factors=protocol['cost_sensitivity']['factors']
    if not factors or any(isinstance(f,bool) or not math.isfinite(f) or f<=0 for f in factors):
        raise ValueError('Invalid cost factors')


def main(argv=None):
    """Execute the declared supplement only after protocol and source verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/course_analysis_protocol.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    from scripts.resources import require_resources
    preflight = require_resources()
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    validate_protocol(protocol)
    package = ROOT/protocol['frozen_package']
    integrity = verify_package(package)
    rows = read_rows(package)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    spec = protocol['synthetic_magnitude_stress']
    stress = [magnitude_stress(n, a, spec['rewards']) for n in spec['horizons'] for a in spec['alphas']]
    costs = {'all conditions': cost_sensitivity(rows, protocol['cost_sensitivity']['factors']),
             'main PILOT, full learning': cost_sensitivity([r for r in rows if r['variant'] == 'main'
                 and r['method'] == 'pilot' and r['learning_fraction'] == 1.], protocol['cost_sensitivity']['factors'])}
    result = dict(protocol=protocol, protocol_sha256=sha256(args.protocol), package_integrity=integrity,
                  source_csv_sha256=sha256(package/'aggregate/per-seed.csv'),
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  stress=stress, cost_sensitivity=costs, comparison=comparison(rows, protocol),
                  preflight=preflight, paid_api_calls=0)
    write_json(args.output_dir/'analysis.json', result)
    figures(args.output_dir, rows, stress, costs)
    write_json(args.output_dir/'package-files.json', {'files': {
        str(p.relative_to(args.output_dir)).replace('\\','/'): sha256(p)
        for p in sorted(args.output_dir.rglob('*')) if p.is_file()}})
    print(json.dumps({'output': str(args.output_dir), 'stress_cases': len(stress),
                      'recorded_conditions': len(rows), 'verified': True}))


if __name__ == '__main__':
    main()

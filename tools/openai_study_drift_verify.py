"""Independent two-arm cosine-UCB replay of every declared task-shift round."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/openai-followup-20261031/drift'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = read(path)
    if digest({k:v for k,v in value.items() if k != 'manifest_sha256'}) != value['manifest_sha256']:
        raise ValueError('Independent drift manifest checksum differs')
    return value


def unit(value):
    scale = np.abs(value).max()
    return (value / scale) / np.linalg.norm(value / scale) if scale else np.zeros_like(value)


def close(a,b):
    if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-11):
        raise ValueError(f'Independent drift numeric mismatch: {a} vs {b}')


def main():
    from scripts.resources import require_resources
    preflight = require_resources(); destination = BASE / 'independent-verification.json'
    if destination.exists():
        raise FileExistsError(destination)
    protocol = sealed(BASE / 'protocol.json'); artifact = sealed(BASE / 'manifest.json')
    if artifact['protocol_sha256'] != sha(BASE / 'protocol.json'):
        raise ValueError('Independent drift protocol lineage differs')
    for record in protocol['sources'] + [protocol['parent'],protocol['hybrid_cohort']]:
        if sha(ROOT / record['path']) != record['sha256']:
            raise ValueError('Frozen drift dependency changed')
    for name,fingerprint in artifact['files'].items():
        if sha(BASE / name) != fingerprint:
            raise ValueError('Drift artifact changed')
    core = ROOT / 'outputs/openai-core-20261031'
    index = sealed(core / 'prior-index.json')
    fit = next(r for r in index['fits'] if r['seed'] == protocol['seed'] and r['dimension'] == 64)
    if sha(ROOT / fit['prior']['path']) != fit['prior']['sha256']:
        raise ValueError('Original OpenAI prior changed')
    envelope = read(ROOT / fit['prior']['path'])
    if digest(envelope['payload']) != envelope['sha256']:
        raise ValueError('Prior envelope checksum differs')
    prior = envelope['payload']; arms = protocol['arms']; d = prior['dimension']
    theta = np.asarray([prior['theta'][prior['arms'].index(a)] for a in arms])
    lambdas = np.asarray([prior['metrics']['validation']['per_arm'][a]['lambda'] for a in arms])
    A = lambdas[:,None,None] * np.eye(d)[None,:,:]; b = lambdas[:,None] * theta
    cache_dir = ROOT / 'outputs/openai-standard-20261030/imported/caches/source-0'
    wrapped = read(cache_dir / 'manifest.json'); cache = wrapped['payload']
    if digest(cache) != wrapped['sha256'] or sha(cache_dir / 'vectors.npy') != cache['vector_sha256']:
        raise ValueError('Drift OpenAI cache checksum differs')
    vectors = np.load(cache_dir / 'vectors.npy',mmap_mode='r'); positions = {q:i for i,q in enumerate(cache['query_ids'])}
    W = np.asarray(prior['W']); bias = np.asarray(prior['b'])
    raw = ROOT / 'data/processed/routerbench_0shot'
    metadata = pd.read_parquet(raw / 'queries.parquet').set_index('query_id')
    outcomes = pd.read_parquet(raw / 'outcomes.parquet',columns=['query_id','arm_id','reward','actual_cost'])
    truth = {(r.query_id,r.arm_id):(float(r.reward),float(r.actual_cost)) for r in outcomes.itertuples(index=False) if r.arm_id in arms}
    expected = [(phase,position,q) for phase in ('warm','before','after') for position,q in enumerate(protocol['stream'][phase],1)]
    groups = [metadata.loc[q,'query_hash'] for _,_,q in expected]
    if len(expected) != 15000 or len(set(groups)) != 15000:
        raise ValueError('Drift prompt groups/counts differ')
    rows = []; phase_counts = dict.fromkeys(('warm','before','after'),0)
    with (BASE / 'trace.jsonl').open(encoding='utf-8') as stream:
        for t,((phase,position,q),line) in enumerate(zip(expected,stream,strict=True)):
            row = json.loads(line)
            if (row['round'],row['rounds_before_feedback'],row['phase'],row['phase_position'],row['query_id']) != (t+1,t,phase,position,q):
                raise ValueError('Drift identity/order/causality differs')
            x = unit(W @ vectors[positions[q]] + bias)
            estimated = np.linalg.solve(A,b[...,None])[...,0]
            means = np.clip(np.asarray([unit(v) @ x for v in estimated]),-1.,1.)
            directions = np.linalg.solve(A,np.broadcast_to(x,b.shape)[...,None])[...,0]
            std = np.sqrt(np.maximum(directions @ x,0.))
            scores = means + protocol['alpha'] * std; selected = int(np.argmax(scores)); arm = arms[selected]
            if row['arm_id'] != arm:
                raise ValueError('Independent drift selected arm differs')
            for a,score in zip(arms,scores):
                close(row['scores'][a],float(score))
            close(row['selected_uncertainty'],float(std[selected]))
            reward,cost = truth[q,arm]; reference_reward,reference_cost = truth[q,arms[0]]
            close(row['reward'],reward); close(row['actual_cost'],cost)
            close(row['gpt4_reward'],reference_reward); close(row['gpt4_cost'],reference_cost)
            oracle = max(truth[q,a][0] for a in arms)
            close(row['oracle_reward'],oracle); close(row['empirical_oracle_regret'],oracle-reward)
            A[selected] += np.outer(x,x); b[selected] += reward*x
            rows.append(row); phase_counts[phase] += 1
    checkpoint = read(BASE / 'checkpoint.json')
    if digest(checkpoint['payload']) != checkpoint['sha256']:
        raise ValueError('Drift final checkpoint envelope differs')
    state = checkpoint['payload']['state']
    if state['rounds'] != 15000 or state['arms'] != arms or state['parameters']['priors'] != theta.tolist() or state['parameters']['lambdas'] != lambdas.tolist():
        raise ValueError('Drift final prior/state identity differs')
    if not np.allclose(state['A'],A,rtol=1e-12,atol=1e-12) or not np.allclose(state['b'],b,rtol=1e-12,atol=1e-12):
        raise ValueError('Independent selected-feedback final matrices differ')
    reported = read(BASE / 'windows.json'); before = [r for r in rows if r['phase']=='before']; after = [r for r in rows if r['phase']=='after']; n=protocol['window']
    windows = {'all-evaluation':before+after,'all-before':before,'all-after':after,'last-before':before[-n:],'first-after':after[:n],'last-after':after[-n:]}
    for name,observations in windows.items():
        result=reported[name]; reward=math.fsum(r['reward'] for r in observations); cost=math.fsum(r['actual_cost'] for r in observations)
        reference_reward=math.fsum(r['gpt4_reward'] for r in observations); reference_cost=math.fsum(r['gpt4_cost'] for r in observations)
        if result['requested']!=len(observations) or result['served']!=len(observations):
            raise ValueError('Independent drift all-query denominator differs')
        for key,value in dict(coverage=1.,total_reward=reward,actual_recorded_cost=cost,service_utility=reward/len(observations),
            gpt4_total_reward=reference_reward,gpt4_total_cost=reference_cost,quality_fraction_gpt4=reward/reference_reward,cost_fraction_gpt4=cost/reference_cost).items():
            close(result[key],value)
    proof=dict(verified=True,rounds=15000,evaluation_queries=10000,phase_counts=phase_counts,prompt_groups=15000,
        all_selected_actions_scores_uncertainties_replayed=True,final_matrices_replayed=True,windows_replayed=len(windows),
        artifact_sha256=sha(BASE/'manifest.json'),protocol_sha256=sha(BASE/'protocol.json'),verifier_sha256=sha(__file__),
        preflight=preflight,paid_api_calls=0,synthetic=False,paper_equivalent=False)
    with destination.open('x',encoding='utf-8') as stream:
        json.dump(proof,stream,indent=2,allow_nan=False)
    print(json.dumps(proof),flush=True)


if __name__ == '__main__':
    main()

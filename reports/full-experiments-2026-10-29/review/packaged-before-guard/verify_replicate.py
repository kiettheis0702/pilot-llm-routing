"""Independent readback of the declared 64D prior; no production numerical helpers."""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--fit-name', required=True)
parser.add_argument('--dimension', type=int, required=True)
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()
DESTINATION = ROOT / ('outputs/research-prior-20261029/verification-' + args.fit_name + '.json')
if DESTINATION.exists():
    raise FileExistsError(DESTINATION)
from scripts.resources import require_resources
preflight = require_resources()
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def filehash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def envelope(path):
    document = read(path)
    assert set(document) == {'payload', 'sha256'}
    assert digest(document['payload']) == document['sha256']
    return document['payload']


fit_dir = ROOT / ('outputs/research-prior-20261029/' + args.fit_name)
prepared_dir = ROOT / 'outputs/local-pipeline-verification-20261029/prepared-identity'
prior = envelope(fit_dir / 'prior.json')
fit = envelope(fit_dir / 'fit-manifest.json')
prepared = read(prepared_dir / 'manifest.json')
assert prepared['manifest_sha256'] == digest({k:v for k,v in prepared.items() if k != 'manifest_sha256'})
for name, expected in prepared['files'].items():
    assert filehash(prepared_dir / name) == expected
assert fit['checkpoint_sha256'] == filehash(fit_dir / 'prior.json')
assert fit['prepared_sha256'] == filehash(prepared_dir / 'manifest.json')
assert not fit['synthetic'] and not fit['paper_equivalent'] and fit['prior_ready']
assert prior['dimension'] == args.dimension and prior['input_dimension'] == 1024
assert fit['arms'] == prepared['arms'] == prior['arms'] and len(prior['arms']) == 8
assert fit['metrics'] == prior['metrics']
protocol = prior['protocol']
assert (protocol['seed'], protocol['projection_epochs'], protocol['prior_epochs']) == (args.seed, 30, 40)
assert protocol['learning_rate'] == .03 and protocol['margin'] == .2

caches, matrices, norms = {}, {}, {}
expected_hashes = {
    'preference-identity': 'dc3a48c7f7f8e6deed3c317761baa72cf075051ce0e5ed11ba0bf7cea2758f8b',
    'routing-full': 'da4c06f8ad62999c7a187c2f08027b5bb7b378ac7a9c3a645830dc5bd94ea4ad',
}
W, b, theta = (np.asarray(prior[k], dtype=np.float64) for k in ('W', 'b', 'theta'))
assert W.shape == (args.dimension, 1024) and b.shape == (args.dimension,) and theta.shape == (8, args.dimension)
assert all(np.isfinite(v).all() for v in (W, b, theta))
for name, expected in expected_hashes.items():
    folder = ROOT / 'outputs/research-embeddings-20261029' / name
    cache = envelope(folder / 'manifest.json')
    assert digest(cache) == expected
    assert filehash(folder / 'vectors.npy') == cache['vector_sha256']
    assert cache['provenance'] == prior['embedding']['recipe'] and not cache['provenance']['synthetic']
    matrix = np.load(folder / 'vectors.npy', mmap_mode='r', allow_pickle=False)
    assert matrix.dtype == np.dtype('float32') and matrix.shape == (len(cache['query_ids']), 1024)
    minimum = float('inf')
    for start in range(0, len(matrix), 256):
        projected = matrix[start:start+256].astype(np.float64) @ W.T + b
        assert np.isfinite(projected).all()
        minimum = min(minimum, float(np.linalg.norm(projected, axis=1).min()))
    assert minimum > 1e-12
    caches[name], matrices[name], norms[name] = cache, matrix, minimum

cache = caches['preference-identity']
assert prior['embedding']['cache_sha256'] == expected_hashes['preference-identity']
assert prior['embedding']['query_ids'] == cache['query_ids']
assert prior['embedding']['prompt_hashes'] == cache['prompt_hashes']
inputs = read(prepared_dir / 'preference-inputs.json')
assert [r['query_id'] for r in inputs] == cache['query_ids']
assert [r['prompt_hash'] for r in inputs] == cache['prompt_hashes']
comparisons = sorted(read(prepared_dir / 'comparisons.json'), key=lambda r:r['battle_id'])
assert digest(comparisons) == prior['data']['comparisons_sha256']
assert digest([{k:r[k] for k in ('battle_id','query_id','split')} for r in comparisons]) == prior['data']['split_sha256']
by_id = {r['query_id']:r for r in comparisons}
assert len(by_id) == len(comparisons) == 1558
train = [r for r in comparisons if r['split'] == 'train']
holdout = [r for r in comparisons if r['split'] == 'holdout']
assert len(train) == 1245 and len(holdout) == 313
hashes = dict(zip(cache['query_ids'], cache['prompt_hashes']))
assert not {hashes[r['query_id']] for r in train} & {hashes[r['query_id']] for r in holdout}
sizes = read(ROOT / 'outputs/research-cost-inputs-20261029/cost-size-identity_match.json')
assert fit['cost_sizes'] == sizes
assert prior['data']['cost_sizes'] == sizes['sizes']
assert json.loads(prior['data']['cost_size_provenance']) == {
    'provenance': sizes['provenance'], 'source_urls': sizes['source_urls']}
assert not sizes['synthetic']
train_ids = {r['query_id'] for r in train}
triplets = prior['data']['sampled_triplets']
assert len(triplets) == 1193 and len({a for a, _, _ in triplets}) == len(triplets)
for a, p, n in triplets:
    assert {a, p, n} <= train_ids and len({hashes[q] for q in (a,p,n)}) == 3
    winner = by_id[a]['winner']
    assert by_id[p]['winner'] == by_id[n]['loser'] == winner
    assert sizes['sizes'][by_id[n]['winner']] < sizes['sizes'][winner]

positions = {q:i for i,q in enumerate(cache['query_ids'])}
arm_index = {a:i for i,a in enumerate(prior['arms'])}
vectors = np.asarray(matrices['preference-identity'], dtype=np.float64)
projected = vectors @ W.T + b
unit = projected / np.linalg.norm(projected, axis=1, keepdims=True)
theta_norms = np.linalg.norm(theta, axis=1, keepdims=True)
assert np.isfinite(theta_norms).all() and (theta_norms > 1e-12).all()
theta_unit = theta / theta_norms
scores = unit @ theta_unit.T
statistics = {a:{'support':0, 'correct':0} for a in prior['arms']}
correct = 0
for row in holdout:
    i = positions[row['query_id']]
    right = int(scores[i,arm_index[row['winner']]] > scores[i,arm_index[row['loser']]])
    correct += right
    for arm in (row['winner'], row['loser']):
        statistics[arm]['support'] += 1
        statistics[arm]['correct'] += right
validation = prior['metrics']['validation']
assert validation['all_lambdas_ready'] and not validation['blockers']
for arm, counts in statistics.items():
    reported = validation['per_arm'][arm]
    assert all(reported[k] == v for k,v in counts.items())
    assert counts['correct'] > 0 and counts['support'] > 0
    assert reported['accuracy'] == counts['correct'] / counts['support']
    assert reported['lambda'] == counts['support'] / counts['correct']
assert sum(r['correct'] for r in statistics.values()) == 2 * correct

final_triplet_loss = math.fsum(max(0., protocol['margin']
    - float(unit[positions[a]] @ unit[positions[p]])
    + float(unit[positions[a]] @ unit[positions[n]])) for a,p,n in triplets) / len(triplets)
final_prior_loss = math.fsum(float(np.logaddexp(0.,
    scores[positions[r['query_id']],arm_index[r['loser']]]
    - scores[positions[r['query_id']],arm_index[r['winner']]])) for r in train) / len(train)
for key, count, recomputed in (('projection_losses',31,final_triplet_loss),('prior_losses',41,final_prior_loss)):
    losses = prior['metrics'][key]
    assert len(losses) == count and all(math.isfinite(v) and v >= 0 for v in losses)
    assert math.isclose(losses[-1], recomputed, rel_tol=1e-12, abs_tol=1e-12)

result = dict(version=1,kind='independent-replicate-prior-verification',verified=True,
    synthetic=False,paper_equivalent=False,preflight=preflight,
    checkpoint_sha256=fit['checkpoint_sha256'],fit_manifest_sha256=filehash(fit_dir/'fit-manifest.json'),
    prepared_sha256=fit['prepared_sha256'],embedding_cache_sha256=expected_hashes,
    script_sha256=filehash(__file__),dimension=args.dimension,seed=args.seed,train_pairs=len(train),holdout_pairs=len(holdout),
    holdout_correct=correct,holdout_accuracy=correct/len(holdout),per_arm=validation['per_arm'],
    validated_training_triplets=len(triplets),projected_min_norm=norms,
    recomputed_final_projection_loss=final_triplet_loss,recomputed_final_prior_loss=final_prior_loss,
    interpretation='independent checkpoint identities, separation, triplet constraints, all-context directions, holdout coefficients and final objectives; not an epoch-by-epoch training replay')
with DESTINATION.open('x',encoding='utf-8') as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({'verified':True,'output':str(DESTINATION),'correct':correct,'holdout':len(holdout),
                  'final_projection_loss':final_triplet_loss,'final_prior_loss':final_prior_loss}))

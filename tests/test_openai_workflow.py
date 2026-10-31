"""Offline API-format recovery and train/calibration/test separation."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.data_gate import query_hash
from scripts.embedding_store import DiskEmbeddingCache
from scripts.pretraining import load_checkpoint
from tools.paper_preflight.embedding_plan import build_plan


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    return path


def write_lines(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    return path


def response_fixture(tmp_path):
    paths = []
    for name, values in [('routing', [('q1', 'alpha'), ('q2', 'beta'), ('q3', 'alpha')]),
                         ('preference', [('p1', 'gamma'), ('p2', 'delta')])]:
        paths.append(write_json(tmp_path/(name+'.json'), [dict(query_id=q, text=text,
            prompt_hash=query_hash(text)) for q, text in values]))
    plan = tmp_path/'plan'
    build_plan(paths, plan, batch_size=2)
    requests = [json.loads(s) for s in (plan/'requests.jsonl').open(encoding='utf-8')]
    vectors = {}
    for i, text in enumerate(['alpha', 'beta', 'gamma', 'delta']):
        vector = [0.] * 1536
        vector[i] = 1.
        vectors[text] = vector
    responses = [dict(id=f'batch_{i}', custom_id=r['custom_id'], error=None,
        response=dict(status_code=200, request_id=f'req_{i}', body=dict(object='list',
            model='text-embedding-3-small', usage=dict(prompt_tokens=2, total_tokens=2),
            data=[dict(object='embedding', index=j, embedding=vectors[text])
                  for j, text in reversed(list(enumerate(r['body']['input'])))])))
        for i, r in enumerate(requests)]
    return plan, paths, responses


def test_import_recovers_reordered_requests_indices_and_duplicate_text(tmp_path):
    from tools.paper_preflight.import_embeddings import import_batch
    plan, paths, responses = response_fixture(tmp_path)
    outputs = write_lines(tmp_path/'responses.jsonl', responses[::-1])
    folder = tmp_path/'imported'
    result = import_batch(plan, [outputs], paths, folder, synthetic=True)
    assert result['status'] == 'complete'
    assert result['accepted_requests'] == 2
    assert result['missing_custom_ids'] == []
    assert result['usage_total_tokens'] == 4
    assert result['api_calls_performed'] == 0
    cache = DiskEmbeddingCache.load(folder/result['caches']['routing.json']['path'])
    assert cache.dimension == 1536 and cache.provenance['synthetic'] is True
    assert cache.ids == ('q1', 'q2', 'q3')
    np.testing.assert_array_equal(cache.get('q1'), cache.get('q3'))
    assert cache.get('q1')[0] == 1 and cache.get('q2')[1] == 1
    with pytest.raises(FileExistsError):
        import_batch(plan, [outputs], paths, folder, synthetic=True)


def test_partial_import_saves_accepted_request_then_resumes_without_new_generation(tmp_path):
    from tools.paper_preflight.import_embeddings import import_batch
    plan, paths, responses = response_fixture(tmp_path)
    partial = write_lines(tmp_path/'partial.jsonl', responses[:1])
    folder = tmp_path/'imported'
    first = import_batch(plan, [partial], paths, folder, synthetic=True)
    assert first['status'] == 'incomplete' and first['accepted_requests'] == 1
    assert first['missing_custom_ids'] == [responses[1]['custom_id']]
    assert first['caches'] == {}
    assert first['accepted_request_ids'] == ['req_0']
    whole = write_lines(tmp_path/'whole.jsonl', responses[::-1])
    final = import_batch(plan, [whole], paths, folder, resume=True, synthetic=True)
    assert final['status'] == 'complete' and final['accepted_requests'] == 2
    again = import_batch(plan, [whole], paths, folder, resume=True, synthetic=True)
    assert again['caches'] == final['caches']
    assert again['usage_total_tokens'] == 4  # No double counting on reread.


def test_failed_request_is_reported_without_publishing_a_cache_or_retrying(tmp_path):
    from tools.paper_preflight.import_embeddings import import_batch
    plan, paths, responses = response_fixture(tmp_path)
    failed = dict(custom_id=responses[0]['custom_id'], response=None,
                  error=dict(code='batch_expired', message='expired'))
    result = import_batch(plan, [write_lines(tmp_path/'failed.jsonl', [failed])], paths,
                          tmp_path/'imported', synthetic=True)
    assert result['status'] == 'incomplete' and result['accepted_requests'] == 0
    assert len(result['missing_custom_ids']) == 2
    assert result['request_errors'][failed['custom_id']]['code'] == 'batch_expired'
    assert result['api_calls_performed'] == 0 and result['caches'] == {}


def test_truncated_output_keeps_previously_accepted_request_for_resume(tmp_path):
    from tools.paper_preflight.import_embeddings import import_batch
    from scripts.pretraining import _read_json
    plan, paths, responses = response_fixture(tmp_path)
    broken = tmp_path/'truncated.jsonl'
    broken.write_text(json.dumps(responses[0])+'\n{"custom_id":', encoding='utf-8')
    folder = tmp_path/'imported'
    with pytest.raises(ValueError):
        import_batch(plan, [broken], paths, folder, synthetic=True)
    state = _read_json(folder/'progress.json')
    assert state['accepted'][responses[0]['custom_id']]['request_id'] == 'req_0'
    recovered = import_batch(plan, [write_lines(tmp_path/'whole.jsonl', responses)], paths,
                             folder, resume=True, synthetic=True)
    assert recovered['status'] == 'complete' and recovered['usage_total_tokens'] == 4


@pytest.mark.parametrize('problem', ['model', 'dimension', 'boolean', 'nan', 'zero',
    'missing_index', 'duplicate_index', 'unknown_request', 'missing_request_id'])
def test_import_rejects_invalid_success_responses_before_publishing(tmp_path, problem):
    from tools.paper_preflight.import_embeddings import import_batch
    plan, paths, responses = response_fixture(tmp_path)
    record = responses[0]
    body = record['response']['body']
    if problem == 'model': body['model'] = 'text-embedding-3-large'
    if problem == 'dimension': body['data'][0]['embedding'] = [1., 0.]
    if problem == 'boolean': body['data'][0]['embedding'][0] = True
    if problem == 'nan': body['data'][0]['embedding'][0] = float('nan')
    if problem == 'zero': body['data'][0]['embedding'] = [0.] * 1536
    if problem == 'missing_index': body['data'] = body['data'][:1]
    if problem == 'duplicate_index': body['data'][0]['index'] = body['data'][1]['index']
    if problem == 'unknown_request': record['custom_id'] = '../../foreign'
    if problem == 'missing_request_id': del record['response']['request_id']
    folder = tmp_path/'imported'
    with pytest.raises(ValueError):
        import_batch(plan, [write_lines(tmp_path/'bad.jsonl', responses)], paths, folder, synthetic=True)
    assert not list(folder.glob('caches/*/manifest.json'))


def test_import_detects_changed_source_conflicting_receipt_and_corrupt_chunk(tmp_path):
    from tools.paper_preflight.import_embeddings import import_batch
    plan, paths, responses = response_fixture(tmp_path)
    whole = write_lines(tmp_path/'whole.jsonl', responses)
    folder = tmp_path/'imported'
    import_batch(plan, [whole], paths, folder, synthetic=True)
    changed = copy.deepcopy(responses)
    changed[0]['response']['request_id'] = 'another_paid_request'
    with pytest.raises(ValueError, match='[Cc]onflict'):
        import_batch(plan, [write_lines(tmp_path/'changed.jsonl', changed)], paths,
                     folder, resume=True, synthetic=True)
    chunk = next((folder/'chunks').glob('*.npy'))
    with chunk.open('ab') as handle: handle.write(b'damaged')
    with pytest.raises(ValueError, match='checksum'):
        import_batch(plan, [whole], paths, folder, resume=True, synthetic=True)
    paths[0].write_text('[]', encoding='utf-8')
    with pytest.raises(ValueError, match='source'):
        import_batch(plan, [whole], paths, tmp_path/'other', synthetic=True)


def test_imported_api_format_cache_aligns_with_existing_pipeline_and_exact_text(tmp_path):
    from tools.paper_preflight.import_embeddings import import_batch
    from scripts.research_pipeline import _aligned_cache
    # Formatter fixture only: this exercises the non-synthetic provenance gate,
    # and is not reported as real OpenAI vectors or performance.
    plan, paths, responses = response_fixture(tmp_path)
    result = import_batch(plan, [write_lines(tmp_path/'responses.jsonl', responses)], paths,
                          tmp_path/'imported')
    path = tmp_path/'imported'/result['caches']['routing.json']['path']
    rows = json.loads(paths[0].read_text(encoding='utf-8'))
    cache = _aligned_cache(path, rows)
    assert cache.dimension == 1536 and cache.provenance['model'] == 'text-embedding-3-small'
    changed = copy.deepcopy(rows)
    changed[0]['text'] = ' alpha '  # Same normalized prompt hash, different encoder text.
    with pytest.raises(ValueError, match='[Ii]nput|[Vv]erbatim'):
        _aligned_cache(path, changed)


def training_fixture(tmp_path, *, flipped_test=False):
    ids, texts, rows, values = [], [], [], []
    for i in range(16):
        q, text = f'p{i}', f'distinct prompt {i}'
        large = i % 2 == 0
        phase = 'train' if i < 8 else 'calibration' if i < 12 else 'test'
        vector = np.zeros(1536, dtype=np.float32)
        vector[:2] = [1 if large else -1, 3]
        vector /= np.linalg.norm(vector)
        if phase == 'test' and flipped_test: vector[0] *= -1
        winner, loser = ('large', 'small') if large else ('small', 'large')
        if phase == 'test' and flipped_test: winner, loser = loser, winner
        ids.append(q); texts.append(text); values.append(vector)
        rows.append(dict(battle_id=f'b{i:02d}', query_id=q, winner=winner, loser=loser, split=phase))
    recipe = dict(model='synthetic1536', version='fixture', source='offline test', normalization='L2', synthetic=True)
    cache = DiskEmbeddingCache.create(tmp_path/'cache', ids, [query_hash(s) for s in texts], np.array(values), recipe)
    comparisons = write_json(tmp_path/'comparisons.json', rows)
    sizes = write_json(tmp_path/'sizes.json', dict(sizes={'large': 4., 'small': 1.}, provenance='synthetic fixture', synthetic=True))
    return cache, comparisons, sizes


CANDIDATES = [dict(id='short', dimension=2, projection_epochs=2, prior_epochs=5),
              dict(id='long', dimension=2, projection_epochs=4, prior_epochs=8)]


def test_training_keeps_calibration_lambda_and_final_test_separate(tmp_path):
    from tools.paper_preflight.train import run_training
    cache, comparisons, sizes = training_fixture(tmp_path)
    output = tmp_path/'trained'
    result = run_training(tmp_path/'cache', comparisons, sizes, output,
                          candidates=CANDIDATES, allow_synthetic=True)
    assert result['counts'] == dict(train=8, calibration=4, test=4)
    assert len(result['candidates']) == 2
    assert result['selected_candidate'] in {'short', 'long'}
    selected = load_checkpoint(output/'selected-prior.json')
    assert sum(v['support'] for v in selected['metrics']['validation']['per_arm'].values()) == 8
    final = json.loads((output/'final-test.json').read_text(encoding='utf-8'))
    assert final['pairs'] == 4 and final['role'] == 'final_test_only'
    assert final['used_for_selection'] is False
    assert result['synthetic'] is True and result['api_calls_performed'] == 0


def test_final_test_changes_do_not_change_candidate_choice_or_fitted_weights(tmp_path):
    from tools.paper_preflight.train import run_training
    checkpoints, results = [], []
    for name, changed in [('original', False), ('changed', True)]:
        folder = tmp_path/name; folder.mkdir()
        _, comparisons, sizes = training_fixture(folder, flipped_test=changed)
        results.append(run_training(folder/'cache', comparisons, sizes, folder/'trained',
            candidates=CANDIDATES, allow_synthetic=True))
        checkpoints.append(load_checkpoint(folder/'trained/selected-prior.json'))
    assert results[0]['selected_candidate'] == results[1]['selected_candidate']
    for key in ('W', 'b', 'theta'):
        np.testing.assert_array_equal(checkpoints[0][key], checkpoints[1][key])
    assert checkpoints[0]['metrics']['validation'] == checkpoints[1]['metrics']['validation']


def test_training_rejects_synthetic_by_default_and_cross_partition_prompt_leakage(tmp_path):
    from tools.paper_preflight.train import run_training
    _, comparisons, sizes = training_fixture(tmp_path)
    with pytest.raises(ValueError, match='synthetic'):
        run_training(tmp_path/'cache', comparisons, sizes, tmp_path/'denied', candidates=CANDIDATES)
    rows = json.loads(comparisons.read_text(encoding='utf-8'))
    rows[-1]['query_id'] = rows[8]['query_id']
    write_json(comparisons, rows)
    with pytest.raises(ValueError, match='group'):
        run_training(tmp_path/'cache', comparisons, sizes, tmp_path/'leaked',
            candidates=CANDIDATES, allow_synthetic=True)
    assert not (tmp_path/'leaked').exists()


def test_training_rejects_invalid_candidate_before_output(tmp_path):
    from tools.paper_preflight.train import run_training
    _, comparisons, sizes = training_fixture(tmp_path)
    bad = [dict(id='../outside', dimension=2, projection_epochs=1, prior_epochs=1)]
    with pytest.raises(ValueError):
        run_training(tmp_path/'cache', comparisons, sizes, tmp_path/'denied',
            candidates=bad, allow_synthetic=True)
    assert not (tmp_path/'denied').exists()


def test_real_vectors_with_synthetic_cost_sizes_require_permission_and_stay_labeled(tmp_path):
    from tools.paper_preflight.train import run_training
    cache, comparisons, sizes = training_fixture(tmp_path)
    real_recipe = dict(cache.provenance, model='real-format-fixture', synthetic=False)
    DiskEmbeddingCache.create(tmp_path/'real-cache', cache.ids, cache.prompt_hashes,
                              cache.vectors, real_recipe)
    with pytest.raises(ValueError, match='synthetic'):
        run_training(tmp_path/'real-cache', comparisons, sizes, tmp_path/'denied', candidates=CANDIDATES)
    result = run_training(tmp_path/'real-cache', comparisons, sizes, tmp_path/'allowed',
                          candidates=CANDIDATES, allow_synthetic=True)
    assert result['synthetic'] is True
    assert json.loads((tmp_path/'allowed/final-test.json').read_text(encoding='utf-8'))['synthetic'] is True


def test_training_requires_explicit_boolean_cost_size_provenance(tmp_path):
    from tools.paper_preflight.train import run_training
    _, comparisons, sizes = training_fixture(tmp_path)
    value = json.loads(sizes.read_text(encoding='utf-8'))
    value.pop('synthetic')
    write_json(sizes, value)
    with pytest.raises(ValueError, match='synthetic'):
        run_training(tmp_path/'cache', comparisons, sizes, tmp_path/'denied',
            candidates=CANDIDATES, allow_synthetic=True)
    assert not (tmp_path/'denied').exists()

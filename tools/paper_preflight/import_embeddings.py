"""Import downloaded OpenAI Batch outputs offline; never submit or retry a request."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from uuid import uuid4

import numpy as np

from scripts.data_gate import query_hash
from scripts.embedding_store import DiskEmbeddingCache
from scripts.openai_cache import MODEL, DIMENSION, provenance as openai_provenance
from scripts.pretraining import _digest, _read_json, _unique_object, _write_json
from tools.paper_preflight.common import sha256, write_json

def _load(path):
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle, object_pairs_hook=_unique_object)


def _lines(path):
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip() or len(line.encode('utf-8')) > 32 * 1024 * 1024:
                raise ValueError('Invalid or oversized JSONL record')
            yield json.loads(line, object_pairs_hook=_unique_object)


def _owned(root, relative):
    relative = Path(relative)
    target = root/relative
    if relative.is_absolute() or '..' in relative.parts or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Import artifact path escapes output')
    for member in (target, *target.parents):
        if member == root:
            break
        if member.is_symlink():
            raise ValueError('Import artifacts cannot be symlinks')
    return target


def load_plan(plan_dir, input_paths):
    plan = Path(plan_dir)
    manifest = _load(plan/'manifest.json')
    if (manifest.get('kind') != 'offline-openai-embedding-plan'
            or manifest.get('model') != MODEL or manifest.get('dimensions') != DIMENSION
            or set(manifest.get('files', {})) != {'requests.jsonl', 'id-map.json'}):
        raise ValueError('Unsupported embedding plan')
    for name, expected in manifest['files'].items():
        if sha256(plan/name) != expected:
            raise ValueError('Embedding plan file checksum mismatch')
    specs = manifest['source_files']
    if len(specs) != len(input_paths) or len({s['file'].casefold() for s in specs}) != len(specs):
        raise ValueError('Embedding source file identities differ')
    requests, positions = {}, {}
    for index, request in enumerate(_lines(plan/'requests.jsonl')):
        custom, body = request.get('custom_id'), request.get('body', {})
        if (not isinstance(custom, str) or not custom or custom in requests
                or request.get('method') != 'POST' or request.get('url') != '/v1/embeddings'
                or body.get('model') != MODEL or body.get('encoding_format') != 'float'
                or 'dimensions' in body or not isinstance(body.get('input'), list) or not body['input']):
            raise ValueError('Invalid prepared embedding request')
        if any(not isinstance(text, str) or not text.strip() for text in body['input']):
            raise ValueError('Invalid prepared embedding text')
        requests[custom] = dict(index=index, texts=body['input'])
        for j, text in enumerate(body['input']):
            positions[custom, j] = hashlib.sha256(text.encode('utf-8')).hexdigest()
    mapping, sources, seen = _load(plan/'id-map.json'), [], set()
    used = set()
    for spec, path in zip(specs, input_paths):
        path = Path(path)
        if path.name != spec['file'] or sha256(path) != spec['sha256']:
            raise ValueError('Embedding source checksum or name mismatch')
        rows = _load(path)
        if not isinstance(rows, list) or len(rows) != spec['rows']:
            raise ValueError('Embedding source row count mismatch')
        for row in rows:
            q, text = row.get('query_id'), row.get('text')
            if not isinstance(q, str) or not q.strip() or q in seen or q not in mapping:
                raise ValueError('Invalid embedding source identity')
            if not isinstance(text, str) or not text.strip():
                raise ValueError('Invalid embedding source text')
            if row.get('prompt_hash', query_hash(text)) != query_hash(text):
                raise ValueError('Embedding source prompt hash mismatch')
            seen.add(q)
            location = mapping[q]
            if not isinstance(location, dict) or type(location.get('index')) is not int:
                raise ValueError('Invalid embedding ID mapping')
            key = location.get('custom_id'), location['index']
            hashed = hashlib.sha256(text.encode('utf-8')).hexdigest()
            if key not in positions or positions[key] != hashed or location.get('text_sha256') != hashed:
                raise ValueError('Embedding source/request text identity mismatch')
            used.add(key)
        sources.append(dict(spec=spec, rows=rows))
    if (seen != set(mapping) or used != set(positions) or len(seen) != manifest['source_rows']
            or len(positions) != manifest['unique_texts'] or len(requests) != manifest['requests']):
        raise ValueError('Embedding plan coverage mismatch')
    return manifest, requests, mapping, sources


def _persist(folder, state):
    temporary = folder/f'.progress-{uuid4().hex}.tmp'
    try:
        _write_json(temporary, state)
        with temporary.open('r+b') as stream:
            os.fsync(stream.fileno())
        temporary.replace(folder/'progress.json')
    finally:
        temporary.unlink(missing_ok=True)


def _chunk(folder, relative, matrix):
    target = _owned(folder, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name('.'+uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as handle:
            np.save(handle, matrix, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        fingerprint = sha256(temporary)
        if target.exists():
            # An interrupted receipt write may leave a valid owned chunk.
            if sha256(target) != fingerprint:
                raise ValueError('Orphan embedding chunk conflicts with recovered response')
        else:
            os.link(temporary, target)
        return fingerprint
    finally:
        temporary.unlink(missing_ok=True)


def _success(record, count):
    response = record.get('response')
    if not isinstance(response, dict) or type(response.get('status_code')) is not int or response['status_code'] != 200:
        raise ValueError('Invalid successful response')
    request_id = response.get('request_id')
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError('Successful response requires its provider request ID')
    body = response.get('body', {})
    if not isinstance(body, dict) or body.get('model') != MODEL or body.get('object') != 'list':
        raise ValueError('Embedding response model or object mismatch')
    data, usage = body.get('data'), body.get('usage', {})
    if (not isinstance(data, list) or len(data) != count
            or any(type(usage.get(k)) is not int or usage[k] < 0 for k in ('prompt_tokens', 'total_tokens'))
            or usage['total_tokens'] < usage['prompt_tokens']):
        raise ValueError('Embedding response coverage or usage mismatch')
    matrix, seen = np.empty((count, DIMENSION), dtype=np.float32), set()
    for item in data:
        if not isinstance(item, dict):
            raise ValueError('Embedding data must contain objects')
        index, vector = item.get('index'), item.get('embedding')
        if (item.get('object') != 'embedding' or type(index) is not int
                or not 0 <= index < count or index in seen or not isinstance(vector, list)
                or len(vector) != DIMENSION
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector)):
            raise ValueError('Invalid embedding index, dimension or numeric values')
        seen.add(index)
        with np.errstate(over='ignore', invalid='ignore'):
            matrix[index] = vector
    if (not np.isfinite(matrix).all()
            or not np.allclose(np.linalg.norm(matrix, axis=1), 1., rtol=1e-3, atol=1e-4)):
        raise ValueError('Embedding response contains non-unit or nonfinite float32 vectors')
    return matrix, request_id, dict(usage)


def import_batch(plan_dir, output_files, input_paths, output_dir, *, resume=False, synthetic=False):
    folder = Path(output_dir)
    if type(synthetic) is not bool or type(resume) is not bool:
        raise ValueError('Import mode flags must be Boolean')
    if folder.exists() and not resume:
        raise FileExistsError(folder)
    if folder.is_symlink():
        raise ValueError('Import output cannot be a symlink')
    manifest, requests, mapping, sources = load_plan(plan_dir, input_paths)
    identity = _digest(manifest)
    if folder.exists():
        state = _read_json(folder/'progress.json')
        if state.get('identity') != identity or state.get('synthetic') is not synthetic:
            raise ValueError('Cannot resume a changed embedding plan or synthetic mode')
        for custom, entry in state['accepted'].items():
            if custom not in requests or entry['chunk'] != f"chunks/{requests[custom]['index']:06d}.npy":
                raise ValueError('Unknown persisted embedding request')
            chunk = _owned(folder, entry['chunk'])
            if sha256(chunk) != entry['vector_sha256']:
                raise ValueError('Persisted embedding chunk checksum mismatch')
    else:
        folder.mkdir(parents=True)
        state = dict(version=1, identity=identity, synthetic=synthetic, accepted={}, request_errors={}, caches={})
        _persist(folder, state)
    for file in output_files:
        for record in _lines(file):
            if not isinstance(record, dict) or record.get('custom_id') not in requests:
                raise ValueError('Unknown embedding request custom_id')
            custom = record['custom_id']
            record_sha = _digest(record)
            if custom in state['accepted']:
                if state['accepted'][custom]['response_sha256'] != record_sha:
                    raise ValueError('Conflicting response for an accepted embedding request')
                continue
            response = record.get('response')
            if record.get('error') is not None or (isinstance(response, dict) and response.get('status_code') != 200):
                error = record.get('error') or (response.get('body', {}).get('error'))
                if not isinstance(error, dict):
                    raise ValueError('Invalid failed-request record')
                state['request_errors'][custom] = error
                _persist(folder, state)
                continue
            matrix, request_id, usage = _success(record, len(requests[custom]['texts']))
            if any(entry['request_id'] == request_id for entry in state['accepted'].values()):
                raise ValueError('Provider request ID is reused by another custom_id')
            relative = f"chunks/{requests[custom]['index']:06d}.npy"
            fingerprint = _chunk(folder, relative, matrix)
            state['accepted'][custom] = dict(chunk=relative, vector_sha256=fingerprint,
                request_id=request_id, usage=usage, response_sha256=record_sha)
            state['request_errors'].pop(custom, None)
            _persist(folder, state)  # Durable after EACH accepted provider request.
    missing = [custom for custom in requests if custom not in state['accepted']]
    if not missing:
        provenance = openai_provenance(synthetic)
        for i, source in enumerate(sources):
            relative = f'caches/source-{i}'
            path = _owned(folder, relative)
            ids = [r['query_id'] for r in source['rows']]
            hashes = [query_hash(r['text']) for r in source['rows']]
            if (path/'manifest.json').exists():
                cache = DiskEmbeddingCache.load(path)
                if cache.ids != tuple(ids) or cache.provenance != provenance or cache.metadata.get('plan_identity') != identity:
                    raise ValueError('Published embedding cache identity mismatch')
            else:
                path.mkdir(parents=True, exist_ok=True)
                matrix = np.lib.format.open_memmap(path/'vectors.npy', mode='w+', dtype=np.float32,
                                                  shape=(len(ids), DIMENSION))
                groups = {}
                for row_index, q in enumerate(ids):
                    location = mapping[q]
                    groups.setdefault(location['custom_id'], []).append((row_index, location['index']))
                for custom, members in groups.items():
                    vectors = np.load(_owned(folder, state['accepted'][custom]['chunk']), mmap_mode='r', allow_pickle=False)
                    for row_index, vector_index in members:
                        matrix[row_index] = vectors[vector_index]
                matrix.flush()
                del matrix
                cache = DiskEmbeddingCache.publish(path, ids, hashes, provenance,
                    metadata=dict(plan_identity=identity, source_sha256=source['spec']['sha256'],
                        input_rows_sha256=_digest(source['rows']),
                        response_authenticity='caller-provided local files; not independently attested',
                        provider_request_ids=[state['accepted'][c]['request_id'] for c in requests],
                        synthetic=synthetic))
            state['caches'][source['spec']['file']] = dict(path=relative, rows=len(ids),
                dimension=cache.dimension, fingerprint=cache.fingerprint())
            _persist(folder, state)
    result = dict(version=1, status='incomplete' if missing else 'complete',
        accepted_requests=len(state['accepted']), missing_custom_ids=missing,
        accepted_request_ids=[state['accepted'][c]['request_id'] for c in requests if c in state['accepted']],
        request_errors=state['request_errors'], caches=state['caches'],
        usage_total_tokens=sum(v['usage']['total_tokens'] for v in state['accepted'].values()),
        planned_tokens=manifest['tokens'], plan_identity=identity, synthetic=synthetic,
        response_authenticity='caller-provided local files; not independently attested',
        api_calls_performed=0, automatic_resubmission_supported=False)
    write_json(folder/'receipt.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-dir', type=Path, required=True)
    parser.add_argument('--outputs', type=Path, nargs='+', required=True)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args(argv)
    from scripts.resources import require_resources
    require_resources()
    result = import_batch(args.plan_dir, args.outputs, args.inputs, args.output_dir,
                          resume=args.resume, synthetic=args.synthetic)
    print({k: result[k] for k in ('status', 'accepted_requests', 'missing_custom_ids', 'api_calls_performed')})


if __name__ == '__main__':
    main()

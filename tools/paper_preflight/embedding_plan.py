"""Prepare OpenAI Batch request files OFFLINE. Deliberately has no submit path."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path

from scripts.data_gate import query_hash
from tools.paper_preflight.common import read_json, sha256, write_json


def build_plan(input_paths, output_dir, *, batch_size=128, max_batch_tokens=50000,
               price_per_million=.02):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    if (type(batch_size) is not int or not 1 <= batch_size <= 2048
            or type(max_batch_tokens) is not int or not 1 <= max_batch_tokens <= 300000
            or isinstance(price_per_million, bool) or not isinstance(price_per_million, (int, float))
            or not math.isfinite(price_per_million) or price_per_million < 0):
        raise ValueError('Invalid embedding configuration')
    import tiktoken
    encoding = tiktoken.get_encoding('cl100k_base')
    seen_ids, texts, entries, sources = set(), {}, [], []
    for path in input_paths:
        rows = read_json(path)
        if not isinstance(rows, list) or not rows:
            raise ValueError('Embedding inputs must be a nonempty list')
        sources.append(dict(file=Path(path).name, sha256=sha256(path), rows=len(rows)))
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('Embedding row must be an object')
            q, text = row.get('query_id'), row.get('text')
            if not isinstance(q, str) or not q.strip() or q in seen_ids:
                raise ValueError('Duplicate or invalid embedding identity')
            if not isinstance(text, str) or not text.strip():
                raise ValueError('Embedding text must be nonempty')
            if 'prompt_hash' in row and row['prompt_hash'] != query_hash(text):
                raise ValueError('Embedding prompt hash mismatch')
            seen_ids.add(q)
            hashed = hashlib.sha256(text.encode('utf-8')).hexdigest()
            if hashed not in texts:
                tokens = len(encoding.encode_ordinary(text))
                if not 1 <= tokens <= 8191 or tokens > max_batch_tokens:
                    raise ValueError(f'Embedding input token limit exceeded for {q}')
                texts[hashed] = dict(text=text, tokens=tokens)
            entries.append((q, hashed))
    if not sources:
        raise ValueError('At least one embedding input file is required')
    requests, positions, batch, tokens = [], {}, [], 0

    def finish():
        if not batch:
            return
        custom_id = f'pilot-embed-{len(requests):06d}'
        requests.append(dict(custom_id=custom_id, method='POST', url='/v1/embeddings',
            body=dict(model='text-embedding-3-small', input=[texts[h]['text'] for h in batch], encoding_format='float')))
        for index, hashed in enumerate(batch):
            positions[hashed] = dict(custom_id=custom_id, index=index, text_sha256=hashed)

    for hashed, value in texts.items():
        if batch and (len(batch) >= batch_size or tokens + value['tokens'] > max_batch_tokens):
            finish()
            batch, tokens = [], 0
        batch.append(hashed)
        tokens += value['tokens']
    finish()
    total = sum(v['tokens'] for v in texts.values())
    result = dict(version=1, kind='offline-openai-embedding-plan', model='text-embedding-3-small',
        dimensions=1536, dimensions_parameter='omitted: OpenAI default, not a confirmed author dimension',
        source_rows=len(entries), unique_texts=len(texts), duplicate_text_rows=len(entries) - len(texts),
        tokens=total, requests=len(requests), batch_size=batch_size, max_batch_tokens=max_batch_tokens,
        per_input_token_cap=8191, tokenizer='cl100k_base', tokenizer_version=version('tiktoken'),
        price_source='https://developers.openai.com/api/docs/models/text-embedding-3-small',
        standard_usd_per_million_tokens=price_per_million,
        estimated_standard_usd=total * price_per_million / 1_000_000,
        estimate_note='Standard-price estimate; no Batch discount assumed; excludes tax or later reruns',
        source_files=sources, paid_api_calls=0, approved=False,
        submission_supported=False, eleven_model_verified=False,
        cache_identity='model + default dimensions + exact original UTF-8 text SHA256',
        prompt_construction='verbatim exported text; no instruction, truncation or whitespace normalization',
        pending='final scientific protocol and paid-run approval; these files are never submitted by this tool')
    output.mkdir(parents=True)
    with (output/'requests.jsonl').open('w', encoding='utf-8', newline='\n') as handle:
        for request in requests:
            handle.write(json.dumps(request, ensure_ascii=False, allow_nan=False) + '\n')
    write_json(output/'id-map.json', {q: positions[h] for q, h in entries})
    result['files'] = {name: sha256(output/name) for name in ('requests.jsonl', 'id-map.json')}
    write_json(output/'manifest.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    from scripts.resources import require_resources
    require_resources()
    result = build_plan(args.inputs, args.output_dir)
    print({k: result[k] for k in ('unique_texts', 'tokens', 'requests', 'estimated_standard_usd', 'paid_api_calls')})


if __name__ == '__main__':
    main()

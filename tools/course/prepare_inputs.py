"""Download pinned HF data serially and reconstruct course inputs in a new folder."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from scripts.fetch_data import fetch_one, verify
from tools.course.analyze import ROOT, sha256, write_json


def ensure_sources(folder, sources, *, downloader=fetch_one):
    """Validate cached source files first, then fetch missing sources one at a time."""
    folder = Path(folder)
    for key, spec in sources.items():
        target = folder/key/Path(spec['file']).name
        if target.exists():
            verify(target, spec)
    records = []
    for key, spec in sources.items():
        records.append(downloader(key, spec, folder))
        verify(folder/key/Path(spec['file']).name, spec)
    return records


def prepare(source_dir, output_dir):
    """Reconstruct normalized tables, grouped splits and preference/embedding inputs.

    Source files come from pinned public HF URLs or a checksum-valid cache. Output
    is new and self-contained. No model loading or provider request is performed.
    """
    from scripts.resources import require_resources
    from scripts.data_io import load_routerbench
    from scripts.data_gate import normalize_routerbench, make_splits
    from scripts.research_pipeline import prepare_data
    from scripts.input_audit import export_inputs
    preflight = require_resources()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    pins = json.loads((ROOT/'configs/data_sources.json').read_text(encoding='utf-8'))
    sources = {k:pins[k] for k in ('routerbench_0shot', 'arena_55k')}
    downloaded = ensure_sources(source_dir, sources)
    output_dir.mkdir(parents=True)
    data_dir = output_dir/'data'
    for key, spec in sources.items():
        target = data_dir/key/Path(spec['file']).name
        target.parent.mkdir(parents=True)
        shutil.copy2(Path(source_dir)/key/Path(spec['file']).name, target)
        verify(target, spec)
    frame = load_routerbench(data_dir/'routerbench_0shot/routerbench_0shot.pkl', sources['routerbench_0shot'])
    queries, outcomes = normalize_routerbench(frame)
    splits = make_splits(queries, seed=42)
    processed = data_dir/'processed/routerbench_0shot'
    processed.mkdir(parents=True)
    queries.to_parquet(processed/'queries.parquet', index=False)
    outcomes.to_parquet(processed/'outcomes.parquet', index=False)
    splits.to_csv(processed/'splits.csv', index=False, lineterminator='\n')
    expected = json.loads((ROOT/'reports/data-gate.json').read_text(encoding='utf-8'))['prepared']['file_sha256']
    hashes = {p.name:sha256(p) for p in sorted(processed.iterdir())}
    if any(hashes.get(k) != v for k,v in expected.items()):
        raise ValueError('Reconstructed tables differ from frozen hashes; inspect library versions, do not repin silently')
    prepared = prepare_data(data_dir, output_dir/'prepared', mapping_policy='identity_match', rows=128, seed=42)
    _, _, inputs = export_inputs(data_dir, output_dir/'prepared', output_dir/'inputs')
    report = dict(version=1, kind='course-hosted-input-reproduction', source_pins=sources,
        source_records=downloaded, processed_sha256=hashes, source_cache_verified=True,
        routing_queries=len(queries), outcomes=len(outcomes), preference_pairs=prepared['eligible_preferences'],
        input_counts=inputs['phase_counts'], preflight=preflight, synthetic=False,
        paid_api_calls=0, gpu_rerun=False, paper_equivalent=False)
    write_json(output_dir/'manifest.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(args.source_dir, args.output_dir)
    print(json.dumps({k:result[k] for k in ('routing_queries','outcomes','preference_pairs','gpu_rerun')}))


if __name__ == '__main__':
    main()

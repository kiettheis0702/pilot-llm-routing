import copy
from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.data_gate import query_hash
from scripts.splits import SplitSelection


def spec():
    value = json.loads((Path(__file__).parents[1] / 'configs/routerbench_costs.json').read_text())
    value['rates'] = {'a': {'input': 1., 'output': 2.}, 'b': {'input': 2., 'output': 4.}}
    return value


def fixture():
    ids = ['t1', 't2', 'l1', 'd1']
    segments = [['hello', 'world'], ['more'], ['learning'], ['deployment']]
    queries = pd.DataFrame(dict(query_id=ids, query=['\n\n'.join(x) for x in segments],
                                query_segments_json=[json.dumps(x) for x in segments], task='fixture'))
    queries['query_hash'] = queries['query'].map(query_hash)
    raw = pd.DataFrame(dict(sample_id=ids, prompt=[repr(x) for x in segments]))
    raw['a|model_response'] = [repr(['aa']), repr(['bbbb']), 'DO NOT READ', None]
    raw['b|model_response'] = [repr(['c']), repr(['ddd']), None, 'DO NOT READ']
    raw['a|total_cost'] = [14 / 1e6, 12 / 1e6, float('nan'), -100]
    raw['b|total_cost'] = [24 / 1e6, 20 / 1e6, float('inf'), -100]
    tuning = SplitSelection(('t1', 't2'), 'tuning', 'abc', 42, 2, 10, 2)
    return raw, queries, tuning


class Counter:
    metadata = {'implementation': 'character fixture', 'synthetic': True}
    def __init__(self): self.seen = []
    def __call__(self, text):
        assert text != 'DO NOT READ'
        self.seen.append(text)
        return len(text)


def test_price_units_preserve_declared_source_factor_and_provenance():
    from scripts.cost_inputs import prices_from_spec, triplet_cost_sizes
    prices = prices_from_spec(spec())
    assert prices['a'].input_per_million == 1
    assert prices['b'].output_per_million == 4
    sizes = triplet_cost_sizes(spec(), ['b', 'a'])
    assert sizes['sizes'] == {'b': 6., 'a': 3.}
    assert sizes['synthetic'] is False
    assert '1:1' in sizes['provenance'] and sizes['source_urls']


@pytest.mark.parametrize('field,value', [('calibration_provenance', ''), ('source_url', 'bad'),
    ('source_file_sha256', 'bad'), ('dataset_multiplier', -1), ('source_divisor', 0)])
def test_invalid_cost_provenance_or_scale_is_rejected(field, value):
    from scripts.cost_inputs import prices_from_spec
    bad = spec(); bad[field] = value
    with pytest.raises(ValueError): prices_from_spec(bad)


def test_cost_tables_use_separate_segments_and_tuning_outputs_only():
    from scripts.cost_inputs import build_tables
    raw, queries, tuning = fixture()
    counter = Counter()
    result = build_tables(raw, queries, tuning, spec(), counter)
    assert result['input_counts'] == {'t1':10, 't2':4, 'l1':8, 'd1':10}
    assert result['output_counts'] == {'t1':{'a':2,'b':1}, 't2':{'a':4,'b':3}}
    assert result['estimator']['mean_output_tokens'] == {'a':3., 'b':2.}
    assert result['calibration']['matched_pairs'] == 4
    assert result['calibration']['all_matched'] is True
    assert 'hello\n\nworld' not in counter.seen


def test_deployment_response_and_cost_mutation_cannot_change_estimates():
    from scripts.cost_inputs import build_tables
    raw, queries, tuning = fixture()
    first = build_tables(raw, queries, tuning, spec(), Counter())
    for column in raw.columns:
        if '|model_response' in column or '|total_cost' in column:
            raw[column] = raw[column].astype(object)
            raw.loc[raw.sample_id.isin(['l1', 'd1']), column] = 'MALFORMED HOLDOUT'
    assert build_tables(raw, queries, tuning, spec(), Counter()) == first


def test_wrong_scale_or_modified_tuning_cost_fails_instead_of_auto_calibration():
    from scripts.cost_inputs import build_tables
    raw, queries, tuning = fixture()
    bad = spec(); bad['dataset_multiplier'] = 1
    with pytest.raises(ValueError, match='cost calibration'):
        build_tables(raw, queries, tuning, bad, Counter())
    raw.loc[0, 'a|total_cost'] *= 2
    with pytest.raises(ValueError, match='cost calibration'):
        build_tables(raw, queries, tuning, spec(), Counter())


def test_counting_rejects_invalid_segments_and_tuning_ids():
    from scripts.cost_inputs import build_tables, count_segments
    for bad in (None, [None], [[1]], 'not a list'):
        with pytest.raises(ValueError): count_segments(bad, Counter())
    assert count_segments(['', 'abc'], Counter()) == 3
    raw, queries, tuning = fixture()
    with pytest.raises(ValueError, match='tuning'):
        build_tables(raw, queries, replace(tuning, split='deployment'), spec(), Counter())
    with pytest.raises(ValueError, match='coverage'):
        build_tables(raw.iloc[1:], queries, tuning, spec(), Counter())
    queries.loc[0, 'query_segments_json'] = '["changed"]'
    with pytest.raises(ValueError, match='prompt'):
        build_tables(raw, queries, tuning, spec(), Counter())


def test_local_counter_rejects_missing_or_corrupt_vocab_before_use(tmp_path):
    from scripts.cost_inputs import HistoricalCounter, VOCAB_CACHE_KEY
    with pytest.raises(FileNotFoundError): HistoricalCounter(tmp_path)
    (tmp_path / VOCAB_CACHE_KEY).write_text('corrupt')
    with pytest.raises(ValueError, match='vocabulary checksum'): HistoricalCounter(tmp_path)


def test_export_readback_rejects_changed_source_and_existing_artifact(tmp_path):
    from tests.test_research_pipeline import project, write_json
    from scripts.cost_inputs import export_cost_inputs, load_cost_inputs
    from scripts.fetch_data import sha256
    root, data = project(tmp_path)
    folder = data / 'processed/routerbench_0shot'
    q = pd.read_parquet(folder / 'queries.parquet')
    q['query_segments_json'] = q['query'].map(lambda x: json.dumps([x]))
    q.to_parquet(folder / 'queries.parquet', index=False)
    raw = pd.DataFrame({'sample_id': q.query_id, 'prompt': q['query'].map(lambda x:repr([x]))})
    for arm, (pi, po) in {'a':(1,2), 'b':(2,4)}.items():
        raw[arm+'|model_response'] = repr(['answer'])
        raw[arm+'|total_cost'] = q['query'].map(lambda x: (len(x)*pi + 6*po)/1e6)
    raw_path = data / 'routerbench_0shot/routerbench_0shot.pkl'
    raw_path.parent.mkdir()
    raw.to_pickle(raw_path)
    source = json.loads((root / 'configs/data_sources.json').read_text())
    source['routerbench_0shot'] = {'file':raw_path.name,'size':raw_path.stat().st_size,'sha256':sha256(raw_path)}
    write_json(root / 'configs/data_sources.json', source)
    write_json(root / 'reports/data-gate.json', {'prepared':{'file_sha256':{p.name:sha256(p) for p in folder.iterdir()}}})
    out = tmp_path / 'costs'
    result = export_cost_inputs(data, out, spec(), root=root, tuning_size=2,
                               counter=Counter(), allow_synthetic=True)
    loaded = load_cost_inputs(out)
    assert len(loaded['input-counts']) == 12
    assert len(loaded['tuning-output-counts']) == 2
    assert result['calibration']['matched_pairs'] == 4
    assert loaded['cost-size-identity_match']['sizes'] == {'a':3.,'b':6.}
    before = (out / 'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        export_cost_inputs(data, out, spec(), root=root, tuning_size=2, counter=Counter(), allow_synthetic=True)
    assert (out / 'manifest.json').read_bytes() == before
    with (out / 'input-counts.json').open('a') as stream: stream.write(' ')
    with pytest.raises(ValueError, match='checksum'): load_cost_inputs(out)
    with raw_path.open('ab') as stream: stream.write(b'tampered')
    with pytest.raises(ValueError, match='mismatch'):
        export_cost_inputs(data, tmp_path/'changed', spec(), root=root, tuning_size=2, counter=Counter(), allow_synthetic=True)

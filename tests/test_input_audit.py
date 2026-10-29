import json

import pytest

from scripts.data_gate import query_hash


def inputs(texts):
    return [dict(query_id=f'q{i}', text=t, prompt_hash=query_hash(t))
            for i, t in enumerate(texts)]


class CharacterCounter:
    """Analytic tokenizer fixture: one token per character plus BOS/EOS."""
    metadata = {'model': 'fixture', 'revision': '1', 'special_tokens': True}

    def __init__(self):
        self.batches = []

    def __call__(self, texts):
        self.batches.append(list(texts))
        return [len(t) + 2 for t in texts]


def test_length_audit_counts_boundaries_and_preserves_order(tmp_path):
    from scripts.input_audit import audit_inputs
    rows = inputs(['a' * 6, 'b' * 7, 'c'])
    phases = {r['query_id']: 'routing:learning' for r in rows}
    counter = CharacterCounter()
    result = audit_inputs(rows, phases, counter, tmp_path / 'audit',
                          sources={'fixture': 'test'}, caps=(8, 16), batch_size=2)
    assert counter.batches == [['aaaaaa', 'bbbbbbb'], ['c']]
    stats = result['by_phase']['routing:learning']
    assert stats['count'] == 3 and stats['max'] == 9
    assert stats['over_cap'] == {'8': 1, '16': 0}
    assert stats['truncated_tokens'] == {'8': 1, '16': 0}
    records = [json.loads(line) for line in (tmp_path / 'audit/lengths.jsonl').read_text().splitlines()]
    assert [r['query_id'] for r in records] == ['q0', 'q1', 'q2']
    assert [r['tokens'] for r in records] == [8, 9, 3]
    assert all('text' not in r for r in records)
    assert result['decision']['cap'] == 16


def test_holdout_and_deployment_outliers_cannot_change_recipe(tmp_path):
    from scripts.input_audit import audit_inputs
    rows = inputs(['aaaa', 'bbbb', 'c' * 500, 'd' * 1000])
    phases = dict(zip(['q0', 'q1', 'q2', 'q3'],
                      ['routing:tuning', 'preference:train', 'routing:deployment', 'preference:holdout']))
    result = audit_inputs(rows, phases, CharacterCounter(), tmp_path / 'audit',
                          sources={'fixture': 'test'}, caps=(8, 16))
    assert result['decision']['cap'] == 8
    assert result['decision']['selection_groups'] == {'routing': 1, 'preference': 1}
    assert result['by_phase']['preference:holdout']['over_cap']['16'] == 1


@pytest.mark.parametrize('error', ['hash', 'duplicate', 'missing-phase', 'unknown-phase'])
def test_invalid_identity_or_phase_fails_before_publishing(tmp_path, error):
    from scripts.input_audit import audit_inputs
    rows = inputs(['first', 'second'])
    phases = {'q0': 'routing:learning', 'q1': 'routing:deployment'}
    if error == 'hash': rows[0]['text'] = 'tampered'
    if error == 'duplicate': rows[1] = rows[0].copy()
    if error == 'missing-phase': phases.pop('q1')
    if error == 'unknown-phase': phases['q0'] = 'unknown'
    counter = CharacterCounter()
    with pytest.raises(ValueError):
        audit_inputs(rows, phases, counter, tmp_path / 'audit', sources={'fixture': 'test'})
    assert not (tmp_path / 'audit/manifest.json').exists()
    assert counter.batches == []


def test_verbatim_identity_and_immutable_completed_output(tmp_path):
    from scripts.input_audit import audit_inputs
    rows = inputs(['code\n    indented'])
    phases = {'q0': 'routing:learning'}
    first = audit_inputs(rows, phases, CharacterCounter(), tmp_path / 'first', sources={'fixture': 'test'})
    changed = inputs(['code indented'])
    assert rows[0]['prompt_hash'] == changed[0]['prompt_hash']
    second = audit_inputs(changed, phases, CharacterCounter(), tmp_path / 'second', sources={'fixture': 'test'})
    assert first['input_sha256'] != second['input_sha256']
    before = (tmp_path / 'first/manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        audit_inputs(rows, phases, CharacterCounter(), tmp_path / 'first', sources={'fixture': 'test'})
    assert (tmp_path / 'first/manifest.json').read_bytes() == before


def test_uncovered_selection_is_explicit_and_bad_counts_leave_no_manifest(tmp_path):
    from scripts.input_audit import audit_inputs
    rows = inputs(['a' * 30])
    phases = {'q0': 'routing:learning'}
    report = audit_inputs(rows, phases, CharacterCounter(), tmp_path / 'too-long',
                          sources={'fixture': 'test'}, caps=(8, 16))
    assert report['decision']['cap'] is None
    assert report['decision']['coverage_target_met'] is False
    class InvalidCounter(CharacterCounter):
        def __call__(self, texts):
            return [0]
    counter = InvalidCounter()
    with pytest.raises(ValueError, match='token'):
        audit_inputs(rows, phases, counter, tmp_path / 'bad', sources={'fixture': 'test'})
    assert not (tmp_path / 'bad/manifest.json').exists()


def test_full_input_export_uses_frozen_splits_and_verified_sources(tmp_path):
    from tests.test_research_pipeline import project
    from scripts.research_pipeline import prepare_data, load_routing
    from scripts.input_audit import export_inputs
    from scripts.fetch_data import sha256
    root, data = project(tmp_path)
    prepared = tmp_path / 'prepared'
    prepare_data(data, prepared, root=root, rows=4, tuning_size=2)
    split_path = data / 'processed/routerbench_0shot/splits.csv'
    original = sha256(split_path)
    rows, phases, manifest = export_inputs(data, prepared, tmp_path / 'full', root=root, tuning_size=2)
    _, _, splits, _ = load_routing(data, root, 2)
    route_ids = [r['query_id'] for r in rows if phases[r['query_id']].startswith('routing:')]
    expected = [q for phase in ('tuning', 'learning', 'deployment') for q in splits.select(phase).query_ids]
    assert route_ids == expected and len(route_ids) == 12
    assert len(rows) == 22 and sha256(split_path) == original
    assert manifest['source_hashes']['splits.csv'] == original
    assert len(json.loads((tmp_path / 'full/routing-inputs.json').read_text())) == 12
    with split_path.open('a') as stream: stream.write('\n')
    with pytest.raises(ValueError, match='checksum'):
        export_inputs(data, prepared, tmp_path / 'corrupt', root=root, tuning_size=2)


def test_partial_manifest_write_is_never_published(tmp_path, monkeypatch):
    import scripts.input_audit as audit
    def disk_full(path, value):
        path.write_text('{"partial":')
        raise OSError('disk full')
    monkeypatch.setattr(audit, 'write_json', disk_full)
    with pytest.raises(OSError, match='disk full'):
        audit.audit_inputs(inputs(['short']), {'q0': 'routing:learning'}, CharacterCounter(),
                           tmp_path / 'audit', sources={'fixture': 'test'})
    assert not (tmp_path / 'audit/manifest.json').exists()
    assert not list((tmp_path / 'audit').glob('*.tmp'))

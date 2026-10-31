"""A restored checkout must not depend on hidden operational preparation."""
import hashlib
import json
import pytest


def test_original_profile_can_bootstrap_default_adapter(tmp_path, monkeypatch):
    from tools.openai_study_reproduce import install_profile, PROFILE_SHA256
    from tools import openai_study_hybrid_default_batch as adapter
    path = install_profile(tmp_path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROFILE_SHA256
    monkeypatch.setattr(adapter, 'BATCH_PROFILE', path)
    scope = adapter.namespace('trainer')
    assert scope['SOURCE_REUSE']['hardware_profile_sha256'] == PROFILE_SHA256
    assert scope['SOURCE_REUSE']['trainer_changes']['train_microbatch'] == 8
    assert install_profile(tmp_path).read_bytes() == path.read_bytes()


def test_conflicting_profile_is_retained(tmp_path):
    from tools.openai_study_reproduce import install_profile, PROFILE_TARGET
    target = tmp_path / PROFILE_TARGET
    target.parent.mkdir(parents=True)
    target.write_bytes(b'existing evidence')
    with pytest.raises(FileExistsError):
        install_profile(tmp_path)
    assert target.read_bytes() == b'existing evidence'


def test_corrupt_packaged_profile_is_rejected_before_write(tmp_path):
    from tools.openai_study_reproduce import install_profile, PROFILE_TARGET
    source = tmp_path / 'bad.json'
    source.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='checksum'):
        install_profile(tmp_path, source=source)
    assert not (tmp_path / PROFILE_TARGET).exists()


def test_all_destinations_checked_before_first_publication(tmp_path):
    from tools.openai_study_reproduce import publish
    existing = tmp_path / 'outputs/existing.json'
    existing.parent.mkdir()
    existing.write_bytes(b'old')
    with pytest.raises(FileExistsError):
        publish(tmp_path, {'outputs/new.json': b'new', 'outputs/existing.json': b'different'})
    assert not (tmp_path / 'outputs/new.json').exists()
    assert existing.read_bytes() == b'old'


def test_missing_core_inputs_fail_before_installing_profile(tmp_path):
    from tools.openai_study_reproduce import prepare, PROFILE_TARGET
    with pytest.raises(ValueError, match='checksum'):
        prepare(tmp_path)
    assert not (tmp_path / PROFILE_TARGET).exists()


def candidate(name, correct, epochs):
    return dict(id=name, dimension=64, projection_epochs=epochs, prior_epochs=epochs,
                calibration=dict(correct=correct, pairs=211, accuracy=correct/211))


def test_calibration_tie_selects_fewer_epochs_and_ignores_test():
    from tools.openai_study_reproduce import selected_recipe
    manifest = dict(seed=42, candidates=[candidate('long',144,60),
        candidate('current',146,30), candidate('short',146,15)], selected_candidate='short')
    expected = dict(id='short', dimension=64, projection_epochs=15, prior_epochs=15)
    assert selected_recipe(manifest) == expected
    manifest['final_test_accuracy'] = 0
    assert selected_recipe(manifest) == expected
    manifest['final_test_accuracy'] = 1
    assert selected_recipe(manifest) == expected


@pytest.mark.parametrize('change', ['seed', 'selection'])
def test_replicates_reject_wrong_selection_lineage(change):
    from tools.openai_study_reproduce import selected_recipe
    manifest = dict(seed=42, candidates=[candidate('short',146,15),candidate('long',144,60)],
                    selected_candidate='short')
    if change == 'seed':
        manifest['seed'] = 7
    else:
        manifest['selected_candidate'] = 'long'
    with pytest.raises(ValueError):
        selected_recipe(manifest)


def preference_fixture(tmp_path, monkeypatch):
    from tools import openai_study_reproduce as module
    from tools.paper_preflight import train
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'FOLLOWUP', tmp_path / 'outputs/openai-followup-20261031')
    monkeypatch.setattr(module, 'CHECKS', tmp_path / 'outputs/checks')
    def refuse_fit(*args, **kwargs):
        pytest.fail('Invalid prerequisites must be rejected before fitting')
    monkeypatch.setattr(train, 'run_training', refuse_fit)
    pair_path = tmp_path / module.SPLIT_TARGET / 'preference-three-way.json'
    pair_path.parent.mkdir(parents=True)
    pair_path.write_bytes(b'[]')
    monkeypatch.setattr(module, 'SPLIT_SHA256', hashlib.sha256(b'[]').hexdigest())
    files = {
        'outputs/research-cost-inputs-20261029/cost-size-identity_match.json': b'{"sizes":{"a":1,"b":2}}',
        'outputs/openai-standard-20261030/imported/caches/source-1/manifest.json': b'{}',
        'outputs/openai-standard-20261030/imported/caches/source-1/vectors.npy': b'fixture-vectors',
    }
    entries = []
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        entries.append(dict(path=name, size=len(content), sha256=hashlib.sha256(content).hexdigest()))
    configs = tmp_path / 'configs'
    configs.mkdir()
    (configs / 'openai-reproduction-files.json').write_text(json.dumps(dict(
        version=1, kind='openai-offline-reproduction-inputs', files=entries)), encoding='utf-8')
    (configs / 'paper_preflight_protocol.json').write_text(json.dumps(dict(pretraining=dict(
        candidates=[dict(id='short', dimension=64, projection_epochs=15, prior_epochs=20)],
        learning_rate=.03, margin=.2))), encoding='utf-8')
    return module, list(files)


@pytest.mark.parametrize('index', [0, 1, 2])
def test_changed_pinned_preference_input_rejected_before_fit(tmp_path, monkeypatch, index):
    module, names = preference_fixture(tmp_path, monkeypatch)
    changed = tmp_path / names[index]
    if index == 0:
        changed.write_bytes(b'{"sizes":{"a":2,"b":4}}')
    else:
        changed.write_bytes(b'changed-validity-is-not-identity')
    with pytest.raises(ValueError, match='checksum'):
        module.preference(42)
    assert not (module.FOLLOWUP / 'preference500').exists()


def test_missing_published_report_rejected_before_fit(tmp_path, monkeypatch):
    module, _ = preference_fixture(tmp_path, monkeypatch)
    with pytest.raises((FileNotFoundError, ValueError)):
        module.preference(42)
    assert not (module.FOLLOWUP / 'preference500').exists()


def test_summary_is_idempotent_and_preserves_conflicting_evidence(tmp_path, monkeypatch):
    from tools import openai_study_reproduce as module
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    path = module.record_ci_summary(dict(verified=True, commit='original'))
    original = path.read_bytes()
    assert module.record_ci_summary(dict(verified=True, commit='original')) == path
    with pytest.raises(ValueError, match='Immutable'):
        module.record_ci_summary(dict(verified=True, commit='different'))
    assert path.read_bytes() == original

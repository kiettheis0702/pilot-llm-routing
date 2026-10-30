import copy
import json
import hashlib
import shutil
from pathlib import Path

import pytest

from tools.course.tracking import build_payloads, record_run, validate_payload

PACKAGE = Path('reports/full-experiments-2026-10-29')
ANALYSIS = Path('reports/course-submission-2026-10-30/analysis')


def test_payloads_are_deterministic_complete_and_only_allow_public_metadata():
    a = build_payloads(PACKAGE, ANALYSIS)
    assert a == build_payloads(PACKAGE, ANALYSIS)
    assert len(a) == 1128
    assert len({r['id'] for r in a}) == len(a)
    assert sum(r['config']['experiment_family'] == 'historical-deployment' for r in a) == 1098
    assert sum(r['config']['experiment_family'] == 'synthetic-magnitude' for r in a) == 18
    assert sum(r['config']['experiment_family'] == 'cost-accounting' for r in a) == 12
    for p in a:
        validate_payload(p)
    text = json.dumps(a)
    for private in ('C:\\Users', 'api_key', 'query_text', 'prompt_text', '@gmail'):
        assert private not in text


def test_modified_metrics_and_added_private_field_fail():
    p = build_payloads(PACKAGE, ANALYSIS)[0]
    changed = copy.deepcopy(p)
    changed['metrics']['coverage'] = 123
    with pytest.raises(ValueError): validate_payload(changed)
    changed = copy.deepcopy(p)
    changed['config']['prompt_text'] = 'private'
    with pytest.raises(ValueError): validate_payload(changed)


def test_online_requires_explicit_approval_and_destination_before_sdk(tmp_path):
    p = build_payloads(PACKAGE, ANALYSIS)[0]
    with pytest.raises(ValueError, match='approval'):
        record_run(p, tmp_path, mode='online', entity='team', approved=False)
    with pytest.raises(ValueError, match='entity'):
        record_run(p, tmp_path, mode='online', approved=True)
    assert not list(tmp_path.iterdir())


def test_offline_default_and_finished_receipt_do_not_reinitialize(tmp_path):
    p = build_payloads(PACKAGE, ANALYSIS)[0]
    calls = []
    class Run:
        id = p['id']
        url = None
        def log(self, data): calls.append(('log', data))
        def finish(self): calls.append(('finish', None))
    class SDK:
        def Settings(self, **kwargs): return kwargs
        def Table(self, **kwargs): return kwargs
        class plot:
            @staticmethod
            def bar(table, *args, **kwargs): return dict(table=table, **kwargs)
        def init(self, **kwargs):
            calls.append(('init', kwargs))
            return Run()
    first = record_run(p, tmp_path, sdk=SDK())
    second = record_run(p, tmp_path, sdk=SDK())
    assert first == second
    assert first['status'] == 'finished' and first['url'] is None
    init = [value for kind,value in calls if kind == 'init']
    assert len(init) == 1 and init[0]['mode'] == 'offline'
    # SDK media names add ~130 characters. Do not repeat long destination names
    # in its working directory: ordinary Windows defaults to MAX_PATH=260.
    sdk_dir = Path(init[0]['dir'])
    assert sdk_dir.is_relative_to(tmp_path)
    assert 'pilot-llm-routing-team8' not in sdk_dir.relative_to(tmp_path).parts
    assert len(str(sdk_dir.relative_to(tmp_path))) <= 16
    assert init[0]['settings']['disable_git']
    assert init[0]['settings']['x_disable_meta']
    assert init[0]['settings']['x_disable_stats']
    logged = [value for kind,value in calls if kind == 'log']
    assert 'quality_summary' in logged[0]
    assert logged[0]['quality_summary']['table']['data'] == [
        ['service_utility',p['metrics']['service_utility']],
        ['coverage',p['metrics']['coverage']],
        ['mean_reward_served',p['metrics']['mean_reward_served']]]


def amended_analysis(tmp_path, changes):
    target=tmp_path/'analysis'
    shutil.copytree(ANALYSIS,target)
    file=target/'analysis.json'
    value=json.loads(file.read_text())
    value.update(changes)
    file.write_text(json.dumps(value))
    manifest=target/'package-files.json'
    inventory=json.loads(manifest.read_text())
    inventory['files']['analysis.json']=hashlib.sha256(file.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(inventory))
    return target


def test_supplement_date_comes_from_verified_execution_timestamp(tmp_path):
    folder=amended_analysis(tmp_path, {'created_utc':'2027-01-02T03:04:05+00:00'})
    records=build_payloads(PACKAGE,folder)
    extras=[r for r in records if r['config']['experiment_family']!='historical-deployment']
    assert len(extras)==30
    assert {r['config']['source_execution_date'] for r in extras}=={'2027-01-02'}


def test_naive_execution_timestamp_is_rejected(tmp_path):
    folder=amended_analysis(tmp_path, {'created_utc':'2027-01-02T03:04:05'})
    with pytest.raises(ValueError,match='timezone'):build_payloads(PACKAGE,folder)


def test_verified_but_mismatched_evidence_packages_are_rejected(tmp_path):
    folder=amended_analysis(tmp_path, {'source_csv_sha256':'0'*64})
    with pytest.raises(ValueError,match='lineage'):build_payloads(PACKAGE,folder)

"""Public-safe namespaced follow-up records; no online action on import."""
from __future__ import annotations

import hashlib
import argparse
import csv
import json
import math
from pathlib import Path
import re

RECIPES={'openai','instructor','deberta'}
FAMILIES={'core','ablations','budget','preference500','routellm-mf','noise','binary',
          'cost-shock','hybrid-native','hybrid-fixed','task-shift'}
CONFIG_KEYS={'recipe','experiment_family','source_run_id','source_manifest_sha256',
    'source_csv_sha256','source_execution_date','paper_equivalent','synthetic','historical_import',
    'domain','seed','method','variant','learning_fraction','budget_fraction','budget',
    'learning_count','dimension','arm_count','policy','epsilon','factor','population',
    'threshold','window','selected_epoch','projection_dimension','input_dimension',
    'counterfactual','future_queries_used','description','choices_unchanged'}
METRIC_KEYS={'requested','served','total_reward','mean_reward_served','service_utility',
    'coverage','estimated_cost','actual_recorded_cost','actual_budget_excess','estimated_budget_excess',
    'learning_rounds','quality_fraction_gpt4','cost_fraction_gpt4','gpt4_total_reward','gpt4_total_cost',
    'allocated_budget_fraction_gpt4','unused_estimated_budget','empirical_mrr_requested',
    'noise_replaced','noise_changed','learning_true_reward','learning_true_oracle_regret',
    'average_reward_delta_vs_zero','correct','pairs','accuracy','accuracy_delta_pp',
    'conditions','over_budget_count','maximum_excess','total_excess','cheap_count','strong_count'}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_record(record):
    if set(record)!={'id','payload_sha256','config','metrics','lab'}:
        raise ValueError('Unknown follow-up tracking envelope fields')
    config=record['config'];metrics=record['metrics']
    if (not isinstance(config,dict) or not isinstance(metrics,dict)
            or set(config)-CONFIG_KEYS or set(metrics)-METRIC_KEYS or not metrics):
        raise ValueError('Unknown/private tracking config or metric field')
    if config.get('recipe') not in RECIPES or config.get('experiment_family') not in FAMILIES:
        raise ValueError('Unknown representation recipe/experiment family')
    if (not re.fullmatch(r'[0-9a-f]{64}',config.get('source_manifest_sha256',''))
            or not re.fullmatch(r'[A-Za-z0-9_./:+-]{1,300}',config.get('source_run_id',''))):
        raise ValueError('Valid immutable public source identity required')
    if any(type(value) not in (str,bool,int,float) or (type(value) is float and not math.isfinite(value)) for value in config.values()):
        raise ValueError('Only finite scalar public config values allowed')
    if any(type(value) not in (int,float) or not math.isfinite(value) for value in metrics.values()):
        raise ValueError('Only finite numeric public metrics allowed')
    if config.get('paper_equivalent') is not False or config.get('synthetic') is not False or config.get('historical_import') is not True:
        raise ValueError('Follow-up records must disclose local historical-data evaluation')
    if record['lab'] not in ('lab1-reproduction','lab2-critique'):
        raise ValueError('Unknown course experiment grouping')
    encoded=json.dumps(dict(config=config,metrics=metrics,lab=record['lab'])).lower()
    if any(marker in encoded for marker in ('api_key','authorization','@gmail','c:\\\\users','c:/users','/home/','/users/')):
        raise ValueError('Private account/path content in tracking record')
    body={key:record[key] for key in ('config','metrics','lab')};fingerprint=digest(body)
    if record['payload_sha256']!=fingerprint or record['id']!='p2-'+fingerprint[:24]:
        raise ValueError('Stale follow-up tracking digest/identity')
    return record


def make_record(recipe,family,source_id,config,metrics,source_manifest_sha256):
    reserved={'recipe','experiment_family','source_run_id','source_manifest_sha256',
              'paper_equivalent','synthetic','historical_import'}
    if set(config)&reserved:
        raise ValueError('Caller cannot replace tracking provenance fields')
    public=dict(config,recipe=recipe,experiment_family=family,source_run_id=source_id,
        source_manifest_sha256=source_manifest_sha256,paper_equivalent=False,synthetic=False,historical_import=True)
    lab='lab1-reproduction' if family in ('core','binary','preference500','routellm-mf','hybrid-native','hybrid-fixed') else 'lab2-critique'
    body=dict(config=public,metrics=dict(metrics),lab=lab);fingerprint=digest(body)
    return validate_record(dict(body,id='p2-'+fingerprint[:24],payload_sha256=fingerprint))


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def scalar(value):
    if value in ('True','False'):
        return value=='True'
    if re.fullmatch(r'-?\d+',value):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def build_records(root):
    """Deterministically rebuild aggregate records from complete public reports.

    A source_manifest_sha256 identifies the WHOLE frozen report package. The
    exact CSV has its own config digest. No raw run/weight/cache paths are needed.
    All required research reports must exist; preparation never silently drops
    the unfinished baseline or drift work.
    """
    from tools.course.analyze import verify_package
    root=Path(root).resolve()
    specs=[
        ('openai','core','openai-core-2026-10-31','per-seed.csv',858,{'arm_count':8}),
        ('openai','ablations','openai-ablations-2026-10-31','per-seed.csv',240,{'arm_count':8}),
        ('openai','budget','openai-budget-2026-10-31','per-seed.csv',240,{'method':'pilot','arm_count':8}),
        ('openai','preference500','openai-preference500-2026-10-31','per-seed.csv',3,{'domain':'human-preference','method':'pilot-prior','dimension':64}),
        ('openai','routellm-mf','openai-routellm-2026-10-31','paired-preference500.csv',3,{'domain':'human-preference','method':'routellm-mf','dimension':128}),
        ('openai','noise','openai-noise-2026-10-31','per-seed.csv',12,{'domain':'mmlu','method':'pilot','arm_count':8}),
        ('openai','binary','openai-binary-2026-10-31','per-seed.csv',231,{'arm_count':2}),
        ('openai','cost-shock','openai-cost-sensitivity-2026-10-31','cost-sensitivity.csv',18,{'arm_count':8}),
        ('instructor','core','instructor-core-2026-10-31','per-seed.csv',858,{'arm_count':8}),
        ('deberta','hybrid-native','openai-hybrid-2026-10-31','native-threshold-front.csv',21,{'domain':'routerbench','method':'hybrid-native','seed':42,'arm_count':2}),
        ('deberta','hybrid-fixed','openai-hybrid-2026-10-31','fixed-models.csv',2,{'domain':'routerbench','seed':42,'arm_count':2}),
        ('openai','task-shift','openai-task-shift-2026-10-31','window-metrics.csv',12,{'domain':'mmlu-to-gsm8k','seed':42,'arm_count':2}),
    ]
    folders=sorted({item[2] for item in specs})
    for name in folders:
        if not (root/'reports'/name/'package-files.json').is_file():
            raise ValueError('Required verified report is incomplete: '+name)
    packages={};public_files=[]
    for name in folders:
        folder=root/'reports'/name;verified=verify_package(folder)
        integrity=json.loads((folder/'package-files.json').read_text(encoding='utf-8'))
        packages[name]=dict(package_manifest_sha256=file_sha(folder/'package-files.json'),verified_files=verified['file_count'])
        # Select only checksum-listed scientific tables/figures/text. Operational
        # account metadata, models, embeddings and unrelated added files cannot
        # enter this list merely by sharing a directory.
        for relative in sorted(integrity['files']):
            path=folder/relative
            if not path.resolve().is_relative_to(folder.resolve()):
                raise ValueError('Public package member escapes its report directory')
            if path.suffix.lower() in ('.csv','.png','.svg','.md'):
                public_files.append(dict(path=path.relative_to(root).as_posix(),sha256=file_sha(path),bytes=path.stat().st_size))
    records=[];counts={}
    row_config_keys=(CONFIG_KEYS-{'recipe','experiment_family','source_run_id','source_manifest_sha256','source_csv_sha256',
        'paper_equivalent','synthetic','historical_import'})-METRIC_KEYS
    for recipe,family,name,filename,expected,extra in specs:
        path=root/'reports'/name/filename
        with path.open(encoding='utf-8',newline='') as stream:
            rows=list(csv.DictReader(stream))
        if family=='budget':
            rows=[r for r in rows if r['source']=='new-frozen-score-simulation']
        if len(rows)!=expected:
            raise ValueError(f'Public record inventory differs for {recipe}/{family}:{len(rows)} vs{expected}')
        csv_fingerprint=file_sha(path)
        for row in rows:
            config={key:scalar(row[key]) for key in sorted(row_config_keys) if key in row and row[key]!=''}
            config.update(extra);config['source_csv_sha256']=csv_fingerprint
            metrics={key:scalar(value) for key,value in row.items() if key in METRIC_KEYS and value!=''}
            source_id=row.get('id')
            if family=='preference500':
                source_id='seed-'+row['seed']
            elif family=='routellm-mf':
                source_id='seed-'+row['seed'];metrics={key:scalar(row[column]) for key,column in
                    [('correct','correct_routellm'),('pairs','pairs_routellm'),('accuracy','accuracy_routellm'),('accuracy_delta_pp','accuracy_delta_pp')]}
            elif family=='cost-shock':
                source_id=row['population']+'/factor-'+row['factor']
            elif family=='hybrid-native':
                source_id='threshold-'+row['threshold']
            elif family=='hybrid-fixed':
                source_id='fixed-'+row['arm'];config['method']=row['arm']
            elif family=='task-shift':
                source_id=row['method']+'/'+row['window']
                recipe='openai' if row['method']=='pilot' else 'deberta'
            if not source_id:
                raise ValueError('Public condition has no stable source identity')
            records.append(make_record(recipe,family,source_id,config,metrics,packages[name]['package_manifest_sha256']))
        counts[name+'/'+family]=len(rows)
    identities=[row['id'] for row in records]
    if len(identities)!=2498 or len(set(identities))!=len(identities):
        raise ValueError('Follow-up total/namespace collision differs from2498 declared summary views')
    return dict(version=1,kind='verified-public-followup-summary-views',runs=records,counts=counts,
        packages=packages,public_files=public_files,historical_legacy_runs_preserved=1128,
        interpretation='conditions/summary views, not live LLM calls or independent datasets; repeated controls/windows retain their scope',
        publication_scope='existing team8 GitHub/W&B project; no billing/visibility changes',additional_paid_model_api_calls=0)


def prepare(output):
    from scripts.resources import require_resources
    require_resources();root=Path(__file__).resolve().parents[1];body=build_records(root)
    payload=dict(body,payload_sha256=digest(body));encoded=json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n'
    public_bytes=sum(row['bytes'] for row in body['public_files'])+len(encoded.encode('utf-8'))
    if public_bytes>100*1024**2:
        raise ValueError('Public payload exceeds the reviewed100MiB raw-file limit')
    output=Path(output)
    if output.exists():
        if json.loads(output.read_text(encoding='utf-8'))!=payload:
            raise ValueError('Existing publication payload differs; retain and inspect')
    else:
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x',encoding='utf-8',newline='\n') as stream:
            stream.write(encoded)
    print(json.dumps(dict(runs=len(body['runs']),public_files=len(body['public_files']),raw_public_bytes=public_bytes,payload_sha256=payload['payload_sha256'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/openai-followup-20261031/publication/tracking-payloads.json'))
    prepare(parser.parse_args().output)


if __name__=='__main__':
    main()

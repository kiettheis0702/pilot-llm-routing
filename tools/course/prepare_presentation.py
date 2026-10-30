"""Derive course slide/chart data and run mapping from checksum-verified evidence."""
from __future__ import annotations

import csv
import argparse
import json
from pathlib import Path
from statistics import mean

from tools.course.analyze import ROOT, read_rows, sha256, verify_package
from tools.course.tracking import build_payloads
from tools.course.publication import publication_state

REPORT = ROOT/'reports/course-submission-2026-10-30'
PACKAGE = ROOT/'reports/full-experiments-2026-10-29'


def build_content():
    """Return deterministic chart data; reject missing or duplicated seed cells."""
    rows = read_rows(PACKAGE)
    verify_package(REPORT/'analysis')
    extra = json.loads((REPORT/'analysis/analysis.json').read_text(encoding='utf-8'))
    methods = ['pilot', 'linucb', 'exploreonly', 'epochgreedy', 'random']
    # Source identifiers are taken from the actual table, never guessed aliases.
    available = sorted({r['method'] for r in rows})
    methods = ['pilot', 'linucb', next(m for m in available if m.replace('-', '') == 'exploreonly'),
               next(m for m in available if m.replace('-', '') == 'epochgreedy'), 'random']
    names = ['PILOT', 'LinUCB', 'ExploreOnly', 'EpochGreedy', 'Random']

    def point(domain, method, fraction, budget, metric, variant='main'):
        group = [r for r in rows if r['domain']==domain and r['method']==method
                 and r['learning_fraction']==fraction and r['budget_fraction']==budget
                 and r['variant']==variant]
        if len(group)!=3 or {r['seed'] for r in group}!={7.,42.,2026.}:
            raise ValueError('Incomplete chart point: '+str((domain,method,fraction,budget,variant)))
        values = [r[metric] for r in group]
        return dict(mean=mean(values), minimum=min(values), maximum=max(values), source_ids=[r['id'] for r in group])

    charts = {}
    budgets = [.05,.1,.2,.3,.5,.75,1.,1.25]
    fractions = [.1,.25,.5,1.]
    for domain in ('routerbench','mmlu'):
        charts[domain+'-budget'] = dict(x=budgets, x_label='Local reference budget fraction', y_label='Service utility',
            series=[dict(name=n, points=[point(domain,m,1.,b,'service_utility') for b in budgets]) for m,n in zip(methods,names)])
        charts[domain+'-learning'] = dict(x=fractions, x_label='Learning stream fraction', y_label='Service utility at half budget',
            series=[dict(name=n, points=[point(domain,m,f,.5,'service_utility') for f in fractions]) for m,n in zip(methods[:3],names[:3])])
        charts[domain+'-regret'] = dict(x=fractions, x_label='Learning stream fraction', y_label='Cumulative empirical oracle regret',
            series=[dict(name=n, points=[point(domain,m,f,.5,'learning_empirical_oracle_regret') for f in fractions]) for m,n in zip(methods[:3],names[:3])])
    stress = next(c for c in extra['stress'] if c['rounds']==2000 and c['alpha']==1)
    checkpoints = [1,100,250,500,1000,1500,2000]
    charts['magnitude'] = dict(x=checkpoints,x_label='Learning rounds (synthetic)',y_label='Cumulative regret',
        series=[dict(name=p['method'],points=[dict(mean=p['trace'][n-1]['regret']) for n in checkpoints]) for p in stress['policies']])
    costs = extra['cost_sensitivity']['main PILOT, full learning']
    charts['cost'] = dict(x=[c['factor'] for c in costs],x_label='Multiplier on recorded costs',y_label='Conditions over budget (%)',
        series=[dict(name='Main PILOT; 48 conditions',points=[dict(mean=100*c['over_budget_count']/c['conditions']) for c in costs])])
    variants = ['main','zero-prior-direction','unit-initialization-strength','prior-only','single-budget-bin','projection-dimension-8']
    ablations = [[v]+[f"{point(d,'pilot',1.,.5,'service_utility',v)['mean']:.4f}" for d in ('routerbench','mmlu')] for v in variants]
    return dict(version=1, source_csv_sha256=sha256(PACKAGE/'aggregate/per-seed.csv'),
        analysis_sha256=sha256(REPORT/'analysis/analysis.json'), charts=charts,
        ablations=ablations, comparison=extra['comparison'], public_wandb_urls_pending=True)


def main(argv=None):
    """Write rebuildable presentation inputs and a public-safe experiment map."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=REPORT)
    parser.add_argument('--public-links',type=Path)
    args=parser.parse_args(argv)
    content = build_content()
    records = build_payloads(PACKAGE, REPORT/'analysis')
    file=args.public_links
    if file is None and (REPORT/'public-run-links.json').exists():file=REPORT/'public-run-links.json'
    published=publication_state(records,file)
    content['publication']=published
    content['public_wandb_urls_pending']=not published['complete']
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'slide-content.json').write_text(json.dumps(content,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    with (args.output_dir/'experiment-map.csv').open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['lab','experiment','script','source_run_id','wandb_run_id','wandb_url','status','source_sha256'])
        writer.writeheader()
        for record in records:
            c=record['config']
            writer.writerow(dict(lab=record['lab'],experiment=c['experiment_family'],
                script='scripts/experiments.py' if c['experiment_family']=='historical-deployment' else 'tools/course/analyze.py',
                source_run_id=c['source_run_id'],wandb_run_id=record['id'],wandb_url=published['urls'].get(record['id'],''),
                status='verified public' if record['id'] in published['urls'] else 'prepared; account and public upload pending',source_sha256=c['source_manifest_sha256']))
    print(json.dumps(dict(charts=len(content['charts']),mapped_runs=len(records))))


if __name__=='__main__':
    main()

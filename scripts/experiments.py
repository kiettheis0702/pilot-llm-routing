"""Sequential, immutable execution of the frozen local research protocol."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from scripts.bandit_contracts import ContextTable
from scripts.cost_policy import _selection_ids
from scripts.data_gate import OfflineBandit
from scripts.deployment import run_deployment
from scripts.deployment_adapter import FrozenDeploymentRouter, policy_state
from scripts.fetch_data import sha256
from scripts.input_audit import _publish_json
from scripts.pilot import PILOT
from scripts.pretraining import _digest
from scripts.tuning import build_policy


def fraction_key(value):
    return format(value, '.12g')


def expected_runs(protocol):
    """Enumerate the declaration, including every unfavorable/missing condition."""
    plan=protocol['primary_plan'];ablations=protocol['ablation_plan'];rows=[]
    methods=[m for m in plan['methods'] if m!='each-fixed-arm']
    methods += [f'fixed-{i:02d}' for i in range(len(protocol['arms']))]
    for domain in plan['datasets']:
        for seed in plan['replicate_seeds']:
            for method in methods:
                for fraction in plan['learning_fractions']:
                    budgets=(plan['budget_fractions'] if fraction==1 else [plan['learning_size_budget_fraction']])
                    for budget in budgets:
                        rows.append(dict(family='primary',domain=domain,seed=seed,method=method,
                            variant='main',learning_fraction=fraction,budget_fraction=budget))
            for variant in ablations['variants']:
                for budget in ablations['budget_fractions']:
                    rows.append(dict(family='ablation',domain=domain,seed=seed,method='pilot',
                        variant=variant['id'],learning_fraction=1.,budget_fraction=budget))
    for row in rows:
        row['learning_count']=math.ceil(plan['frozen_counts'][row['domain']]['learning']*row['learning_fraction'])
        row['dimension']=8 if row['variant']=='projection-dimension-8' else plan['pretraining']['dimension']
        row['id']='/'.join((row['domain'],f"seed-{row['seed']}",row['variant'],row['method'],
            'learn-'+fraction_key(row['learning_fraction']),'budget-'+fraction_key(row['budget_fraction'])))
    if len(rows)!=ablations['inventory']['primary_plus_ablation_deployment_runs']:
        raise ValueError('Declared inventory count disagrees with conditions')
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate experiment condition')
    return rows


def _json(value):
    return json.loads(json.dumps(value,ensure_ascii=False,allow_nan=False))


def load_artifact(folder, *, request=None):
    folder=Path(folder)
    if not (folder/'manifest.json').is_file():raise ValueError('Incomplete artifact: '+str(folder))
    result=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if result.get('manifest_sha256')!=_digest({k:v for k,v in result.items() if k!='manifest_sha256'}):
        raise ValueError('Artifact manifest checksum mismatch')
    if request is not None and result['request']!=_json(request):raise ValueError('Artifact request identity mismatch')
    names=result.get('files',{})
    if not names:raise ValueError('Empty artifact inventory')
    for name,fingerprint in names.items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts or relative.name=='manifest.json':
            raise ValueError('Unsafe artifact path')
        file=folder/relative
        if file.is_symlink() or not file.is_file() or sha256(file)!=fingerprint:
            raise ValueError('Artifact checksum mismatch: '+name)
    return result


def _resume(folder,request,resume):
    if not folder.exists():return None
    if not resume:raise FileExistsError(folder)
    return load_artifact(folder,request=request)


def _finish(folder,result,names):
    result=_json(result)
    result['files']={name:sha256(folder/name) for name in names}
    result['manifest_sha256']=_digest(result)
    _publish_json(folder/'manifest.json',result)
    return result


def variant_policy(prior,variant,parameters):
    if variant not in ('zero-prior-direction','unit-initialization-strength','prior-only',
                        'single-budget-bin','projection-dimension-8'):
        raise ValueError('Unknown ablation variant')
    model=build_policy(prior,'pilot',parameters)
    if variant not in ('zero-prior-direction','unit-initialization-strength'):return model
    return PILOT(model.arms,model.dimension,alpha=model.alpha,
        priors={a:np.zeros(model.dimension) if variant=='zero-prior-direction' else model._priors[i]
                for i,a in enumerate(model.arms)},
        lambdas={a:1. if variant=='unit-initialization-strength' else model._lambdas[i]
                 for i,a in enumerate(model.arms)},provenance=model.provenance+'; '+variant)


def learn_stream(*,model,queries,outcomes,contexts,selection,seed,fractions,lineage,
                 output_dir,training_mode='full',resume=False):
    """Save nested snapshots from one pass; no reset or reordering at prefixes."""
    ids=_selection_ids(selection,'learning');output=Path(output_dir)
    if training_mode not in ('full','none'):raise ValueError('Unknown training mode')
    if (not isinstance(fractions,list) or not fractions or fractions[-1]!=1
            or any(isinstance(f,bool) or not isinstance(f,(int,float)) or not 0<f<=1 for f in fractions)
            or sorted(set(fractions))!=fractions or (training_mode=='none' and fractions!=[1.])):
        raise ValueError('Invalid nested learning fractions')
    model.reset(seed)
    counts={fraction_key(f):math.ceil(len(ids)*f) for f in fractions}
    q=queries.loc[queries.query_id.isin(ids),['query_id','query','task']].copy()
    if set(q.query_id)!=set(ids) or q.query_id.duplicated().any():raise ValueError('Missing/duplicate learning query')
    q=q.set_index('query_id').loc[list(ids)].reset_index()
    try:ctx=ContextTable({i:contexts.get(i) for i in ids},label=contexts.label)
    except KeyError as exc:raise ValueError('Missing learning context') from exc
    if ctx.dimension!=model.dimension or any(not np.any(ctx.get(i)) for i in ids):raise ValueError('Invalid learning context')
    selected=None;environment=None
    if training_mode=='full':
        selected=outcomes.loc[outcomes.query_id.isin(ids)&outcomes.arm_id.isin(model.arms),
                              ['query_id','arm_id','reward','actual_cost']].copy()
        environment=OfflineBandit(q,selected,arm_catalog=model.arms)
    request=_json(dict(kind='learning-stream',version=1,seed=seed,training_mode=training_mode,
        selection=asdict(selection),fractions=fractions,initial_state=policy_state(model),lineage=lineage,
        context_sha256=ctx.fingerprint(ids),queries_sha256=_digest(q.to_dict('records')),
        outcomes_sha256=None if selected is None else _digest(selected.sort_values(['query_id','arm_id']).to_dict('records'))))
    existing=_resume(output,request,resume)
    if existing is not None:return existing
    oracle={} if selected is None else selected.groupby('query_id').reward.max().to_dict()
    output.mkdir(parents=True,exist_ok=False)
    snapshots={};names=['trace.jsonl'];reward=cost=regret=0.;actions=dict.fromkeys(model.arms,0)
    def metrics():
        return dict(rounds=model.rounds,total_reward=reward,total_actual_cost=cost,
                    cumulative_empirical_oracle_regret=regret,action_counts=actions.copy())
    def snapshot(keys):
        name=f'prefix-{model.rounds:09d}.json'
        FrozenDeploymentRouter(model,seed=seed).save_checkpoint(output/name);names.append(name)
        for key in keys:snapshots[key]=dict(file=name,rounds=model.rounds,metrics=metrics(),
            requested_learning_count=counts[key],source_state_sha256=_digest(policy_state(model)))
    with (output/'trace.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
        if training_mode=='none':snapshot(['1'])
        else:
            for t,query_id in enumerate(ids,1):
                x=ctx.get(query_id);arm=model.select(x)
                if arm not in model.arms:raise ValueError('Unknown learning arm')
                feedback=environment.step(query_id,arm)
                model.observe(arm,ctx.get(query_id),feedback.reward)
                reward+=feedback.reward;cost+=feedback.actual_cost;gap=float(oracle[query_id])-feedback.reward;regret+=gap
                actions[arm]+=1
                row=dict(round=t,query_id=query_id,arm_id=arm,reward=feedback.reward,actual_cost=feedback.actual_cost,
                    oracle_reward=float(oracle[query_id]),empirical_oracle_regret=gap,
                    cumulative_reward=reward,cumulative_actual_cost=cost,cumulative_empirical_oracle_regret=regret)
                stream.write(json.dumps(row,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
                keys=[key for key,count in counts.items() if count==t]
                if keys:snapshot(keys)
    return _finish(output,dict(version=1,kind='learning-stream',request=request,metrics=metrics(),
        snapshots=snapshots,training_mode=training_mode,
        learning_selection_role='unused_disjoint_reference' if training_mode=='none' else 'observed_selected_feedback',
        synthetic=bool(lineage.get('synthetic',False)),paper_equivalent=False),names)


def deploy_run(*,router,queries,outcomes,contexts,learning_selection,deployment_selection,
               estimator,input_tokens,input_token_provenance,budget,bin_size,lower_bound,upper_bound,
               lineage,output_dir,resume=False):
    ids=_selection_ids(deployment_selection,'deployment');output=Path(output_dir)
    q=queries.loc[queries.query_id.isin(ids),['query_id','query','task']].copy()
    o=outcomes.loc[outcomes.query_id.isin(ids)&outcomes.arm_id.isin(router.arms),
                   ['query_id','arm_id','reward','actual_cost']].copy()
    OfflineBandit(q,o,arm_catalog=router.arms)
    request=_json(dict(version=1,kind='deployment-run',lineage=lineage,
        router=router.metadata(),learning_selection=asdict(learning_selection),deployment_selection=asdict(deployment_selection),
        context_sha256=contexts.fingerprint(ids),estimator=estimator.metadata(),input_tokens_sha256=_digest(input_tokens),
        input_token_provenance=input_token_provenance,budget=budget,bin_size=bin_size,
        lower_bound=lower_bound,upper_bound=upper_bound,
        queries_sha256=_digest(q.set_index('query_id').loc[list(ids)].reset_index().to_dict('records')),
        outcomes_sha256=_digest(o.sort_values(['query_id','arm_id']).to_dict('records'))))
    existing=_resume(output,request,resume)
    if existing is not None:return existing
    before=router.metadata()['source_state_sha256']
    result=run_deployment(router,queries,outcomes,contexts,learning_selection,deployment_selection,
        estimator,input_tokens,input_token_provenance=input_token_provenance,budget=budget,bin_size=bin_size,
        lower_bound=lower_bound,upper_bound=upper_bound)
    if before!=router.metadata()['source_state_sha256']:raise ValueError('Deployment changed frozen state')
    output.mkdir(parents=True,exist_ok=False)
    with (output/'trace.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
        for row in result.trace:stream.write(json.dumps(asdict(row),ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
    _publish_json(output/'summary.json',result.summary)
    return _finish(output,dict(version=1,kind='deployment-run',request=request,summary=result.summary,
        synthetic=bool(lineage.get('synthetic',False)),paper_equivalent=False),['trace.jsonl','summary.json'])


def verify_pins(value,root):
    """Check all nested file identities without permitting paths outside the root."""
    root=Path(root).resolve()
    if isinstance(value,dict):
        if set(value)=={'path','sha256'}:
            file=(root/value['path']).resolve()
            if not file.is_relative_to(root):raise ValueError('Pinned path escapes project')
            if not file.is_file() or sha256(file)!=value['sha256']:
                raise ValueError('Pinned checksum mismatch: '+value['path'])
        else:
            for child in value.values():verify_pins(child,root)
    elif isinstance(value,list):
        for child in value:verify_pins(child,root)


def execute_group(bundle,output_dir,*,method,variant='main',resume=False,preflight=None):
    """One full learning stream followed by its budgets, never parallel jobs."""
    from scripts.resources import require_resources
    protocol=bundle['protocol'];plan=protocol['primary_plan'];domain=bundle['domain'];seed=bundle['seed']
    rows=[r for r in expected_runs(protocol) if (r['domain'],r['seed'],r['method'],r['variant'])==(domain,seed,method,variant)]
    if not rows:raise ValueError('Condition is outside the frozen inventory')
    root=Path(output_dir);parameters=protocol['tuning'][domain]['selected_parameters'][method]
    learning=bundle['selections']['learning'];deployment=bundle['selections']['deployment']
    if (len(learning.query_ids)!=plan['frozen_counts'][domain]['learning']
            or len(deployment.query_ids)!=plan['frozen_counts'][domain]['deployment']):
        raise ValueError('Full stream counts differ from frozen protocol')
    source_variant='main' if variant=='single-budget-bin' else variant
    learning_dir=root/'learning'/domain/f'seed-{seed}'/source_variant/method
    if variant=='single-budget-bin' and not (learning_dir/'manifest.json').is_file():
        raise ValueError('Single-bin ablation requires completed main learning')
    (require_resources if preflight is None else preflight)()
    lineage=dict(bundle['lineage'],domain=domain,seed=seed,method=method,variant=source_variant)
    model=(build_policy(bundle['prior'],method,parameters) if source_variant=='main'
           else variant_policy(bundle['prior'],source_variant,parameters))
    fractions=plan['learning_fractions'] if source_variant=='main' else [1.]
    trained=learn_stream(model=model,queries=bundle['queries'],outcomes=bundle['outcomes'],contexts=bundle['contexts'],
        selection=learning,seed=seed,fractions=fractions,lineage=lineage,output_dir=learning_dir,
        training_mode='none' if source_variant=='prior-only' else 'full',
        resume=resume or variant=='single-budget-bin')
    results=[]
    tuning=protocol['tuning'][domain]
    for row in rows:
        snap=trained['snapshots'][fraction_key(row['learning_fraction'])]
        checkpoint=learning_dir/snap['file']
        selected=replace(learning,query_ids=learning.query_ids[:row['learning_count']],
                         limit=row['learning_count'],subset_seed=None)
        # The existing deployment engine validates a complete rectangular table.
        # Keep just the declared reference learning prefix and deployment rows.
        members=set(selected.query_ids+deployment.query_ids)
        queries=bundle['queries'].loc[bundle['queries'].query_id.isin(members)].copy()
        outcomes=bundle['outcomes'].loc[bundle['outcomes'].query_id.isin(members)&
                                      bundle['outcomes'].arm_id.isin(protocol['arms'])].copy()
        deployment_lineage=dict(bundle['lineage'],condition=row,
            learning_manifest_sha256=sha256(learning_dir/'manifest.json'),
            learning_checkpoint_sha256=sha256(checkpoint),learning_rounds=snap['rounds'],
            learning_metrics=snap['metrics'],learning_selection_role=trained['learning_selection_role'])
        results.append(deploy_run(router=FrozenDeploymentRouter.load_checkpoint(checkpoint),
            queries=queries,outcomes=outcomes,contexts=bundle['contexts'],learning_selection=selected,
            deployment_selection=deployment,estimator=bundle['estimator'],input_tokens=bundle['input_counts'],
            input_token_provenance=bundle['input_token_provenance'],
            budget=tuning['absolute_reference_budget']*row['budget_fraction'],
            bin_size=len(deployment.query_ids) if variant=='single-budget-bin' else plan['bin_size'],
            **tuning['bounds'],lineage=deployment_lineage,output_dir=root/'deployment'/row['id'],resume=resume))
    return results


def _manifest(path,kind):
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    if value.get('manifest_sha256')!=_digest({k:v for k,v in value.items() if k!='manifest_sha256'}):
        raise ValueError('Manifest checksum mismatch: '+str(path))
    if value.get('version')!=1 or value.get('kind')!=kind:raise ValueError('Unexpected manifest kind')
    if value.get('synthetic') is not False:raise ValueError('Real experiments refuse synthetic inputs')
    return value


def load_real_bundle(protocol_path,index_path,*,domain,seed,dimension=64,root=None):
    """Pin all dependencies before constructing any real experiment condition."""
    import pandas as pd
    from scripts.fetch_data import ROOT
    from scripts.cost_inputs import load_cost_inputs
    from scripts.cost_policy import TuningTokenCostEstimator,TokenPrices
    from scripts.pretraining import load_checkpoint,_read_json
    from scripts.research_pipeline import load_prepared,load_routing,_aligned_cache
    root=Path(ROOT if root is None else root)
    protocol=_manifest(protocol_path,'frozen-local-experiment-protocol')
    index=_manifest(index_path,'verified-research-prior-index')
    verify_pins(protocol,root);verify_pins(index,root)
    if index['protocol']['sha256']!=sha256(protocol_path):raise ValueError('Prior index/protocol identity mismatch')
    plan=protocol['primary_plan']
    if domain not in plan['datasets'] or seed not in plan['replicate_seeds'] or dimension not in (64,8):
        raise ValueError('Replicate outside frozen protocol')
    expected={(d,s) for d in (64,8) for s in plan['replicate_seeds']}
    if len(index['fits'])!=len(expected) or {(v['dimension'],v['seed']) for v in index['fits']}!=expected:
        raise ValueError('Incomplete prior inventory')
    entry=next(v for v in index['fits'] if (v['dimension'],v['seed'])==(dimension,seed))
    prior=load_checkpoint(root/entry['prior']['path'])
    fit=_read_json(root/entry['fit_manifest']['path'])
    verification=json.loads((root/entry['verification']['path']).read_text(encoding='utf-8'))
    prepared_dir=root/'outputs/local-pipeline-verification-20261029/prepared-identity'
    prepared=load_prepared(prepared_dir)
    if (fit['checkpoint_sha256']!=entry['prior']['sha256'] or fit['prepared_sha256']!=sha256(prepared_dir/'manifest.json')
            or fit['synthetic'] or not fit['prior_ready'] or fit['arms']!=protocol['arms']
            or prior['arms']!=protocol['arms'] or not prior['metrics']['validation']['all_lambdas_ready']
            or prior['embedding']['recipe']['synthetic'] or prior['dimension']!=dimension):
        raise ValueError('Invalid real prior/preparation lineage')
    for name,want in dict(seed=seed,**{k:plan['pretraining'][k] for k in
                                     ('projection_epochs','prior_epochs','learning_rate','margin')}).items():
        if prior['protocol'][name]!=want:raise ValueError('Prior training protocol mismatch: '+name)
    if (verification['verified'] is not True or verification['synthetic']
            or verification['checkpoint_sha256']!=entry['prior']['sha256']
            or verification['fit_manifest_sha256']!=entry['fit_manifest']['sha256']
            or (verification['dimension'],verification['seed'])!=(dimension,seed)):
        raise ValueError('Independent prior verification mismatch')
    pref_manifest=_read_json(root/'outputs/research-embeddings-20261029/preference-identity/manifest.json')
    if prior['embedding']['cache_sha256']!=_digest(pref_manifest):raise ValueError('Preference cache identity mismatch')
    input_dir=root/'outputs/research-input-audit-20261029/inputs'
    inputs=json.loads((input_dir/'manifest.json').read_text(encoding='utf-8'))
    if inputs['manifest_sha256']!=_digest({k:v for k,v in inputs.items() if k!='manifest_sha256'}):
        raise ValueError('Invalid routing input bundle')
    if set(inputs['files'])!={'routing-inputs.json','preference-inputs.json','phases.json'}:
        raise ValueError('Unexpected routing input inventory')
    for name,fingerprint in inputs['files'].items():
        if sha256(input_dir/name)!=fingerprint:raise ValueError('Routing input checksum mismatch')
    cache=_aligned_cache(root/'outputs/research-embeddings-20261029/routing-full',
                         json.loads((input_dir/'routing-inputs.json').read_text(encoding='utf-8')))
    if cache.provenance!=prior['embedding']['recipe'] or cache.dimension!=prior['input_dimension']:
        raise ValueError('Prior/routing embedding recipe mismatch')
    folder,queries,splits,hashes=load_routing(root/'data',root=root)
    costs=load_cost_inputs(root/'outputs/research-cost-inputs-20261029')
    if (costs['manifest']['synthetic'] or costs['manifest']['source_hashes']!=hashes
            or inputs['source_hashes']!=hashes or prepared['manifest']['source_hashes']!=hashes):
        raise ValueError('Source/cost lineage mismatch')
    selections={phase:splits.select(phase) for phase in ('tuning','learning','deployment')}
    if domain=='mmlu':
        members=set(queries.loc[queries.task.str.startswith('mmlu-'),'query_id'])
        for phase,selection in selections.items():
            ids=tuple(q for q in selection.query_ids if q in members)
            selections[phase]=replace(selection,query_ids=ids,full_size=len(ids))
    if {p:len(s.query_ids) for p,s in selections.items()}!=plan['frozen_counts'][domain]:
        raise ValueError('Frozen phase counts differ')
    tuning=protocol['tuning'][domain];arms=prior['arms'];metadata=costs['estimator'];tune_ids=selections['tuning'].query_ids
    estimator=TuningTokenCostEstimator.fit(selections['tuning'],
        {q:{a:costs['tuning-output-counts'][q][a] for a in arms} for q in tune_ids},
        {a:TokenPrices(**metadata['prices'][a]) for a in arms},
        token_provenance=metadata['token_provenance'],price_provenance=metadata['price_provenance'])
    if estimator.metadata()!=json.loads((root/tuning['estimator']['path']).read_text(encoding='utf-8')):
        raise ValueError('Frozen estimator differs from tuning-only counts')
    ids=selections['learning'].query_ids+selections['deployment'].query_ids
    contexts=ContextTable({q:prior['W']@cache.get(q)+prior['b'] for q in ids},
                          label='REAL pinned BGE-M3 with verified preference projection')
    queries=queries.loc[queries.query_id.isin(ids)].copy()
    outcomes=pd.read_parquet(folder/'outcomes.parquet',columns=['query_id','arm_id','reward','actual_cost'])
    outcomes=outcomes.loc[outcomes.query_id.isin(ids)&outcomes.arm_id.isin(arms)].copy()
    code={str(p.relative_to(root)).replace('\\','/'):sha256(p) for p in sorted((root/'scripts').glob('*.py'))}
    lineage=dict(protocol_sha256=sha256(protocol_path),prior_index_sha256=sha256(index_path),
        prior_file_sha256=entry['prior']['sha256'],prior_verification_sha256=entry['verification']['sha256'],
        routing_cache_sha256=cache.fingerprint(),source_hashes=hashes,
        tuning_stage_sha256=tuning['stage']['sha256'],cost_manifest_sha256=costs['manifest']['manifest_sha256'],
        code_sha256=code,synthetic=False,paper_equivalent=False)
    return dict(protocol=protocol,prior=prior,domain=domain,seed=seed,queries=queries,outcomes=outcomes,
        contexts=contexts,selections=selections,estimator=estimator,
        input_counts={q:costs['input-counts'][q] for q in selections['deployment'].query_ids},
        input_token_provenance=metadata['token_provenance'],lineage=lineage)


def main(argv=None):
    import argparse,os,shutil,time
    from scripts.fetch_data import ROOT
    from scripts.resources import require_resources
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/frozen_experiment_protocol.json')
    parser.add_argument('--prior-index',type=Path,default=ROOT/'configs/research_prior_index.json')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--domain',choices=['routerbench','mmlu'],required=True)
    parser.add_argument('--seed',type=int,choices=[7,42,2026],required=True)
    parser.add_argument('--method',default='all')
    parser.add_argument('--variant',default='main',choices=['main','zero-prior-direction','unit-initialization-strength',
                        'prior-only','single-budget-bin','projection-dimension-8'])
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    if any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')):
        raise RuntimeError('Set OMP_NUM_THREADS, MKL_NUM_THREADS and OPENBLAS_NUM_THREADS to 1 before starting Python')
    print(json.dumps({'preflight':require_resources()}),flush=True)
    destination=args.output_dir.resolve()
    if not destination.is_relative_to((ROOT/'outputs').resolve()):raise ValueError('Experiment outputs must remain under project outputs')
    ancestor=destination
    while not ancestor.exists():ancestor=ancestor.parent
    if shutil.disk_usage(ancestor).free<16*1024**3:raise RuntimeError('Need at least 16 GiB free disk for declared trace allowance')
    started=time.perf_counter()
    bundle=load_real_bundle(args.protocol,args.prior_index,domain=args.domain,seed=args.seed,
                           dimension=8 if args.variant=='projection-dimension-8' else 64)
    methods=list(bundle['protocol']['tuning'][args.domain]['selected_parameters']) if args.method=='all' else [args.method]
    if args.variant!='main':methods=['pilot'] if args.method=='all' else methods
    for method in methods:
        before=time.perf_counter()
        results=execute_group(bundle,args.output_dir,method=method,variant=args.variant,resume=args.resume)
        print(json.dumps({'domain':args.domain,'seed':args.seed,'method':method,'variant':args.variant,
            'completed_deployment_runs':len(results),'elapsed_seconds':time.perf_counter()-before}),flush=True)
        del results
    print(json.dumps({'total_elapsed_seconds':time.perf_counter()-started}),flush=True)


if __name__=='__main__':main()

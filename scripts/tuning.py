"""Persist complete prequential searches using tuning labels only."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.bandit_contracts import ContextTable
from scripts.cost_policy import TuningTokenCostEstimator, TokenPrices, _selection_ids, _finite
from scripts.data_gate import OfflineBandit
from scripts.deployment_adapter import FrozenDeploymentRouter
from scripts.fetch_data import ROOT, sha256
from scripts.input_audit import _publish_json
from scripts.pilot import PILOT
from scripts.policies import LinearUCB, ExploreOnly, EpochGreedy, RandomPolicy, FixedPolicy
from scripts.pretraining import _digest, _integer, _schema, _validate_checkpoint, _read_json, load_checkpoint


SCORE_RULE = 'maximum_prequential_total_reward; first_grid_entry_on_exact_tie'


def validate_grid(grid):
    _schema(grid,{'version','seed','alpha_grid','window_grid'},'tuning grid')
    if type(grid['version']) is not int or grid['version'] != 1:
        raise ValueError('Unsupported tuning grid')
    _integer(grid['seed'],'tuning seed',0)
    for name in ('alpha_grid','window_grid'):
        values=grid[name]
        if not isinstance(values,list) or not values:
            raise ValueError('Tuning grids must be nonempty lists')
        for value in values:
            if name=='window_grid':_integer(value,'window')
            else:_finite(value,'alpha',minimum=0)
        if sorted(set(values)) != values:
            raise ValueError('Tuning grids must be unique and increasing')
    return json.loads(json.dumps(grid,allow_nan=False))


def candidate_specs(arms,grid):
    grid=validate_grid(grid)
    specs=[(method,{'alpha':a}) for method in ('pilot','linucb') for a in grid['alpha_grid']]
    specs += [('epoch-greedy',{'window':w}) for w in grid['window_grid']]
    specs += [('explore-only',{}),('random',{})]
    specs += [(f'fixed-{i:02d}',{'arm_id':a}) for i,a in enumerate(arms)]
    return [dict(id=f'candidate-{i:03d}',method=method,parameters=params) for i,(method,params) in enumerate(specs)]


def build_policy(prior,method,parameters):
    prior=_validate_checkpoint(prior)
    if not prior['metrics']['validation']['all_lambdas_ready']:
        raise ValueError('Prior validation is blocked')
    arms=tuple(prior['arms']);dimension=prior['dimension']
    if method in ('pilot','linucb'):
        _schema(parameters,{'alpha'},'UCB parameters')
        if method=='linucb':return LinearUCB(arms,dimension,**parameters)
        return PILOT(arms,dimension,priors=dict(zip(arms,prior['theta'])),
            lambdas={a:prior['metrics']['validation']['per_arm'][a]['lambda'] for a in arms},
            provenance='preference cache sha256='+prior['embedding']['cache_sha256'],**parameters)
    if method=='epoch-greedy':
        _schema(parameters,{'window'},'epoch parameters')
        return EpochGreedy(arms,dimension,**parameters)
    if method in ('explore-only','random'):
        _schema(parameters,set(),'constant parameters')
        return (ExploreOnly if method=='explore-only' else RandomPolicy)(arms,dimension)
    if method.startswith('fixed-'):
        _schema(parameters,{'arm_id'},'fixed parameters')
        return FixedPolicy(arms,dimension,**parameters)
    raise ValueError('Unsupported tuning method')


def _best(candidates):
    selected={}
    for candidate in candidates:
        current=selected.get(candidate['method'])
        if current is None or candidate['total_reward']>current['total_reward']:
            selected[candidate['method']]=candidate
    return selected


def run_search(prior,queries,outcomes,contexts,selection,grid,output_dir,*,
               allow_synthetic=False,lineage=None):
    output=Path(output_dir)
    if output.exists():raise FileExistsError(output)
    ids=_selection_ids(selection,'tuning')
    grid=validate_grid(grid);prior=_validate_checkpoint(prior)
    if not prior['metrics']['validation']['all_lambdas_ready']:
        raise ValueError('Prior validation is blocked')
    synthetic=prior['embedding']['recipe']['synthetic']
    if synthetic and not allow_synthetic:raise ValueError('Real tuning refuses synthetic prior')
    arms=tuple(prior['arms'])
    # This isolation precedes outcome validation/hashing: held-out labels cannot
    # affect search success, scores, snapshots or fingerprints.
    selected_queries=queries.loc[queries.query_id.isin(ids)].copy()
    selected_outcomes=outcomes.loc[outcomes.query_id.isin(ids)&outcomes.arm_id.isin(arms),
                                  ['query_id','arm_id','reward','actual_cost']].copy()
    if set(selected_queries.query_id)!=set(ids):raise ValueError('Missing tuning query')
    OfflineBandit(selected_queries,selected_outcomes,arm_catalog=arms)
    try:
        selected_contexts=ContextTable({q:contexts.get(q) for q in ids},label=contexts.label)
    except KeyError as exc:raise ValueError('Missing tuning context') from exc
    if selected_contexts.dimension!=prior['dimension'] or any(not np.any(selected_contexts.get(q)) for q in ids):
        raise ValueError('Invalid tuning context dimension or zero direction')
    specifications=candidate_specs(arms,grid)
    # Construct all policy configurations before creating an output directory.
    for spec in specifications:build_policy(prior,spec['method'],spec['parameters'])
    serialized_prior={**prior,**{k:prior[k].tolist() for k in ('W','b','theta')}}
    provenance=dict(prior_sha256=_digest(serialized_prior),
        queries_sha256=_digest(selected_queries.set_index('query_id').loc[list(ids),['query','task']].reset_index().to_dict('records')),
        outcomes_sha256=_digest(selected_outcomes.sort_values(['query_id','arm_id']).to_dict('records')),
        context_sha256=selected_contexts.fingerprint(ids),context_label=selected_contexts.label,
        lineage={} if lineage is None else lineage)
    _digest(provenance)
    output.mkdir(parents=True,exist_ok=False)
    candidates=[];files={}
    for spec in specifications:
        environment=OfflineBandit(selected_queries,selected_outcomes,arm_catalog=arms)
        model=build_policy(prior,spec['method'],spec['parameters']);model.reset(grid['seed'])
        folder=output/spec['id'];folder.mkdir()
        rewards=[]
        with (folder/'trace.jsonl').open('x',encoding='utf-8') as stream:
            for t,q in enumerate(ids):
                x=selected_contexts.get(q)
                predictions=model.predict_rewards(x)
                arm=model.select(x)
                if arm not in arms:raise ValueError('Unknown selected tuning arm')
                feedback=environment.step(q,arm)
                row=dict(round=t+1,query_id=q,arm_id=arm,reward=feedback.reward,
                         actual_cost=feedback.actual_cost,predictions=predictions,rounds_before_feedback=model.rounds)
                model.observe(arm,selected_contexts.get(q),feedback.reward)
                rewards.append(feedback.reward)
                stream.write(json.dumps(row,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
        frozen=FrozenDeploymentRouter(model,seed=grid['seed'])
        frozen.save_checkpoint(folder/'router.json')
        total=math.fsum(rewards)
        candidate=dict(spec,rounds=len(ids),total_reward=total,mean_reward=total/len(ids),
            trace=spec['id']+'/trace.jsonl',checkpoint=spec['id']+'/router.json',
            final_state_sha256=frozen.metadata()['source_state_sha256'])
        candidates.append(candidate)
        for name in ('trace','checkpoint'):files[candidate[name]]=sha256(output/candidate[name])
    result=dict(version=1,kind='tuning-search',phase='tuning',selection={**asdict(selection),'query_ids':list(ids)},
        arms=list(arms),grid=grid,score_rule=SCORE_RULE,candidates=candidates,selected=_best(candidates),
        provenance=provenance,synthetic=synthetic,paper_equivalent=False,files=files)
    result['manifest_sha256']=_digest(result)
    _publish_json(output/'manifest.json',result)
    return result


def load_search(path):
    path=Path(path);result=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    if (result.get('version')!=1 or result.get('kind')!='tuning-search' or result.get('phase')!='tuning'
            or result.get('score_rule')!=SCORE_RULE
            or result.get('manifest_sha256')!=_digest({k:v for k,v in result.items() if k!='manifest_sha256'})):
        raise ValueError('Invalid tuning manifest or checksum')
    expected=candidate_specs(result['arms'],result['grid'])
    candidates=result['candidates']
    if len(candidates)!=len(expected):raise ValueError('Incomplete candidate inventory')
    filenames=set()
    for candidate,spec in zip(candidates,expected):
        if any(candidate.get(k)!=v for k,v in spec.items()):raise ValueError('Candidate specification mismatch')
        for key,name in (('trace','trace.jsonl'),('checkpoint','router.json')):
            relative=spec['id']+'/'+name
            if candidate.get(key)!=relative:raise ValueError('Invalid candidate artifact path')
            filenames.add(relative)
    if set(result.get('files',{}))!=filenames or result['selected']!=_best(candidates):
        raise ValueError('Incomplete files or inconsistent parameter selection')
    for name,fingerprint in result['files'].items():
        file=path/name
        if not file.is_file() or sha256(file)!=fingerprint:raise ValueError('Tuning artifact checksum mismatch: '+name)
    return result


def calibrate_budget(path,input_counts,estimator,*,reference_arm):
    result=load_search(path);ids=result['selection']['query_ids'];arms=result['arms']
    if not isinstance(estimator,TuningTokenCostEstimator):raise ValueError('Expected tuning-only estimator')
    meta=estimator.metadata()
    if (meta['tuning_query_ids']!=ids or meta['split_manifest_sha256']!=result['selection']['manifest_sha256']
            or set(meta['prices'])!=set(arms) or set(input_counts)!=set(ids) or reference_arm not in arms):
        raise ValueError('Tuning cost provenance or coverage mismatch')
    estimates={q:estimator.estimate(input_counts[q]) for q in ids}
    ratios=[];excluded=0
    for candidate in result['selected'].values():
        rows=[json.loads(line) for line in (Path(path)/candidate['trace']).read_text(encoding='utf-8').splitlines()]
        if [r['query_id'] for r in rows]!=ids:raise ValueError('Tuning trace query order mismatch')
        for row in rows:
            if set(row['predictions'])!=set(arms):raise ValueError('Tuning prediction catalog mismatch')
            for arm in arms:
                score=_finite(row['predictions'][arm],'prequential score');cost=estimates[row['query_id']][arm]
                if score>0 and cost>0:ratios.append(_finite(score/cost,'positive reward/cost ratio',minimum=0))
                else:excluded+=1
    if not ratios:raise ValueError('No positive tuning ratios; cannot infer positive bounds')
    lower=_finite(min(ratios)*.5,'lower bound',minimum=0)
    upper=_finite(max(ratios)*2.,'upper bound',minimum=0)
    if lower==0:raise ValueError('Positive ratio bound underflow')
    return dict(version=1,rule='pooled_selected_prequential_positive_ratios; LB=0.5*min; UB=2*max',
        lower_bound=lower,upper_bound=upper,positive_ratio_count=len(ratios),excluded_nonpositive_or_zero_cost=excluded,
        reference_arm=reference_arm,reference_estimated_cost_per_query=math.fsum(estimates[q][reference_arm] for q in ids)/len(ids),
        tuning_query_count=len(ids),search_manifest_sha256=result['manifest_sha256'],
        input_counts_sha256=_digest(input_counts),estimator_sha256=_digest(meta),
        interpretation='empirical tuning assumptions; no deployment ratio guarantee')


def run_real_search(data_dir,prepared_dir,input_dir,cache_dir,fit_dir,cost_dir,grid,output_dir,*,domain='routerbench'):
    from scripts.research_pipeline import load_prepared,load_routing,_aligned_cache
    from scripts.cost_inputs import load_cost_inputs
    output=Path(output_dir)
    if output.exists():raise FileExistsError(output)
    if domain not in ('routerbench','mmlu'):raise ValueError('Unknown tuning domain')
    prepared=load_prepared(prepared_dir);fit_dir=Path(fit_dir);input_dir=Path(input_dir)
    fit=_read_json(fit_dir/'fit-manifest.json');prior=load_checkpoint(fit_dir/'prior.json')
    if (fit['prepared_sha256']!=sha256(Path(prepared_dir)/'manifest.json')
            or fit['checkpoint_sha256']!=sha256(fit_dir/'prior.json')
            or fit['arms']!=prior['arms'] or fit['arms']!=prepared['manifest']['arms'] or fit['synthetic']):
        raise ValueError('Real fit/preparation lineage mismatch')
    bundle=json.loads((input_dir/'manifest.json').read_text(encoding='utf-8'))
    if (set(bundle['files'])!={'routing-inputs.json','preference-inputs.json','phases.json'}
            or bundle['manifest_sha256']!=_digest({k:v for k,v in bundle.items() if k!='manifest_sha256'})):
        raise ValueError('Invalid full-input bundle')
    for name,fingerprint in bundle['files'].items():
        if sha256(input_dir/name)!=fingerprint:raise ValueError('Full input checksum mismatch')
    cache=_aligned_cache(cache_dir,json.loads((input_dir/'routing-inputs.json').read_text(encoding='utf-8')))
    if cache.provenance!=prior['embedding']['recipe'] or cache.dimension!=prior['input_dimension']:
        raise ValueError('Prior/routing embedding recipe mismatch')
    folder,queries,splits,hashes=load_routing(data_dir)
    costs=load_cost_inputs(cost_dir)
    if (costs['manifest']['synthetic'] or costs['manifest']['source_hashes']!=hashes
            or bundle['source_hashes']!=hashes or prepared['manifest']['source_hashes']!=hashes):
        raise ValueError('Source/cost lineage mismatch')
    selection=splits.select('tuning')
    if domain=='mmlu':
        members=set(queries.loc[queries.task.str.startswith('mmlu-'),'query_id'])
        ids=tuple(q for q in selection.query_ids if q in members)
        selection=replace(selection,query_ids=ids,full_size=len(ids))
    ids=selection.query_ids
    contexts=ContextTable({q:prior['W']@cache.get(q)+prior['b'] for q in ids},
                          label='REAL pinned BGE-M3 with verified preference projection')
    outcomes=pd.read_parquet(folder/'outcomes.parquet')
    lineage=dict(domain=domain,prior_file_sha256=sha256(fit_dir/'prior.json'),
        fit_manifest_sha256=sha256(fit_dir/'fit-manifest.json'),prepared_sha256=sha256(Path(prepared_dir)/'manifest.json'),
        input_bundle_sha256=bundle['manifest_sha256'],routing_cache_sha256=cache.fingerprint(),
        source_hashes=hashes,cost_manifest_sha256=costs['manifest']['manifest_sha256'],
        scope='real tuning-only parameter selection; not deployment performance')
    result=run_search(prior,queries,outcomes,contexts,selection,grid,output/'search',lineage=lineage)
    arms=prior['arms'];metadata=costs['estimator']
    estimator=TuningTokenCostEstimator.fit(selection,
        {q:{a:costs['tuning-output-counts'][q][a] for a in arms} for q in ids},
        {a:TokenPrices(**metadata['prices'][a]) for a in arms},
        token_provenance=metadata['token_provenance'],price_provenance=metadata['price_provenance'])
    calibration=calibrate_budget(output/'search',{q:costs['input-counts'][q] for q in ids},estimator,
                                 reference_arm='gpt-4-1106-preview')
    _publish_json(output/'calibration.json',calibration)
    _publish_json(output/'estimator.json',estimator.metadata())
    stage=dict(version=1,kind='real-tuning-stage',domain=domain,tuning_rows=len(ids),
        candidates=len(result['candidates']),selected=result['selected'],lineage=lineage,
        synthetic=False,paper_equivalent=False,search_manifest_sha256=result['manifest_sha256'],
        files={name:sha256(output/name) for name in ('search/manifest.json','calibration.json','estimator.json')})
    stage['manifest_sha256']=_digest(stage)
    _publish_json(output/'manifest.json',stage)
    return stage


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir','prepared-dir','input-dir','cache-dir','fit-dir','cost-dir','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--grid',type=Path,default=ROOT/'configs/tuning_grid.json')
    parser.add_argument('--domain',choices=('routerbench','mmlu'),default='routerbench')
    args=parser.parse_args(argv)
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    from scripts.resources import require_resources
    print(json.dumps({'preflight':require_resources()}),flush=True)
    result=run_real_search(args.data_dir,args.prepared_dir,args.input_dir,args.cache_dir,args.fit_dir,
        args.cost_dir,json.loads(args.grid.read_text()),args.output_dir,domain=args.domain)
    print(json.dumps({'output_dir':str(args.output_dir),'tuning_rows':result['tuning_rows'],
        'candidates':result['candidates'],'selected':{m:c['parameters'] for m,c in result['selected'].items()}}))


if __name__=='__main__':main()

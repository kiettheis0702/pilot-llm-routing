"""Audited reuse of immutable Hybrid code with the official8/16 batch default.

Only batching, its synthetic probe, output paths and source metadata change.
The two source files stay byte-identical. Model/data/labels/optimizer/epoch and
checkpoint-selection logic are reused. Verification separately reuses the
independent reader, with matching8-example inference grouping.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PARENT_TRAINER=ROOT/'tools/openai_study_hybrid_train.py'
PARENT_VERIFIER=ROOT/'tools/openai_study_hybrid_verify.py'
BATCH_PROFILE=ROOT/'outputs/openai-followup-20261031/hybrid-batch-profile.json'
TRAINER_SHA='e692d57e187902d3a3f50b355fd05356c3ddf6adb79170b135af9ca77a2a1ef2'
VERIFIER_SHA='ae669c0345cd2f508d98ca5ec974b53b8e6a081e30710ab639185656a75fdb60'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def function(tree,name):
    matches=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
    if len(matches)!=1:
        raise ValueError('Expected exactly one reviewed function: '+name)
    return matches[0]


def assignment(callable_node,name):
    matches=[n for n in ast.walk(callable_node) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
    if len(matches)!=1:
        raise ValueError('Reviewed assignment changed: '+name)
    return matches[0]


def patch_keyword(call,name,before,after):
    matches=[k for k in call.keywords if k.arg==name]
    if len(matches)!=1 or not isinstance(matches[0].value,ast.Constant) or matches[0].value.value!=before:
        raise ValueError('Reviewed keyword changed: '+name)
    matches[0].value=ast.Constant(after)


def patch_batches(node,expected):
    ranges=slices=0
    for item in list(ast.walk(node)):
        if (isinstance(item,ast.Call) and isinstance(item.func,ast.Name) and item.func.id=='range'
                and len(item.args)==3 and isinstance(item.args[2],ast.Constant) and item.args[2].value==2):
            item.args[2]=ast.Constant(8);ranges+=1
        if (isinstance(item,ast.Slice) and isinstance(item.upper,ast.BinOp)
                and isinstance(item.upper.left,ast.Name) and item.upper.left.id=='offset'
                and isinstance(item.upper.op,ast.Add) and isinstance(item.upper.right,ast.Constant)
                and item.upper.right.value==2):
            item.upper.right=ast.Constant(8);slices+=1
    if (ranges,slices)!=(expected,expected):
        raise ValueError(f'Reviewed batch loops changed: {(ranges,slices)}')


def patch_base(tree,before,after):
    item=assignment(tree,'BASE')
    if not isinstance(item.value,ast.BinOp) or not isinstance(item.value.right,ast.Constant) or item.value.right.value!=before:
        raise ValueError('Reviewed output base changed')
    item.value.right=ast.Constant(after)


def add_reuse(call):
    if any(k.arg=='source_reuse' for k in call.keywords):
        raise ValueError('Unexpected existing source-reuse declaration')
    call.keywords.append(ast.keyword(arg='source_reuse',value=ast.Name(id='SOURCE_REUSE',ctx=ast.Load())))


def adapt_trainer_source(raw):
    if hashlib.sha256(raw).hexdigest()!=TRAINER_SHA:
        raise ValueError('Frozen parent trainer source differs')
    tree=ast.parse(raw);patch_base(tree,'hybrid','hybrid-default-batch')
    probe=function(tree,'probe');rows=assignment(probe,'rows')
    if not isinstance(rows.value,ast.List) or len(rows.value.elts)!=2:
        raise ValueError('Reviewed synthetic probe rows changed')
    rows.value=ast.BinOp(left=rows.value,op=ast.Mult(),right=ast.Constant(4))
    tensors=shapes=0
    for item in list(ast.walk(probe)):
        if (isinstance(item,ast.Call) and isinstance(item.func,ast.Attribute) and item.func.attr=='tensor'
                and item.args and isinstance(item.args[0],ast.List)):
            value=ast.literal_eval(item.args[0])
            if value in ([[2.,-1.],[-1.,2.]],[0,1],[1,0]):
                item.args[0]=ast.BinOp(left=item.args[0],op=ast.Mult(),right=ast.Constant(4));tensors+=1
        if isinstance(item,ast.Tuple):
            try:value=ast.literal_eval(item)
            except (ValueError,TypeError):continue
            if value==(2,128):item.elts[0]=ast.Constant(8);shapes+=1
    if (tensors,shapes)!=(5,1):
        raise ValueError(f'Reviewed eight-example semantic probe changed:{tensors}/{shapes}')
    proof=assignment(probe,'proof').value;patch_keyword(proof,'microbatch',2,8);add_reuse(proof)
    declaration=function(tree,'declare');config=assignment(declaration,'config').value
    patch_keyword(config,'microbatch',2,8);patch_keyword(config,'accumulation',64,16)
    config.keywords.append(ast.keyword(arg='eval_microbatch',value=ast.Constant(8)))
    protocol=assignment(declaration,'protocol').value;add_reuse(protocol)
    patch_keyword(protocol,'differences',
        'verified Llama2 pair; local splits; micro2/accum64; local validation CE; native threshold front; not author-exact',
        'verified Llama2 pair; local splits; official micro8/accum16 and eval8; local validation CE; native threshold front; not author-exact')
    pins=next(k.value for k in protocol.keywords if k.arg=='source_pins')
    if not isinstance(pins,ast.List):raise ValueError('Reviewed source pins changed')
    pins.elts[:0]=[ast.Call(func=ast.Name(id='pin',ctx=ast.Load()),args=[ast.Name(id=name,ctx=ast.Load())],keywords=[])
                   for name in ('PARENT_TRAINER','BATCH_PROFILE')]
    patch_batches(function(tree,'train'),2)
    ast.fix_missing_locations(tree)
    audit=dict(train_microbatch=8,accumulation=16,eval_microbatch=8,effective_batch=128,
        changed_functions=['probe','declare','train'],changed_global='output BASE',
        model_data_labels_optimizer_epochs_selection='unchanged parent function bodies/rules')
    return tree,audit


def adapt_verifier_source(raw):
    if hashlib.sha256(raw).hexdigest()!=VERIFIER_SHA:
        raise ValueError('Frozen independent verifier source differs')
    tree=ast.parse(raw)
    patch_base(tree,'outputs/openai-followup-20261031/hybrid','outputs/openai-followup-20261031/hybrid-default-batch')
    verify=function(tree,'verify')
    nested=[n for n in ast.walk(verify) if isinstance(n,ast.FunctionDef) and n.name=='forward']
    if len(nested)!=1:raise ValueError('Reviewed independent forward function changed')
    patch_batches(nested[0],1);add_reuse(assignment(verify,'proof').value)
    ast.fix_missing_locations(tree)
    return tree,dict(eval_microbatch=8,changes='output BASE, inference grouping and adapter provenance only',
        metric_label_checkpoint_and_neural_math='unchanged')


def namespace(mode):
    profile=json.loads(BATCH_PROFILE.read_text(encoding='utf-8'))
    if (not profile['synthetic_hardware_only'] or not profile['no_binary_outcomes_used']
            or profile['network_attempts']!=0 or profile['paid_api_calls']!=0
            or profile['parent_trainer_sha256']!=TRAINER_SHA):
        raise ValueError('Actual hardware-only default-batch decision proof required')
    modes={r['microbatch']:r for r in profile['results']}
    if (set(modes)!={2,8} or not all(r['success'] for r in modes.values())
            or modes[8]['median_seconds_per128']>=modes[2]['median_seconds_per128']):
        raise ValueError('Default batch did not meet the declared feasibility/throughput rule')
    trainer,trainer_audit=adapt_trainer_source(PARENT_TRAINER.read_bytes())
    verifier,verifier_audit=adapt_verifier_source(PARENT_VERIFIER.read_bytes())
    reuse=dict(parent_trainer_sha256=TRAINER_SHA,parent_verifier_sha256=VERIFIER_SHA,
        adapter_sha256=sha(__file__),hardware_profile_sha256=sha(BATCH_PROFILE),
        trainer_ast_sha256=hashlib.sha256(ast.dump(trainer).encode()).hexdigest(),
        verifier_ast_sha256=hashlib.sha256(ast.dump(verifier).encode()).hexdigest(),
        trainer_changes=trainer_audit,verifier_changes=verifier_audit,
        scope='binary only; official8/16 training and8 evaluation; conservative drift recipe preserved')
    scope=dict(__name__='audited_hybrid_default_batch',__file__=str(Path(__file__).resolve()),
        SOURCE_REUSE=reuse,PARENT_TRAINER=PARENT_TRAINER,PARENT_VERIFIER=PARENT_VERIFIER,BATCH_PROFILE=BATCH_PROFILE)
    selected=trainer if mode=='trainer' else verifier
    exec(compile(selected,str(PARENT_TRAINER if mode=='trainer' else PARENT_VERIFIER),'exec'),scope)
    return scope


def main():
    from scripts.resources import require_resources
    require_resources();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['probe','audit','declare','train','verify']);args=parser.parse_args()
    scope=namespace('verifier' if args.action=='verify' else 'trainer')
    if args.action=='probe':scope['probe']()
    elif args.action=='verify':scope['verify']('binary')
    else:scope[args.action]('binary')


if __name__=='__main__':
    main()

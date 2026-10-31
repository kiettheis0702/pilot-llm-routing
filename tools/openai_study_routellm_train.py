"""Pinned RouteLLM MF training with local vectors and calibration-only selection.

Only the reviewed training ClassDef is executed. Upstream module imports and
inference forward (which initializes/calls an OpenAI client) are never executed.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import socket
from pathlib import Path

import numpy as np

from scripts.embedding_store import DiskEmbeddingCache
from scripts.resources import require_resources

COMMIT='0b64fdafe049e596a3f5657c219329f24af24198'
ARM_NAMES={
    'WizardLM/WizardLM-13B-V1.2':'wizardlm-13b',
    'claude-instant-v1':'claude-instant-1', 'claude-v1':'claude-1',
    'gpt-3.5-turbo-1106':'gpt-3.5-turbo-1106',
    'gpt-4-1106-preview':'gpt-4-1106-preview',
    'meta/code-llama-instruct-34b-chat':'codellama-34b-instruct',
    'meta/llama-2-70b-chat':'llama-2-70b-chat',
    'zero-one-ai/Yi-34B-Chat':'yi-34b-chat',
}


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def reviewed_tree(path,expected_sha256):
    if sha(path)!=expected_sha256:raise ValueError('Reviewed source checksum mismatch')
    return ast.parse(Path(path).read_text(encoding='utf-8'),filename=str(path))


def load_training_class(path,expected_sha256,namespace):
    tree=reviewed_tree(path,expected_sha256)
    classes=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='MFModel_Train']
    if len(classes)!=1:raise ValueError('Expected one reviewed MF training class')
    if classes[0].decorator_list:raise ValueError('Unexpected class decorator')
    selected=ast.Module(body=classes,type_ignores=[])
    scope=dict(namespace);scope['__name__']='reviewed_routellm_mf'
    exec(compile(selected,str(path),'exec'),scope)
    return scope['MFModel_Train']


def load_model_ids(path,expected_sha256):
    tree=reviewed_tree(path,expected_sha256)
    values=[node.value for node in tree.body if isinstance(node,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='MODEL_IDS' for t in node.targets)]
    if len(values)!=1:raise ValueError('Expected one literal MODEL_IDS assignment')
    ids=ast.literal_eval(values[0])
    if (not isinstance(ids,dict) or not all(isinstance(k,str) and type(v) is int for k,v in ids.items())
            or sorted(ids.values())!=list(range(len(ids)))):raise ValueError('Invalid model IDs')
    return ids


def best_epoch(history):
    if not history or any(type(r['epoch']) is not int or r['epoch']<1
            or not math.isfinite(r['calibration_loss']) for r in history):
        raise ValueError('Invalid calibration history')
    return min(history,key=lambda r:(r['calibration_loss'],r['epoch']))['epoch']


def partition_pairs(records,query_hashes,arm_ids):
    groups={s:[] for s in ('train','calibration','test')};seen={}
    for row in records:
        q,w,l,s=(row[k] for k in ('query_id','winner','loser','split'))
        if q not in query_hashes or w not in arm_ids or l not in arm_ids or w==l or s not in groups:
            raise ValueError('Unknown query, model or split')
        h=query_hashes[q]
        if h in seen and seen[h]!=s:raise ValueError('Question overlap across splits')
        seen[h]=s;groups[s].append((q,arm_ids[w],arm_ids[l]))
    return groups


def write_json(path,payload):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(payload,stream,indent=2,allow_nan=False)


def source_inputs(root):
    base=root/'outputs/openai-followup-20261031'
    sources=base/'sources';manifest=json.loads((sources/'manifest.json').read_text(encoding='utf-8'))
    folder='lm-sys__RouteLLM/code/routellm/routers/matrix_factorization/'
    found={}
    for name in ('model.py','train_matrix_factorization.py'):
        record=next(r for r in manifest['records'] if r.get('path')==folder+name and r['ok'])
        if f'/{COMMIT}/' not in record['url']:raise ValueError('Wrong RouteLLM commit')
        path=sources/record['path'];reviewed_tree(path,record['sha256'])
        found[name]=dict(path=str(path.relative_to(root)),sha256=record['sha256'],url=record['url'])
    return found


def freeze(root):
    root=Path(root);base=root/'outputs/openai-followup-20261031/routellm'
    if base.exists():raise FileExistsError(base)
    sources=source_inputs(root)
    cache_path=root/'outputs/openai-standard-20261030/imported/caches/source-1'
    pairs_path=root/'outputs/paper-preflight-20261030/input-audit/preference-three-way.json'
    cache=DiskEmbeddingCache.load(cache_path)
    ids=load_model_ids(root/sources['model.py']['path'],sources['model.py']['sha256'])
    arm_ids={arm:ids[name] for arm,name in ARM_NAMES.items()}
    records=json.loads(pairs_path.read_text(encoding='utf-8'))
    groups=partition_pairs(records,dict(zip(cache.ids,cache.prompt_hashes)),arm_ids)
    if {s:len(r) for s,r in groups.items()}!={'train':847,'calibration':211,'test':500}:
        raise ValueError('Unexpected three-way preference population')
    protocol=dict(version=1,kind='routellm-mf-local-matched500',source_commit=COMMIT,sources=sources,
        pair_file=str(pairs_path.relative_to(root)),pair_sha256=sha(pairs_path),
        cache_path=str(cache_path.relative_to(root)),cache_sha256=cache.fingerprint(),
        model_ids=arm_ids,num_models=len(ids),num_prompts=len(cache.ids),text_dim=cache.dimension,
        split_counts={s:len(r) for s,r in groups.items()},seeds=[7,42,2026],
        settings=dict(dim=128,batch_size=64,epochs=100,lr=.0003,weight_decay=.00001,noise_alpha=.1),
        selection='minimum noise-free calibration BCE, earliest epoch on exact tie; no final-test selection',
        adaptations=['matched847/211/500 instead of upstream95/5',
          'validation forward uses test=True (no Gaussian evaluation noise)',
          'best calibration loss checkpoint is saved, whereas upstream example only reports progress',
          'shared verified8-arm data; preserve64-row official model table',
          'local cached text-embedding-3-small only; no remote inference'],
        historical_test_overlap='500 re-split historical rows, not author-identical or project-wide unseen',
        source_code_sha256=sha(Path(__file__)),
        scoring_code_sha256=sha(root/'tools/openai_study_routellm.py'),inference_api_calls=0)
    base.mkdir(parents=True);write_json(base/'protocol.json',protocol)
    return protocol


def torch_environment():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    return torch


def train_one(torch,cls,vectors,train_rows,calibration_rows,settings,num_models,seed):
    """Final-test records are deliberately absent from this function's inputs."""
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=cls(dim=settings['dim'],num_models=num_models,num_prompts=len(vectors),
        text_dim=vectors.shape[1],num_classes=1,use_proj=True,npy_path=vectors.filename).to(device)
    # The official frozen Q initialization is retained; no text API is available.
    optimizer=torch.optim.Adam(model.parameters(),lr=settings['lr'],weight_decay=settings['weight_decay'])
    criterion=torch.nn.BCEWithLogitsLoss(reduction='mean')
    row_index={q:i for i,q in enumerate(vectors.query_ids)}
    def tensors(rows):
        return (torch.tensor([r[1] for r in rows],dtype=torch.long),
            torch.tensor([r[2] for r in rows],dtype=torch.long),
            torch.tensor([row_index[r[0]] for r in rows],dtype=torch.long))
    train=tensors(train_rows);calibration=tensors(calibration_rows)
    history=[];best_state=None;best_loss=math.inf
    for epoch in range(1,settings['epochs']+1):
        model.train();order=torch.randperm(len(train_rows));loss_sum=0.
        for start in range(0,len(order),settings['batch_size']):
            chosen=order[start:start+settings['batch_size']]
            output=model(*(x[chosen] for x in train),alpha=settings['noise_alpha']).reshape(-1)
            loss=criterion(output,torch.ones_like(output))
            optimizer.zero_grad();loss.backward();optimizer.step()
            loss_sum+=float(loss.detach().cpu())*len(chosen)
        model.eval()
        with torch.no_grad():
            logits=model(*calibration,test=True).reshape(-1)
            cal_loss=float(criterion(logits,torch.ones_like(logits)).cpu())
            correct=int((logits>0).sum().cpu())
        history.append(dict(epoch=epoch,train_loss=loss_sum/len(train_rows),
            calibration_loss=cal_loss,calibration_correct=correct))
        if cal_loss<best_loss:
            best_loss=cal_loss
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k!='Q.weight'}
        if epoch%20==0:print(f'seed={seed} epoch={epoch} calibration_loss={cal_loss:.6f}',flush=True)
    return best_state,history,best_epoch(history)


def evaluate_arrays(cache,rows,arm_ids,state):
    from tools.openai_study_routellm import mf_scores
    arms=list(arm_ids);models=np.asarray(state['P.weight'])[list(arm_ids.values())]
    W=np.asarray(state['text_proj.weight']);head=np.asarray(state['classifier.weight'])
    positions={a:i for i,a in enumerate(arm_ids.values())};predictions=[]
    for q,w,l in rows:
        scores=mf_scores(cache.get(q),models,W,head)
        difference=float(scores[positions[w]]-scores[positions[l]])
        predictions.append(dict(query_id=q,winner=arms[positions[w]],loser=arms[positions[l]],
            difference=difference,correct=difference>0))
    differences=np.array([r['difference'] for r in predictions])
    return dict(count=len(rows),correct=sum(r['correct'] for r in predictions),
        loss=float(np.logaddexp(0.,-differences).mean()),predictions=predictions)


def run(root,seed,*,mutation_probe=False):
    root=Path(root);base=root/'outputs/openai-followup-20261031/routellm'
    protocol=json.loads((base/'protocol.json').read_text(encoding='utf-8'))
    if sha(Path(__file__))!=protocol['source_code_sha256']:raise ValueError('Frozen trainer source changed')
    if sha(root/'tools/openai_study_routellm.py')!=protocol['scoring_code_sha256']:
        raise ValueError('Frozen MF scoring source changed')
    if seed not in protocol['seeds']:raise ValueError('Undeclared seed')
    preflight=require_resources()
    destination=base/(f'mutation-seed-{seed}' if mutation_probe else f'seed-{seed}')
    if destination.exists():raise FileExistsError(destination)
    cache=DiskEmbeddingCache.load(root/protocol['cache_path'])
    if cache.fingerprint()!=protocol['cache_sha256']:raise ValueError('Preference cache changed')
    pair_file=root/protocol['pair_file']
    if sha(pair_file)!=protocol['pair_sha256']:raise ValueError('Preference split changed')
    records=json.loads(pair_file.read_text(encoding='utf-8'))
    if mutation_probe:
        for row in records:
            if row['split']=='test':row['winner'],row['loser']=row['loser'],row['winner']
    groups=partition_pairs(records,dict(zip(cache.ids,cache.prompt_hashes)),protocol['model_ids'])
    attempts=[]
    def denied(*args,**kwargs):
        attempts.append(str(args[1:]));raise RuntimeError('Network disabled for cache-only MF training')
    socket.socket.connect=denied
    torch=torch_environment()
    source=protocol['sources']['train_matrix_factorization.py']
    cls=load_training_class(root/source['path'],source['sha256'],
        dict(torch=torch,np=np,nn=torch.nn,F=torch.nn.functional))
    vectors=np.load(root/protocol['cache_path']/'vectors.npy',mmap_mode='r',allow_pickle=False)
    vectors.query_ids=list(cache.ids)
    destination.mkdir()
    state,history,selected=train_one(torch,cls,vectors,groups['train'],groups['calibration'],
        protocol['settings'],protocol['num_models'],seed)
    torch.save(state,destination/'weights.pt')
    np.savez(destination/'weights.npz',**{k:v.numpy() for k,v in state.items()})
    state_arrays={k:v.numpy() for k,v in state.items()}
    results={split:evaluate_arrays(cache,rows,protocol['model_ids'],state_arrays) for split,rows in groups.items()}
    selection=dict(seed=seed,selected_epoch=selected,history=history,protocol_sha256=sha(base/'protocol.json'),
        weights_sha256=sha(destination/'weights.npz'),mutated_test_labels=mutation_probe,
        selected_calibration=next(r for r in history if r['epoch']==selected),
        environment=dict(torch=torch.__version__,device='cuda' if torch.cuda.is_available() else 'cpu',
            gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,threads=1),
        preflight=preflight,network_attempts=attempts,inference_api_calls=0)
    if attempts:raise RuntimeError('Unexpected network attempt')
    write_json(destination/'selection.json',selection);write_json(destination/'predictions.json',results)
    print(json.dumps(dict(seed=seed,selected_epoch=selected,
        scores={s:dict(correct=r['correct'],count=r['count']) for s,r in results.items()})),flush=True)
    return selection


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run'])
    parser.add_argument('--root',type=Path,default=Path.cwd());parser.add_argument('--seed',type=int)
    parser.add_argument('--mutation-probe',action='store_true');args=parser.parse_args()
    if args.action=='freeze':print(json.dumps(freeze(args.root),indent=2))
    else:run(args.root,args.seed,mutation_probe=args.mutation_probe)


if __name__=='__main__':main()

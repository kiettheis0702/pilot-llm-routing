"""Local pinned Instructor-XL: full length audit, resumable encoding, aligned exports."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import numpy as np

from scripts.data_gate import query_hash
from scripts.embedding_store import DiskEmbeddingCache
from scripts.pretraining import _digest
from scripts.resources import require_resources
from tools.openai_study_instructor import MODEL,REVISION,INSTRUCTION,MAX_LENGTH,format_instructor_input,pool_dense_normalize
from tools.openai_study_instructor_journal import EmbeddingJournal

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/openai-followup-20261031/instructor'
INPUTS=ROOT/'outputs/research-input-audit-20261029/inputs'


def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def raw_sha(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def once(p,value):
    if p.exists():
        if read(p)!=value:raise ValueError('Immutable encoding artifact differs: '+str(p))
    else:
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x',encoding='utf-8') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)


def build_index(roles,lengths):
    unique={};seen=set()
    for rows in roles.values():
        for row in rows:
            q,text,h=(row[k] for k in ('query_id','text','prompt_hash'))
            key=raw_sha(text)
            if (q in seen or not isinstance(q,str) or not q or not text.strip() or query_hash(text)!=h
                    or key not in lengths or type(lengths[key]) is not int or lengths[key]<1):
                raise ValueError('Input identity or token length missing/invalid')
            seen.add(q)
            unique.setdefault(key,dict(query_id='utf8:'+key,prompt_hash=h,raw_sha256=key,token_length=lengths[key]))
    ordered=sorted(unique.values(),key=lambda r:r['token_length'])
    position={r['raw_sha256']:i for i,r in enumerate(ordered)}
    return dict(version=1,unique=ordered,roles={role:[dict(query_id=r['query_id'],prompt_hash=r['prompt_hash'],
        raw_sha256=raw_sha(r['text']),unique_index=position[raw_sha(r['text'])]) for r in rows] for role,rows in roles.items()},
        compute_order='stable ascending full token length; original role order restored on export')


def publish_role_caches(unique,index,destination):
    if (unique.ids!=tuple(r['query_id'] for r in index['unique'])
            or unique.prompt_hashes!=tuple(r['prompt_hash'] for r in index['unique'])):
        raise ValueError('Completed unique cache does not match the declared input index')
    destination=Path(destination);result={}
    for role,rows in index['roles'].items():
        if role not in ('source-0','source-1'):raise ValueError('Unknown embedding role')
        folder=destination/role;metadata=dict(unique_cache_sha256=unique.fingerprint(),index_sha256=_digest(index),role=role)
        ids=[r['query_id'] for r in rows];hashes=[r['prompt_hash'] for r in rows]
        positions=[r['unique_index'] for r in rows]
        if (folder/'manifest.json').exists():
            cache=DiskEmbeddingCache.load(folder)
            if cache.ids!=tuple(ids) or cache.prompt_hashes!=tuple(hashes) or cache.metadata!=metadata or cache.provenance!=unique.provenance:
                raise ValueError('Existing role cache identity mismatch')
        else:
            folder.mkdir(parents=True,exist_ok=True)
            if any(p.name!='vectors.npy' for p in folder.iterdir()):raise ValueError('Unexpected partial role-cache files')
            values=np.lib.format.open_memmap(folder/'vectors.npy',mode='w+',dtype=np.float32,shape=(len(ids),unique.dimension))
            for start in range(0,len(ids),256):values[start:start+256]=unique._vectors[positions[start:start+256]]
            values.flush();values._mmap.close()
            cache=DiskEmbeddingCache.publish(folder,ids,hashes,unique.provenance,metadata=metadata)
        result[role]=cache
    return result


def load_inputs():
    manifest=read(INPUTS/'manifest.json')
    if manifest['manifest_sha256']!=_digest({k:v for k,v in manifest.items() if k!='manifest_sha256'}):
        raise ValueError('Input manifest checksum mismatch')
    roles={};pins={}
    for role,name in [('source-0','routing-inputs.json'),('source-1','preference-inputs.json')]:
        if sha(INPUTS/name)!=manifest['files'][name]:raise ValueError('Input text file changed')
        roles[role]=read(INPUTS/name);pins[str((INPUTS/name).relative_to(ROOT))]=manifest['files'][name]
    pins[str((INPUTS/'manifest.json').relative_to(ROOT))]=sha(INPUTS/'manifest.json')
    texts={}
    for rows in roles.values():
        for row in rows:
            key=raw_sha(row['text'])
            if key in texts and texts[key]!=row['text']:raise ValueError('Raw text hash collision')
            texts[key]=row['text']
    return roles,texts,pins


def offline_environment():
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    os.environ.update(TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    attempts=[]
    def denied(*args,**kwargs):attempts.append(str(args[1:]));raise RuntimeError('Instructor inference network is disabled')
    socket.socket.connect=denied
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    return torch,attempts


def load_encoder(torch,local):
    from transformers import AutoTokenizer,T5EncoderModel
    tokenizer=AutoTokenizer.from_pretrained(local,local_files_only=True,use_fast=False,legacy=True,trust_remote_code=False)
    model=T5EncoderModel.from_pretrained(local,local_files_only=True,dtype=torch.float32).eval().to('cuda')
    model.requires_grad_(False)
    state=torch.load(local/'2_Dense/pytorch_model.bin',map_location='cpu',weights_only=True)
    if set(state)!={'linear.weight'}:raise ValueError('Unexpected Instructor dense state')
    return tokenizer,model,state['linear.weight'].float().to('cuda')


def encode_batch(torch,tokenizer,model,weight,texts,*,independent=False):
    features=tokenizer([format_instructor_input(INSTRUCTION,t) for t in texts],return_tensors='pt',
        padding=True,truncation=True,max_length=MAX_LENGTH)
    prefix=len(tokenizer(INSTRUCTION.strip(),add_special_tokens=True)['input_ids'])-1
    mask=features['attention_mask'].clone();mask[:,:prefix]=0
    if bool((mask.sum(1)<=0).any()):raise ValueError('No question tokens after instruction masking')
    with torch.inference_mode():
        hidden=model(**{k:v.to('cuda') for k,v in features.items()}).last_hidden_state
        active=mask.to('cuda').float()
        pooled=(hidden*active[...,None]).sum(1)/active.sum(1,keepdim=True)
        actual=torch.nn.functional.normalize(pooled@weight.T,p=2,dim=1).cpu().numpy()
    difference=None
    if independent:
        expected=pool_dense_normalize(hidden.cpu().numpy(),features['attention_mask'].numpy(),[prefix]*len(texts),weight.cpu().numpy())
        np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
        difference=float(np.max(np.abs(actual-expected)))
    return actual,difference


def prepare():
    preflight=require_resources()
    if BASE.exists():raise FileExistsError(BASE)
    roles,texts,pins=load_inputs()
    download=read(ROOT/'outputs/openai-followup-20261031/instructor-download.json')
    if download['revision']!=REVISION:raise ValueError('Instructor revision changed')
    local=Path(download['local_path']);old_probe=read(ROOT/'outputs/openai-followup-20261031/instructor-probe.json')
    for name,want in old_probe['weights_sha256'].items():
        if sha(local/name)!=want:raise ValueError('Pinned model weights changed')
    for path in sorted(local.rglob('*')):
        if path.is_file():pins[str(path.relative_to(ROOT))]=sha(path)
    for path in (Path(__file__),ROOT/'tools/openai_study_instructor.py',ROOT/'tools/openai_study_instructor_journal.py',
            ROOT/'scripts/embedding_store.py',ROOT/'scripts/data_gate.py',ROOT/'scripts/pretraining.py'):
        pins[str(path.relative_to(ROOT))]=sha(path)
    torch,attempts=offline_environment()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(local,local_files_only=True,use_fast=False,legacy=True,trust_remote_code=False)
    lengths={}
    for n,(key,text) in enumerate(texts.items(),1):
        lengths[key]=len(tokenizer(format_instructor_input(INSTRUCTION,text),add_special_tokens=True,truncation=False)['input_ids'])
        if n%5000==0:print(f'Instructor token audit{n}/{len(texts)}',flush=True)
    index=build_index(roles,lengths)
    if len(index['unique'])!=38018 or {k:len(v) for k,v in roles.items()}!={'source-0':36497,'source-1':1558}:
        raise ValueError('Unexpected full input population')
    audit={role:dict(rows=len(rows),full_token_sum=sum(lengths[raw_sha(r['text'])] for r in rows),
        truncated_rows=sum(lengths[raw_sha(r['text'])]>MAX_LENGTH for r in rows),
        maximum_full_tokens=max(lengths[raw_sha(r['text'])] for r in rows)) for role,rows in roles.items()}
    tokenizer,model,weight=load_encoder(torch,local)
    probe_texts=[texts[r['raw_sha256']] for r in index['unique'][-4:]]
    started=time.monotonic();actual,difference=encode_batch(torch,tokenizer,model,weight,probe_texts,independent=True)
    torch.cuda.synchronize();elapsed=time.monotonic()-started
    assert actual.shape==(4,768) and not attempts
    probe=dict(verified=True,real_input_count=4,maximum_padded_length=MAX_LENGTH,dimension=768,
        full_lengths=[r['token_length'] for r in index['unique'][-4:]],max_independent_difference=difference,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=elapsed,
        gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,transformers_version=__import__('transformers').__version__,
        network_attempts=attempts,inference_api_calls=0,preflight=preflight)
    provenance=dict(model=MODEL,version=REVISION,source='Pinned public HuggingFace weights; local offline T5EncoderModel',
        normalization='instruction-masked mean; Dense1024->768; L2; max512; float32',synthetic=False)
    BASE.mkdir(parents=True,exist_ok=False)
    once(BASE/'input-index.json',index);once(BASE/'length-audit.json',audit);once(BASE/'batch4-probe.json',probe)
    protocol=dict(version=1,kind='pinned-local-instructor-encoding',revision=REVISION,
        model_path=str(local.relative_to(ROOT)),source_pins=pins,index_sha256=sha(BASE/'input-index.json'),
        audit_sha256=sha(BASE/'length-audit.json'),probe_sha256=sha(BASE/'batch4-probe.json'),
        provenance=provenance,instruction=INSTRUCTION,instruction_status='declared local assumption, not author-confirmed',
        max_length=MAX_LENGTH,dtype='float32',batch_size=4,chunk_size=64,dimension=768,
        unique_texts=38018,source_rows=38055,inference_api_calls=0)
    once(BASE/'encoding-protocol.json',protocol)
    print(json.dumps(dict(prepared=True,probe=probe,length_audit=audit),indent=2),flush=True)


def encode(limit_chunks=None):
    preflight=require_resources();protocol=read(BASE/'encoding-protocol.json')
    for path,want in protocol['source_pins'].items():
        if sha(ROOT/path)!=want:raise ValueError('Frozen Instructor source/input/model changed: '+path)
    for name,key in [('input-index.json','index_sha256'),('length-audit.json','audit_sha256'),('batch4-probe.json','probe_sha256')]:
        if sha(BASE/name)!=protocol[key]:raise ValueError('Encoding preparation changed')
    roles,texts,_=load_inputs();index=read(BASE/'input-index.json');unique=index['unique']
    metadata=dict(encoding_protocol_sha256=sha(BASE/'encoding-protocol.json'),index_sha256=protocol['index_sha256'])
    with EmbeddingJournal(BASE/'unique-cache',ids=[r['query_id'] for r in unique],prompt_hashes=[r['prompt_hash'] for r in unique],
            provenance=protocol['provenance'],dimension=protocol['dimension'],metadata=metadata) as writer:
        chunks=0;started=time.monotonic()
        if writer.next_row<len(unique):
            torch,attempts=offline_environment();tokenizer,model,weight=load_encoder(torch,ROOT/protocol['model_path'])
            for start in range(writer.next_row,len(unique),protocol['chunk_size']):
                end=min(start+protocol['chunk_size'],len(unique));batch=[]
                for offset in range(start,end,protocol['batch_size']):
                    selected=unique[offset:min(offset+protocol['batch_size'],end)]
                    actual,_=encode_batch(torch,tokenizer,model,weight,[texts[r['raw_sha256']] for r in selected],independent=chunks==0 and offset==start)
                    batch.append(actual)
                writer.commit(start,np.concatenate(batch,axis=0));chunks+=1
                progress=dict(completed=writer.next_row,total=len(unique),elapsed_seconds=time.monotonic()-started,
                    peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),network_attempts=len(attempts),inference_api_calls=0)
                with (BASE/'progress.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(progress)+'\n')
                print(json.dumps(progress),flush=True)
                if limit_chunks and chunks>=limit_chunks:return
            assert not attempts
        cache=writer.finish()
        exported=publish_role_caches(cache,index,BASE/'caches')
    once(BASE/'completion.json',dict(complete=True,unique_texts=len(unique),source_rows=sum(len(r) for r in roles.values()),
        dimension=768,protocol_sha256=sha(BASE/'encoding-protocol.json'),unique_cache_sha256=cache.fingerprint(),
        role_caches={k:dict(rows=len(v.ids),fingerprint=v.fingerprint()) for k,v in exported.items()},inference_api_calls=0))
    print('Instructor full cache completed and aligned',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','encode'])
    parser.add_argument('--limit-chunks',type=int);args=parser.parse_args()
    if args.action=='prepare':prepare()
    else:encode(args.limit_chunks)


if __name__=='__main__':main()

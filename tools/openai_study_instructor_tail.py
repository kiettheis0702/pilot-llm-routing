"""Approved recovery of only the unfinished Instructor prefix, with explicit overlay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from scripts.resources import require_resources
from tools.openai_study_instructor_encode import BASE,ROOT,load_inputs,load_encoder,offline_environment,once,publish_role_caches,read,sha
from tools.openai_study_instructor import format_instructor_input,pool_dense_normalize
from tools.openai_study_instructor_journal import EmbeddingJournal


def make_overlay(protocol,proof,producer_sha256):
    start=proof['committed_rows'];end=protocol['unique_texts']
    if (not 0<start<end or proof['remaining_rows']!=end-start or proof['dimension']!=protocol['dimension']
            or protocol['dtype']!='float32' or protocol['max_length']!=512):
        raise ValueError('Recovery scope differs from the verified parent prefix')
    return dict(version=1,kind='approved-local-instructor-tail-recovery',start_row=start,end_row=end,
        original_protocol_sha256=proof['protocol_sha256'],source_prefix_sha256=proof['prefix_vectors_sha256'],
        producer_sha256=producer_sha256,revision=protocol['revision'],instruction=protocol['instruction'],
        dimension=protocol['dimension'],max_length=512,dtype='float32',batch_size=1,padding='max_length',
        cache_release='after each committed chunk; no memory cap',
        authorization='User explicitly approved stopping only the stalled encoder and resuming450 vectors from cache',
        scope='Original prefix unchanged; same weights/input/normalization; runtime batching overlay, not hidden protocol mutation',
        inference_api_calls=0)


def verify_original_prefix(cache_path,proof):
    matrix=np.load(Path(cache_path)/'vectors.npy',mmap_mode='r',allow_pickle=False)
    count=proof['committed_rows'];hasher=hashlib.sha256()
    if not 0<count<=len(matrix) or matrix.ndim!=2 or matrix.shape[1]!=proof['dimension']:
        raise ValueError('Original prefix shape changed')
    for start in range(0,count,256):hasher.update(matrix[start:min(start+256,count)].tobytes())
    fingerprint=hasher.hexdigest()
    if fingerprint!=proof['prefix_vectors_sha256']:raise ValueError('Original committed prefix checksum changed')
    return fingerprint


def encode_one(torch,tokenizer,model,weight,text,protocol):
    features=tokenizer([format_instructor_input(protocol['instruction'],text)],return_tensors='pt',
        padding='max_length',truncation=True,max_length=512)
    prefix=len(tokenizer(protocol['instruction'].strip(),add_special_tokens=True)['input_ids'])-1
    mask=features['attention_mask'].clone();mask[:,:prefix]=0
    if bool((mask.sum(1)<=0).any()):raise ValueError('No question tokens after instruction masking')
    with torch.inference_mode():
        hidden=model(**{k:v.to('cuda') for k,v in features.items()}).last_hidden_state
        active=mask.to('cuda').float()
        pooled=(hidden*active[...,None]).sum(1)/active.sum(1,keepdim=True)
        actual=torch.nn.functional.normalize(pooled@weight.T,p=2,dim=1).cpu().numpy()
    independent=pool_dense_normalize(hidden.cpu().numpy(),features['attention_mask'].numpy(),[prefix],weight.cpu().numpy())
    np.testing.assert_allclose(actual,independent,rtol=1e-5,atol=1e-6)
    return actual,float(np.max(np.abs(actual-independent)))


def run(limit_chunks=None):
    preflight=require_resources();protocol=read(BASE/'encoding-protocol.json');proof=read(BASE/'stall-prefix-proof.json')
    if proof['protocol_sha256']!=sha(BASE/'encoding-protocol.json') or proof['index_sha256']!=sha(BASE/'input-index.json'):
        raise ValueError('Parent recovery pins changed')
    for path,fingerprint in protocol['source_pins'].items():
        if sha(ROOT/path)!=fingerprint:raise ValueError('Frozen model/input/code changed before recovery')
    verify_original_prefix(BASE/'unique-cache',proof)
    overlay=make_overlay(protocol,proof,sha(Path(__file__)));once(BASE/'recovery-overlay.json',overlay)
    roles,texts,_=load_inputs();index=read(BASE/'input-index.json');unique=index['unique']
    metadata=dict(encoding_protocol_sha256=sha(BASE/'encoding-protocol.json'),index_sha256=protocol['index_sha256'])
    with EmbeddingJournal(BASE/'unique-cache',ids=[r['query_id'] for r in unique],prompt_hashes=[r['prompt_hash'] for r in unique],
            provenance=protocol['provenance'],dimension=768,metadata=metadata) as writer:
        if writer.next_row<overlay['start_row']:raise ValueError('Committed recovery prefix was lost')
        start_row=writer.next_row;chunks=0;started=time.monotonic();maximum=0.
        if writer.next_row<len(unique):
            torch,attempts=offline_environment();tokenizer,model,weight=load_encoder(torch,ROOT/protocol['model_path'])
            for start in range(writer.next_row,len(unique),protocol['chunk_size']):
                end=min(start+protocol['chunk_size'],len(unique));values=[]
                for position in range(start,end):
                    actual,difference=encode_one(torch,tokenizer,model,weight,texts[unique[position]['raw_sha256']],protocol)
                    maximum=max(maximum,difference);values.append(actual)
                writer.commit(start,np.concatenate(values,axis=0));chunks+=1
                progress=dict(completed=writer.next_row,total=len(unique),recovered_this_invocation=writer.next_row-start_row,
                    elapsed_seconds=time.monotonic()-started,max_numpy_difference=maximum,
                    peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),inference_api_calls=0)
                with (BASE/'recovery-progress.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(progress)+'\n')
                print(json.dumps(progress),flush=True);torch.cuda.empty_cache()
                if limit_chunks and chunks>=limit_chunks:return
            if attempts:raise RuntimeError('Unexpected recovery network attempt')
        verify_original_prefix(BASE/'unique-cache',proof)
        cache=writer.finish();exported=publish_role_caches(cache,index,BASE/'caches')
    once(BASE/'completion.json',dict(complete=True,unique_texts=len(unique),source_rows=sum(len(v) for v in roles.values()),
        dimension=768,protocol_sha256=sha(BASE/'encoding-protocol.json'),recovery_overlay_sha256=sha(BASE/'recovery-overlay.json'),
        original_prefix_unchanged_sha256=proof['prefix_vectors_sha256'],unique_cache_sha256=cache.fingerprint(),
        role_caches={k:dict(rows=len(v.ids),fingerprint=v.fingerprint()) for k,v in exported.items()},inference_api_calls=0))
    print('Instructor recovery complete; prefix unchanged, role caches aligned',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--limit-chunks',type=int);args=parser.parse_args();run(args.limit_chunks)

"""Full cache/alignment audit plus pinned official pooling/token-mask GPU replay.

Never imports the encoder, its pooling helper, or its journal implementation.
"""
if not __debug__:raise RuntimeError('Independent verification requires assertions')

import ast
import hashlib
import json
import os
from pathlib import Path
import socket
from typing import Union,List,Dict

import numpy as np

from scripts.data_gate import query_hash
from scripts.embedding_store import DiskEmbeddingCache
from scripts.resources import require_resources

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/openai-followup-20261031/instructor'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    preflight=require_resources();protocol=read(BASE/'encoding-protocol.json');completion=read(BASE/'completion.json')
    assert completion['complete'] and completion['protocol_sha256']==sha(BASE/'encoding-protocol.json')
    for name,want in protocol['source_pins'].items():assert sha(ROOT/name)==want
    index=read(BASE/'input-index.json');assert sha(BASE/'input-index.json')==protocol['index_sha256']
    unique=DiskEmbeddingCache.load(BASE/'unique-cache')
    assert unique.fingerprint()==completion['unique_cache_sha256'] and unique.dimension==768
    assert unique.ids==tuple(r['query_id'] for r in index['unique'])
    assert unique.prompt_hashes==tuple(r['prompt_hash'] for r in index['unique'])
    assert unique.provenance==protocol['provenance'] and unique.provenance['synthetic'] is False
    overlay=None
    if 'recovery_overlay_sha256' in completion:
        assert sha(BASE/'recovery-overlay.json')==completion['recovery_overlay_sha256']
        overlay=read(BASE/'recovery-overlay.json')
        assert overlay['original_protocol_sha256']==completion['protocol_sha256']
        assert overlay['producer_sha256']==sha(ROOT/'tools/openai_study_instructor_tail.py')
        assert (overlay['start_row'],overlay['end_row'],overlay['batch_size'],overlay['padding'],overlay['dtype'],overlay['max_length'])==(37568,38018,1,'max_length','float32',512)
        hasher=hashlib.sha256()
        for start in range(0,overlay['start_row'],256):hasher.update(unique._vectors[start:min(start+256,overlay['start_row'])].tobytes())
        assert hasher.hexdigest()==overlay['source_prefix_sha256']==completion['original_prefix_unchanged_sha256']
    for _,batch in unique.iter_vectors(256):
        np.testing.assert_allclose(np.linalg.norm(batch.astype(float),axis=1),1.,atol=1e-6)
    end=0;chunks=0
    state=read(BASE/'unique-cache/journal-state.json');assert digest(state['payload'])==state['sha256']
    for path in sorted((BASE/'unique-cache').glob('chunk-*.json')):
        wrapped=read(path);assert digest(wrapped['payload'])==wrapped['sha256'];row=wrapped['payload']
        assert row['identity']==state['sha256'] and row['start']==end and row['start']<row['end']<=38018
        assert hashlib.sha256(unique._vectors[row['start']:row['end']].tobytes()).hexdigest()==row['vector_sha256']
        end=row['end'];chunks+=1
    assert end==38018 and len(unique.ids)==38018
    original_inputs=ROOT/'outputs/research-input-audit-20261029/inputs'
    raw_texts={};roles={};row_count=0
    for role,name in [('source-0','routing-inputs.json'),('source-1','preference-inputs.json')]:
        rows=read(original_inputs/name);roles[role]=rows;row_count+=len(rows)
        cache=DiskEmbeddingCache.load(BASE/'caches'/role)
        assert cache.fingerprint()==completion['role_caches'][role]['fingerprint']
        assert cache.metadata['unique_cache_sha256']==unique.fingerprint() and cache.metadata['index_sha256']==digest(index)
        assert cache.ids==tuple(r['query_id'] for r in rows) and cache.prompt_hashes==tuple(r['prompt_hash'] for r in rows)
        assert cache.provenance==unique.provenance
        assert len(index['roles'][role])==len(rows)==completion['role_caches'][role]['rows']
        for start in range(0,len(rows),256):
            group=index['roles'][role][start:start+256]
            np.testing.assert_array_equal(cache._vectors[start:start+256],unique._vectors[[r['unique_index'] for r in group]])
        for row,record in zip(rows,index['roles'][role],strict=True):
            raw=hashlib.sha256(row['text'].encode('utf-8')).hexdigest()
            assert record['query_id']==row['query_id'] and record['prompt_hash']==row['prompt_hash']==query_hash(row['text'])
            assert record['raw_sha256']==raw
            assert index['unique'][record['unique_index']]['raw_sha256']==raw
            if raw in raw_texts:assert raw_texts[raw]==row['text']
            raw_texts[raw]=row['text']
    assert row_count==38055 and len(raw_texts)==38018
    assert set(raw_texts)=={r['raw_sha256'] for r in index['unique']}
    assert all(r['query_id']=='utf8:'+r['raw_sha256'] for r in index['unique'])
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_TELEMETRY='1')
    attempts=[]
    def denied(*args,**kwargs):attempts.append(str(args[1:]));raise RuntimeError('Independent Instructor inference is offline')
    socket.socket.connect=denied
    import torch
    from transformers import AutoTokenizer,T5EncoderModel
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    local=ROOT/protocol['model_path']
    assert read(local/'sentence_bert_config.json')=={'max_seq_length':512,'do_lower_case':False}
    dense_config=read(local/'2_Dense/config.json')
    assert dense_config=={'in_features':1024,'out_features':768,'bias':False,'activation_function':'torch.nn.modules.linear.Identity'}
    tokenizer=AutoTokenizer.from_pretrained(local,local_files_only=True,use_fast=False,legacy=True,trust_remote_code=False)
    instruction=protocol['instruction'].strip();lengths={}
    for n,(raw,text) in enumerate(raw_texts.items(),1):
        lengths[raw]=len(tokenizer(instruction+text.strip(),add_special_tokens=True,truncation=False)['input_ids'])
        if n%10000==0:print('Independent Instructor length audit',n,flush=True)
    assert [r['token_length'] for r in index['unique']]==[lengths[r['raw_sha256']] for r in index['unique']]
    assert [r['token_length'] for r in index['unique']]==sorted(lengths.values())
    audit=read(BASE/'length-audit.json')
    for role,rows in roles.items():
        values=[lengths[hashlib.sha256(r['text'].encode('utf-8')).hexdigest()] for r in rows]
        assert audit[role]==dict(rows=len(rows),full_token_sum=sum(values),truncated_rows=sum(v>512 for v in values),maximum_full_tokens=max(values))
    # Execute only reviewed official pooling ClassDef and static mask method;
    # no SentenceTransformer/InstructorEmbedding module imports are executed.
    sources=ROOT/'outputs/openai-followup-20261031/sources';manifest=read(sources/'manifest.json')
    record=next(r for r in manifest['records'] if r.get('path')=='xlang-ai__instructor-embedding/code/InstructorEmbedding/instructor.py')
    assert sha(sources/record['path'])==record['sha256']
    tree=ast.parse((sources/record['path']).read_text(encoding='utf-8'))
    pooling=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='INSTRUCTORPooling')
    instructor=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='INSTRUCTOR')
    mask_method=next(n for n in instructor.body if isinstance(n,ast.FunctionDef) and n.name=='prepare_input_features')
    mask_method.decorator_list=[]
    namespace=dict(__name__='pinned_official_instructor',torch=torch,nn=torch.nn,Tensor=torch.Tensor,Union=Union,List=List,Dict=Dict,os=os,json=json)
    exec(compile(ast.Module(body=[pooling,mask_method],type_ignores=[]),'pinned_official_instructor','exec'),namespace)
    pool=namespace['INSTRUCTORPooling'](word_embedding_dimension=1024)
    model=T5EncoderModel.from_pretrained(local,local_files_only=True,dtype=torch.float32).eval().to('cuda');model.requires_grad_(False)
    weight=torch.load(local/'2_Dense/pytorch_model.bin',map_location='cpu',weights_only=True)['linear.weight'].float().to('cuda')
    selected=sorted({round(i*(len(unique.ids)-1)/32) for i in range(33)}|{63,64})
    if overlay is not None:selected=sorted(set(selected)|{overlay['start_row']-1,overlay['start_row']})
    replay=[];maximum=0.
    for start in range(0,len(selected),4):
        positions=selected[start:start+4];texts=[raw_texts[index['unique'][i]['raw_sha256']] for i in positions]
        features=tokenizer([instruction+text.strip() for text in texts],padding='max_length',truncation='longest_first',max_length=512,return_tensors='pt')
        instruction_features=tokenizer([instruction]*len(texts),padding='max_length',truncation='longest_first',max_length=512,return_tensors='pt')
        features=namespace['prepare_input_features'](features,instruction_features)
        with torch.inference_mode():
            hidden=model(input_ids=features['input_ids'].to('cuda'),attention_mask=features['attention_mask'].to('cuda')).last_hidden_state
            pooled=pool(dict(token_embeddings=hidden,attention_mask=(features['attention_mask']*features['instruction_mask']).to('cuda')))['sentence_embedding']
            expected=torch.nn.functional.normalize(torch.nn.functional.linear(pooled,weight),p=2,dim=1).cpu().numpy()
        actual=np.asarray(unique._vectors[positions])
        np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=3e-5)
        difference=float(np.max(np.abs(actual-expected)));maximum=max(maximum,difference)
        replay.append(dict(indices=positions,max_absolute_difference=difference))
    assert not attempts
    proof=dict(verified=True,complete=True,unique_texts=38018,source_rows=38055,dimension=768,journal_chunks=chunks,
        all_committed_vectors_hashed=True,all_role_vectors_equal=True,all_token_lengths_recomputed=True,
        official_source=record,official_pool_and_instruction_mask_replay_samples=len(selected),
        replay=replay,max_official_replay_difference=maximum,
        padding_comparison='production dynamic right padding versus official fixed512; masked outputs agree within declared float tolerance',
        tolerance=dict(atol=3e-6,rtol=3e-5),completion_sha256=sha(BASE/'completion.json'),
        recovery_overlay_sha256=completion.get('recovery_overlay_sha256'),
        verifier_sha256=sha(__file__),preflight=preflight,network_attempts=attempts,inference_api_calls=0)
    with (BASE/'independent-verification.json').open('x',encoding='utf-8') as stream:json.dump(proof,stream,indent=2)
    print(json.dumps(dict(verified=True,all_rows=row_count,replay_samples=len(selected),max_official_difference=maximum)),flush=True)


if __name__=='__main__':main()

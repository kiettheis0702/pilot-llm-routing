"""Local BGE-M3 dense encoding; resumable bounded batches, no paid API path."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from scripts.data_gate import query_hash
from scripts.embedding_store import DiskEmbeddingCache, vector_file
from scripts.pretraining import _digest, _names, _hashes
from scripts.resources import require_resources

MODEL = 'BAAI/bge-m3'
REVISION = '5617a9f61b028005a4858fdac845db406aefb181'


def validate_recipe(recipe):
    expected = {'model','revision','dimension','max_length','pooling','normalization',
                'inference_dtype','prompt_construction'}
    if (not isinstance(recipe,dict) or set(recipe)!=expected or recipe['model']!=MODEL
            or recipe['revision']!=REVISION or recipe['dimension']!=1024
            or type(recipe['max_length']) is not int or not 8<=recipe['max_length']<=8192
            or recipe['pooling']!='cls' or recipe['normalization']!='l2'
            or recipe['inference_dtype'] not in ('float16','float32')
            or recipe['prompt_construction']!='verbatim-no-instruction'):
        raise ValueError('Unsupported or unpinned local embedding recipe')
    return dict(recipe)


def provenance(recipe):
    recipe=validate_recipe(recipe)
    return dict(model=recipe['model'],version=recipe['revision'],
                source=json.dumps(recipe,sort_keys=True,separators=(',',':')),
                normalization='L2 float32 output',synthetic=False)


def validate_rows(rows):
    if not isinstance(rows,list) or not rows:
        raise ValueError('Nonempty input rows required')
    for row in rows:
        if (not isinstance(row,dict) or set(row)!={'query_id','prompt_hash','text'}
                or not isinstance(row['text'],str) or not row['text'].strip()
                or query_hash(row['text'])!=row['prompt_hash']):
            raise ValueError('Invalid text or prompt hash')
    return _names([r['query_id'] for r in rows],'query IDs'),_hashes([r['prompt_hash'] for r in rows])


def _batch_hash(matrix):
    return hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()


def _save_progress(path,state):
    temporary=path/'progress.tmp'
    temporary.write_text(json.dumps(state,sort_keys=True,allow_nan=False),encoding='utf-8')
    temporary.replace(path/'progress.json')


def encode_rows(rows,path,recipe,encoder,*,batch_size=2,resume=False,checkpoint_batches=1):
    path=Path(path);recipe=validate_recipe(recipe);ids,hashes=validate_rows(rows)
    if type(batch_size) is not int or not 1<=batch_size<=32:
        raise ValueError('Batch size must be 1..32')
    if type(checkpoint_batches) is not int or not 1<=checkpoint_batches<=64:
        raise ValueError('Checkpoint interval must be 1..64 batches')
    identity=_digest({'rows':rows,'recipe':recipe})
    if path.exists():
        if not resume:raise FileExistsError(path)
        if (path/'manifest.json').exists():
            cache=DiskEmbeddingCache.load(path)
            if cache.metadata.get('input_recipe_sha256')!=identity or cache.provenance!=provenance(recipe):
                raise ValueError('Cannot resume with changed inputs or recipe')
            return cache
        state=json.loads((path/'progress.json').read_text())
        if state.get('identity')!=identity or state.get('version')!=1:
            raise ValueError('Cannot resume with changed inputs or recipe')
        matrix=np.load(vector_file(path),mmap_mode='r+',allow_pickle=False)
        if matrix.shape!=(len(rows),1024) or matrix.dtype!=np.float32:
            raise ValueError('Invalid resume matrix')
        done=0
        for chunk in state['chunks']:
            if chunk['start']!=done or not done<chunk['end']<=len(rows):
                raise ValueError('Invalid resume progress')
            if _batch_hash(matrix[done:chunk['end']])!=chunk['sha256']:
                raise ValueError('Resume prefix checksum mismatch')
            done=chunk['end']
        if state['completed']!=done:raise ValueError('Invalid resume cursor')
    else:
        path.mkdir(parents=True,exist_ok=False)
        matrix=np.lib.format.open_memmap(path/'vectors.npy',mode='w+',dtype=np.float32,shape=(len(rows),1024))
        state=dict(version=1,identity=identity,completed=0,chunks=[],truncated_queries=0,
                   max_observed_tokens=0,total_original_tokens=0,encode_seconds=0.)
        _save_progress(path,state)
    checkpoint_start=state['completed'];pending_batches=0
    for start in range(state['completed'],len(rows),batch_size):
        texts=[r['text'] for r in rows[start:start+batch_size]]
        began=time.perf_counter()
        values,lengths=encoder(texts)
        values=np.asarray(values)
        if (values.shape!=(len(texts),1024) or values.dtype.kind not in 'f'
                or not np.isfinite(values).all() or len(lengths)!=len(texts)
                or any(type(n) is not int or n<1 for n in lengths)
                or not np.allclose(np.linalg.norm(values,axis=1),1.,rtol=1e-3,atol=1e-4)):
            raise ValueError('Invalid encoder output shape, norm, tokens or finite values')
        end=start+len(texts)
        matrix[start:end]=values.astype(np.float32)
        state['completed']=end
        state['truncated_queries']+=sum(n>recipe['max_length'] for n in lengths)
        state['max_observed_tokens']=max(state['max_observed_tokens'],*lengths)
        state['total_original_tokens']+=sum(lengths)
        state['encode_seconds']+=time.perf_counter()-began
        pending_batches+=1
        if pending_batches==checkpoint_batches or end==len(rows):
            matrix.flush()
            state['chunks'].append(dict(start=checkpoint_start,end=end,
                                        sha256=_batch_hash(matrix[checkpoint_start:end])))
            _save_progress(path,state)
            checkpoint_start=end;pending_batches=0
    del matrix
    metadata={key:state[key] for key in ('truncated_queries','max_observed_tokens','total_original_tokens','encode_seconds')}
    metadata.update(input_recipe_sha256=identity,recipe=recipe,rows=len(rows))
    return DiskEmbeddingCache.publish(path,ids,hashes,provenance(recipe),metadata=metadata)


class LocalBGEEncoder:
    def __init__(self,recipe,*,device='cuda',download=False,cache_dir='.cache/huggingface'):
        self.recipe=validate_recipe(recipe)
        self.preflight=require_resources()
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if device not in ('cpu','cuda'):raise ValueError('Device must be cpu or cuda')
        if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent fallback')
        if device=='cpu' and recipe['inference_dtype']!='float32':raise ValueError('CPU requires an explicit float32 recipe')
        location=snapshot_download(MODEL,revision=REVISION,cache_dir=cache_dir,token=False,
            local_files_only=not download,max_workers=1,allow_patterns=[
                'config.json','pytorch_model.bin','tokenizer.json','tokenizer_config.json',
                'sentencepiece.bpe.model','special_tokens_map.json'])
        require_resources()  # Memory may have changed during download.
        self.torch,self.device=torch,device
        self.tokenizer=AutoTokenizer.from_pretrained(location,local_files_only=True,trust_remote_code=False)
        dtype=torch.float16 if recipe['inference_dtype']=='float16' else torch.float32
        self.model=AutoModel.from_pretrained(location,local_files_only=True,trust_remote_code=False,
                                           torch_dtype=dtype,attn_implementation='sdpa').to(device).eval()
        self.runtime={'torch':torch.__version__,'device':device,
            'device_name':torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'}

    def __call__(self,texts):
        lengths=[len(ids) for ids in self.tokenizer(texts,truncation=False,return_attention_mask=False)['input_ids']]
        encoded=self.tokenizer(texts,padding=True,truncation=True,max_length=self.recipe['max_length'],return_tensors='pt')
        encoded={k:v.to(self.device) for k,v in encoded.items()}
        with self.torch.inference_mode():
            values=self.model(**encoded).last_hidden_state[:,0].float()
            values=self.torch.nn.functional.normalize(values,p=2,dim=1)
        return values.cpu().numpy(),lengths


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--recipe',type=Path,default=Path('configs/local_embedding.json'))
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--checkpoint-batches',type=int,default=1,
                        help='Persist a verified prefix every N batches (1..64); unfinished suffix is re-encoded')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--download',action='store_true',help='Allow download of pinned public model weights')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    os.environ['HF_HUB_DISABLE_XET']='1'
    rows=json.loads(args.inputs.read_text(encoding='utf-8'))
    recipe=validate_recipe(json.loads(args.recipe.read_text()))
    validate_rows(rows)
    encoder=None
    def lazy_encode(texts):
        nonlocal encoder
        if encoder is None:encoder=LocalBGEEncoder(recipe,device=args.device,download=args.download)
        return encoder(texts)
    cache=encode_rows(rows,args.output_dir,recipe,lazy_encode,batch_size=args.batch_size,
                      resume=args.resume,checkpoint_batches=args.checkpoint_batches)
    runtime={} if encoder is None else encoder.runtime
    if encoder is not None and encoder.device=='cuda':
        runtime['peak_cuda_allocated_bytes']=encoder.torch.cuda.max_memory_allocated()
        runtime['peak_cuda_reserved_bytes']=encoder.torch.cuda.max_memory_reserved()
    print(json.dumps({'output_dir':str(args.output_dir),'rows':len(cache.ids),
        'dimension':cache.dimension,'cache_sha256':cache.fingerprint(),'diagnostics':cache.metadata,'runtime':runtime}))


if __name__=='__main__':main()

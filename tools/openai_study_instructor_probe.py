"""One sequential local Instructor-XL compatibility/pooling probe; no network."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'

import hashlib
import json
import socket
from pathlib import Path

import numpy as np

from scripts.resources import require_resources
from tools.openai_study_instructor import INSTRUCTION, MAX_LENGTH, REVISION, format_instructor_input, pool_dense_normalize


def main():
    preflight=require_resources();root=Path.cwd();base=root/'outputs/openai-followup-20261031'
    download=json.loads((base/'instructor-download.json').read_text(encoding='utf-8'))
    assert download['revision']==REVISION
    local=Path(download['local_path'])
    weights={}
    for name in ('pytorch_model.bin','2_Dense/pytorch_model.bin'):
        with (local/name).open('rb') as stream:weights[name]=hashlib.file_digest(stream,'sha256').hexdigest()
    attempts=[]
    def denied(*args,**kwargs):
        attempts.append(str(args[1:]));raise RuntimeError('Network disabled during local Instructor probe')
    socket.socket.connect=denied
    import torch
    from transformers import AutoTokenizer, T5EncoderModel
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available()
    print('Loading pinned Instructor-XL locally; network denied',flush=True)
    tokenizer=AutoTokenizer.from_pretrained(local,local_files_only=True,use_fast=False,legacy=True)
    model=T5EncoderModel.from_pretrained(local,local_files_only=True,dtype=torch.float32).eval().to('cuda')
    model.requires_grad_(False)
    dense_state=torch.load(local/'2_Dense/pytorch_model.bin',map_location='cpu',weights_only=True)
    assert set(dense_state)=={'linear.weight'}
    weight=dense_state['linear.weight'].float().to('cuda')
    samples=['What is two plus two?','Explain why a contextual bandit needs exploration.', 'question '*900]
    text=[format_instructor_input(INSTRUCTION,t) for t in samples]
    lengths=[len(tokenizer(t,add_special_tokens=True)['input_ids']) for t in text]
    features=tokenizer(text,return_tensors='pt',padding=True,truncation=True,max_length=MAX_LENGTH)
    prefix=len(tokenizer(INSTRUCTION.strip(),add_special_tokens=True)['input_ids'])-1
    mask=features['attention_mask'].clone();mask[:,:prefix]=0
    with torch.inference_mode():
        hidden=model(**{k:v.to('cuda') for k,v in features.items()}).last_hidden_state
        active=mask.to('cuda').float()
        pooled=(hidden*active[...,None]).sum(1)/active.sum(1,keepdim=True)
        actual=torch.nn.functional.normalize(pooled@weight.T,p=2,dim=1).cpu().numpy()
    independent=pool_dense_normalize(hidden.cpu().numpy(),features['attention_mask'].numpy(),[prefix]*len(samples),weight.cpu().numpy())
    assert actual.shape==(3,768) and np.isfinite(actual).all()
    np.testing.assert_allclose(actual,independent,rtol=1e-5,atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(actual,axis=1),1.,atol=1e-6)
    assert not attempts
    result=dict(verified=True,diagnostic_only=True,synthetic_input_probe=True,revision=REVISION,
        dtype='float32',instruction=INSTRUCTION,max_length=MAX_LENGTH,output_dimension=768,
        input_token_lengths=lengths,instruction_prefix_tokens=prefix,
        max_pooling_difference=float(np.max(np.abs(actual-independent))),weights_sha256=weights,
        torch_version=torch.__version__,transformers_version=__import__('transformers').__version__,
        gpu=torch.cuda.get_device_name(),peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
        network_attempts=attempts,inference_api_calls=0,preflight=preflight)
    with (base/'instructor-probe.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

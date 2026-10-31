"""Independent NumPy readback of MF weights and all three preference splits.

Does not import the trainer, production MF scorer, or any inference/API SDK.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.embedding_store import DiskEmbeddingCache
from scripts.resources import require_resources

EXPECTED_IDS={'WizardLM/WizardLM-13B-V1.2':59,'claude-instant-v1':8,'claude-v1':5,
    'gpt-3.5-turbo-1106':20,'gpt-4-1106-preview':24,
    'meta/code-llama-instruct-34b-chat':9,'meta/llama-2-70b-chat':30,'zero-one-ai/Yi-34B-Chat':61}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify(root):
    root=Path(root);preflight=require_resources();base=root/'outputs/openai-followup-20261031/routellm'
    protocol=read(base/'protocol.json');cache=DiskEmbeddingCache.load(root/protocol['cache_path'])
    assert cache.fingerprint()==protocol['cache_sha256']
    assert sha(root/protocol['pair_file'])==protocol['pair_sha256']
    assert protocol['model_ids']==EXPECTED_IDS
    assert sha(root/'tools/openai_study_routellm_train.py')==protocol['source_code_sha256']
    assert sha(root/'tools/openai_study_routellm.py')==protocol['scoring_code_sha256']
    for record in protocol['sources'].values():assert sha(root/record['path'])==record['sha256']
    rows=read(root/protocol['pair_file']);ids=list(cache.ids);index={q:i for i,q in enumerate(ids)}
    texts=np.asarray(cache._vectors,dtype=np.float64);hashes=dict(zip(ids,cache.prompt_hashes))
    split_hashes={s:{hashes[r['query_id']] for r in rows if r['split']==s} for s in ('train','calibration','test')}
    assert all(not split_hashes[a]&split_hashes[b] for a,b in [('train','calibration'),('train','test'),('calibration','test')])
    summary=[];proofs={};array_sets={}
    for folder in [f'seed-{s}' for s in protocol['seeds']]+['mutation-seed-42']:
        directory=base/folder;selection=read(directory/'selection.json');actual=read(directory/'predictions.json')
        assert selection['protocol_sha256']==sha(base/'protocol.json')
        assert selection['weights_sha256']==sha(directory/'weights.npz')
        assert not selection['network_attempts'] and selection['inference_api_calls']==0
        with np.load(directory/'weights.npz',allow_pickle=False) as archive:
            state={k:archive[k].copy() for k in archive.files}
        assert set(state)=={'P.weight','text_proj.weight','classifier.weight'}
        P=state['P.weight'].astype(float);W=state['text_proj.weight'].astype(float);head=state['classifier.weight'].astype(float)
        assert P.shape==(64,128) and W.shape==(128,1536) and head.shape==(1,128)
        assert all(np.isfinite(a).all() for a in (P,W,head))
        P=P/np.sqrt(np.sum(P*P,axis=1,keepdims=True))
        # Batched scalar bilinear model; no call into the production scorer.
        score=(texts@W.T*head)@P.T
        scores={}
        for split in ('train','calibration','test'):
            local=[dict(r) for r in rows if r['split']==split]
            if folder.startswith('mutation') and split=='test':
                for r in local:r['winner'],r['loser']=r['loser'],r['winner']
            differences=np.array([score[index[r['query_id']],EXPECTED_IDS[r['winner']]]
                 -score[index[r['query_id']],EXPECTED_IDS[r['loser']]] for r in local])
            expected=actual[split]
            assert len(local)==expected['count']==protocol['split_counts'][split]
            assert int((differences>0).sum())==expected['correct']
            np.testing.assert_allclose(np.logaddexp(0.,-differences).mean(),expected['loss'],atol=1e-12,rtol=1e-12)
            for r,d,p in zip(local,differences,expected['predictions']):
                assert all(r[k]==p[k] for k in ('query_id','winner','loser'))
                np.testing.assert_allclose(d,p['difference'],atol=1e-11,rtol=1e-11)
                assert bool(d>0)==p['correct']
            scores[split]=dict(count=len(local),correct=int((differences>0).sum()),
                loss=float(np.logaddexp(0.,-differences).mean()))
        history=selection['history'];assert [r['epoch'] for r in history]==list(range(1,101))
        best=min(history,key=lambda r:(r['calibration_loss'],r['epoch']))
        assert selection['selected_epoch']==best['epoch']
        assert best['calibration_correct']==scores['calibration']['correct']
        np.testing.assert_allclose(best['calibration_loss'],scores['calibration']['loss'],atol=2e-6,rtol=1e-6)
        proofs[folder]=dict(selected_epoch=best['epoch'],scores=scores,
            selection_sha256=sha(directory/'selection.json'),weights_sha256=sha(directory/'weights.npz'))
        array_sets[folder]=state
        if not folder.startswith('mutation'):
            summary.append(dict(seed=selection['seed'],selected_epoch=best['epoch'],
                correct=scores['test']['correct'],pairs=500,accuracy=scores['test']['correct']/500))
    for key in array_sets['seed-42']:
        np.testing.assert_array_equal(array_sets['seed-42'][key],array_sets['mutation-seed-42'][key])
    assert proofs['seed-42']['scores']['test']['correct']+proofs['mutation-seed-42']['scores']['test']['correct']==500
    proof=dict(verified=True,complete=True,seeds=protocol['seeds'],proofs=proofs,
        actual_test_label_mutation_weights_identical=True,
        mutation_correct=proofs['mutation-seed-42']['scores']['test']['correct'],
        verifier_sha256=sha(Path(__file__)),protocol_sha256=sha(base/'protocol.json'),preflight=preflight)
    with (base/'independent-verification.json').open('x',encoding='utf-8') as out:json.dump(proof,out,indent=2)
    with (base/'per-seed.csv').open('x',encoding='utf-8',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    print(json.dumps(dict(verified=True,summary=summary,mutation_correct=proof['mutation_correct']),indent=2))
    return proof


if __name__=='__main__':verify(Path.cwd())

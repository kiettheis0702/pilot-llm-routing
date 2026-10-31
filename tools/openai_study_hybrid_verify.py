"""Independent Hybrid threshold accounting and raw source-token neural replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/openai-followup-20261031/hybrid'


def cheap_probabilities(logits):
    values = np.asarray(logits, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite N-by2 Hybrid logits required')
    scaled = np.exp(values - values.max(axis=1, keepdims=True))
    return scaled[:,0] / scaled.sum(axis=1)


def native_front(ids, logits, truth, arms, thresholds):
    ids = list(ids); arms = tuple(arms); probabilities = cheap_probabilities(logits)
    if (len(ids) != len(probabilities) or len(set(ids)) != len(ids) or len(arms) != 2
            or any((q,a) not in truth for q in ids for a in arms)):
        raise ValueError('Aligned unique identities and complete two-arm outcomes required')
    reference_reward = math.fsum(truth[q, arms[1]][0] for q in ids)
    reference_cost = math.fsum(truth[q, arms[1]][1] for q in ids)
    if reference_reward <= 0 or reference_cost <= 0:
        raise ValueError('Positive same-query strong-model reference required')
    rows = []
    for threshold in thresholds:
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('Threshold outside [0,1]')
        actions = [arms[0] if p >= threshold else arms[1] for p in probabilities]
        reward = math.fsum(truth[q,a][0] for q,a in zip(ids, actions))
        cost = math.fsum(truth[q,a][1] for q,a in zip(ids, actions))
        rows.append(dict(threshold=threshold, requested=len(ids), served=len(ids), coverage=1.,
            cheap_count=actions.count(arms[0]), strong_count=actions.count(arms[1]),
            total_reward=reward, service_utility=reward / len(ids), actual_recorded_cost=cost,
            gpt4_total_reward=reference_reward, gpt4_total_cost=reference_cost,
            quality_fraction_gpt4=reward / reference_reward, cost_fraction_gpt4=cost / reference_cost))
    return rows


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def ce(logits, labels):
    values = np.asarray(logits, dtype=float); labels = np.asarray(labels, dtype=np.int64)
    shifted = values - values.max(axis=1, keepdims=True)
    return float(np.mean(np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(labels)), labels]))


def verify(kind):
    from scripts.resources import require_resources
    preflight = require_resources()
    folder = BASE / kind; protocol = read(folder / 'protocol.json'); selection = read(folder / 'selection.json')
    canonical=json.dumps({k:v for k,v in protocol.items() if k!='manifest_sha256'},sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest()!=protocol['manifest_sha256']:
        raise ValueError('Independent Hybrid protocol checksum differs')
    destination = folder / 'independent-verification.json'
    if destination.exists():
        raise FileExistsError(destination)
    if selection['protocol_sha256'] != sha(folder / 'protocol.json') or selection['network_attempts'] != 0:
        raise ValueError('Hybrid fit protocol/network lineage differs')
    for record in protocol['source_pins'] + list(protocol['official_sources'].values()) + selection['epoch_checkpoints'] + [selection['predictions']]:
        if sha(ROOT / record['path']) != record['sha256']:
            raise ValueError('Hybrid source/checkpoint/prediction hash differs')
    from scripts.research_pipeline import load_routing
    import pandas as pd
    raw, queries, _, source_hashes = load_routing(ROOT / 'data')
    if source_hashes != protocol['source_hashes']:
        raise ValueError('Hybrid historical source hash differs')
    metadata = queries.set_index('query_id'); partitions = protocol['partitions']
    group_sets = [{metadata.loc[q,'query_hash'] for q in partitions[p]} for p in ('train','validation','test')]
    if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('Hybrid classifier prompt-group leakage')
    outcomes = pd.read_parquet(raw / 'outcomes.parquet', columns=['query_id','arm_id','reward','actual_cost'])
    arms = protocol['arms']
    truth = {(r.query_id,r.arm_id):(float(r.reward),float(r.actual_cost)) for r in outcomes.itertuples(index=False) if r.arm_id in arms}
    labels = np.asarray([0 if truth[q,arms[0]][0] >= truth[q,arms[1]][0] else 1 for q in partitions['validation']], dtype=np.int64)
    saved = np.load(ROOT / selection['predictions']['path'], allow_pickle=False)
    for phase in ('validation','test'):
        if saved[phase+'_ids'].tolist() != partitions[phase]:
            raise ValueError('Hybrid prediction identity/order differs')
        cheap_probabilities(saved[phase+'_logits'])
    declared = min(selection['history'], key=lambda r:(r['validation_loss'],r['epoch']))['epoch']
    if declared != selection['selected_epoch'] or len(selection['history']) != 5:
        raise ValueError('Hybrid calibration-only checkpoint selection differs')
    if selection['checkpoint'] != selection['epoch_checkpoints'][declared-1]:
        raise ValueError('Selected Hybrid checkpoint does not match declared epoch')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ.update(TOKENIZERS_PARALLELISM='false', CUBLAS_WORKSPACE_CONFIG=':4096:8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    attempts = []
    def deny(*args, **kwargs):
        attempts.append(1); raise RuntimeError('Network denied during independent Hybrid replay')
    socket.socket.connect = deny
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    assets = read(ROOT / 'outputs/openai-followup-20261031/deberta-asset-verification.json')
    local = Path(assets['local_path'])
    for name,fingerprint in assets['files'].items():
        if sha(local / name) != fingerprint:
            raise ValueError('Independent Hybrid backbone asset hash differs')
    tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True, use_fast=True)
    tokenizer.add_tokens(['<|source|>']); prefix_id = tokenizer.convert_tokens_to_ids('<|source|>')
    backbone = AutoModel.from_pretrained(local, local_files_only=True, dtype=torch.float32)
    backbone.resize_token_embeddings(len(tokenizer), mean_resizing=False); backbone.eval().to('cuda')
    text = metadata['query'].to_dict()
    def forward(ids, state):
        weight = state['head_layer.1.weight'].to('cuda'); bias = state['head_layer.1.bias'].to('cuda'); result = []
        with torch.inference_mode():
            for offset in range(0,len(ids),2):
                rendered = ['<|source|>' + text[q].replace(tokenizer.sep_token or tokenizer.eos_token,' ') for q in ids[offset:offset+2]]
                batch = tokenizer.batch_encode_plus(rendered,max_length=128,padding='max_length',truncation=True,return_tensors='pt')
                positions = torch.where(batch['input_ids'] == prefix_id)
                if len(positions[0]) != len(rendered):
                    raise ValueError('Independent Hybrid prefix count differs')
                with torch.autocast('cuda',dtype=torch.float16):
                    hidden = backbone(**{k:v.to('cuda') for k,v in batch.items()},output_hidden_states=True).hidden_states[-1]
                    features = hidden[positions[0].to('cuda'),positions[1].to('cuda'),:]
                    result.append(torch.nn.functional.linear(features,weight,bias).float().cpu().numpy())
        return np.concatenate(result)
    replay_losses = []; maximum_difference = 0.; sample_count = 0
    for epoch in range(1,6):
        record = selection['epoch_checkpoints'][epoch-1]
        if not record['path'].endswith(f'candidate-epoch-{epoch}.pt'):
            raise ValueError('Hybrid candidate checkpoint order differs')
        state = torch.load(ROOT / record['path'],map_location='cpu',weights_only=True)
        backbone.load_state_dict({k.removeprefix('pretrained_model.'):v for k,v in state.items() if k.startswith('pretrained_model.')},strict=True)
        actual = forward(partitions['validation'],state)
        loss = ce(actual,labels); replay_losses.append(loss)
        if not math.isclose(loss,selection['history'][epoch-1]['validation_loss'],rel_tol=1e-7,abs_tol=1e-7):
            raise ValueError('Independent candidate validation CE differs')
        if epoch == declared:
            maximum_difference = max(maximum_difference,float(np.max(np.abs(actual - saved['validation_logits']))))
            checked = forward(partitions['test'],state)
            maximum_difference = max(maximum_difference,float(np.max(np.abs(checked-saved['test_logits']))))
            sample_count = len(partitions['test'])
        del state
        print(f'Independent Hybrid {kind} epoch{epoch}/5 validationCE={loss:.8f}',flush=True)
    if min(range(5),key=lambda i:(replay_losses[i],i))+1 != declared or maximum_difference > 1e-5 or attempts:
        raise ValueError('Independent Hybrid checkpoint/neural output/network check failed')
    # Exercise the actual production input extractor against changed TEST
    # outcomes without writing data or refitting the costly neural model.
    from unittest.mock import patch
    from tools.openai_study_hybrid_train import training_data
    _,original_labels=training_data(protocol)
    original_reader=pd.read_parquet; test_ids=set(partitions['test']); changed=[0]
    def mutated_reader(path,*args,**kwargs):
        frame=original_reader(path,*args,**kwargs)
        if Path(path).name=='outcomes.parquet':
            mask=frame['query_id'].isin(test_ids)&frame['arm_id'].isin(arms)
            old=frame.loc[mask,'reward'].copy(); frame.loc[mask,'reward']=1.-old
            changed[0]+=int((old!=frame.loc[mask,'reward']).sum())
        return frame
    with patch.object(pd,'read_parquet',mutated_reader):
        _,mutated_labels=training_data(protocol)
    for phase in ('train','validation'):
        fingerprint=hashlib.sha256(original_labels[phase].tobytes()).hexdigest()
        if fingerprint!=selection['supervised_label_sha256'][phase] or not np.array_equal(original_labels[phase],mutated_labels[phase]):
            raise ValueError('Changed test labels leaked into actual Hybrid training/selection inputs')
    if changed[0]==0:
        raise ValueError('Hybrid held-out mutation negative control changed no labels')
    front = native_front(partitions['test'],saved['test_logits'],truth,arms,protocol['threshold_grid'])
    import csv
    with (folder / 'native-front.csv').open('x',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(front[0])); writer.writeheader(); writer.writerows(front)
    proof = dict(verified=True,cohort=kind,protocol_sha256=sha(folder/'protocol.json'),selection_sha256=sha(folder/'selection.json'),
        verifier_sha256=sha(__file__),all_epoch_validation_replayed=5,validation_queries_per_epoch=len(labels),
        independent_source_token_head=True,neural_test_replay_samples=sample_count,maximum_logit_difference=maximum_difference,
        all_test_logits_neurally_replayed=True,
        test_label_mutation=dict(changed=changed[0],actual_training_input_extractor=True,train_validation_arrays_identical=True,full_training_refit=False),
        independent_validation_losses=replay_losses,selected_epoch=declared,all_native_thresholds=len(front),
        native_front_sha256=sha(folder/'native-front.csv'),network_attempts=len(attempts),paid_api_calls=0,preflight=preflight)
    with destination.open('x',encoding='utf-8') as stream:
        json.dump(proof,stream,indent=2,allow_nan=False)
    print(json.dumps(proof),flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--cohort',choices=['binary','drift'],required=True)
    verify(parser.parse_args().cohort)


if __name__ == '__main__':
    main()

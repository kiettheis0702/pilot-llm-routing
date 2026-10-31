"""Local pinned HybridLLM source-token classifier; no paid API integration."""
from __future__ import annotations

import argparse
import ast
from enum import Enum
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
from types import SimpleNamespace
from typing import Dict, List

from tools.openai_study_hybrid import deterministic_labels, make_prompt_disjoint_stream, partition_classifier_training, select_validation_checkpoint

ROOT = Path(__file__).resolve().parents[1]
FOLLOWUP = ROOT / 'outputs/openai-followup-20261031'
BASE = FOLLOWUP / 'hybrid'
CHEAP = 'meta/llama-2-70b-chat'
STRONG = 'gpt-4-1106-preview'
COMMIT = 'b13c5bdaaf9a87af7587a9a83afc9b7219c2509d'
MODEL_REVISION = '64a8c8eab3e352a784c658aef62be1662607476f'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def sealed_protocol(path):
    value=read(path)
    if digest({k:v for k,v in value.items() if k!='manifest_sha256'})!=value['manifest_sha256']:
        raise ValueError('Hybrid protocol checksum differs')
    return value


def once(path, payload):
    path = Path(path)
    if path.exists():
        if read(path) != payload:
            raise ValueError('Immutable Hybrid artifact differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')


def pin(path):
    return dict(path=Path(path).relative_to(ROOT).as_posix(), sha256=sha(path))


def sources():
    config = read(ROOT / 'configs/followup_assets.json')
    wanted = ['hybrid_llm/pair_ranker/ranker.py', 'hybrid_llm/pair_ranker/collator.py', 'llm-blender-collator.py']
    result = {}
    for suffix in wanted:
        record = next(r for r in config['source_files'] if r['destination'].endswith(suffix))
        path = FOLLOWUP / record['destination']
        if sha(path) != record['sha256']:
            raise ValueError('Reviewed official Hybrid dependency changed')
        result[suffix] = dict(pin(path), url=record['url'])
    return result


def reviewed_definitions(record, names, namespace):
    path = ROOT / record['path']
    if sha(path) != record['sha256']:
        raise ValueError('Reviewed Hybrid source checksum differs')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if len(nodes) != len(names):
        raise ValueError('Missing/duplicate reviewed Hybrid definitions')
    scope = dict(namespace, __name__='reviewed_hybrid')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope


def runtime():
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ.update(TOKENIZERS_PARALLELISM='false', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    attempts = []
    def deny(*args, **kwargs):
        attempts.append(1)
        raise RuntimeError('Network disabled during local Hybrid training/inference')
    socket.socket.connect = deny
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise ValueError('The declared Hybrid recipe requires the local CUDA GPU')
    return torch, attempts


def environment(torch):
    import importlib.metadata
    return dict(packages={name:importlib.metadata.version(name) for name in
        ('torch','transformers','numpy','sentencepiece','tokenizers','safetensors','protobuf')},
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())


def build(torch, seed=42):
    from transformers import AutoModel, AutoTokenizer
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    assets = read(FOLLOWUP / 'deberta-asset-verification.json')
    if assets['revision'] != MODEL_REVISION:
        raise ValueError('Wrong pinned DeBERTa checkpoint')
    folder = Path(assets['local_path'])
    for name, fingerprint in assets['files'].items():
        if sha(folder / name) != fingerprint:
            raise ValueError('Hybrid backbone/tokenizer asset changed')
    tokenizer = AutoTokenizer.from_pretrained(folder, use_fast=True, local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError('Declared Hybrid AutoTokenizer fast implementation unavailable')
    official = sources()
    namespace = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, Enum=Enum, List=List, Dict=Dict)
    ranker = reviewed_definitions(official['hybrid_llm/pair_ranker/ranker.py'],
        {'NClassRerankerLossType', 'NClassReranker'}, namespace)['NClassReranker']
    encode = reviewed_definitions(official['llm-blender-collator.py'], {'encode_texts'}, namespace)['encode_texts']
    collator_cls = reviewed_definitions(official['hybrid_llm/pair_ranker/collator.py'], {'NClassCollator'},
        dict(namespace, PreTrainedTokenizer=object, encode_texts=encode, SOURCE_PREFIX='<|source|>'))['NClassCollator']
    collator = collator_cls(128, tokenizer, 128)
    backbone = AutoModel.from_pretrained(folder, local_files_only=True, dtype=torch.float32)
    backbone.config.out_hidden_state_size = backbone.config.hidden_size
    # Recreate the old random initialization for newly added tokens; newer
    # transformers defaults to mean-resizing, which the upstream era did not use.
    backbone.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    args = SimpleNamespace(n_tasks=1, num_pos=1, num_neg=1, n_candidates=2, match_t=0.,
        sub_sampling_mode='all_pair', sub_sampling_ratio=.5, loss_type='det_2cls',
        drop_out=.05, inference_mode='bubble')
    model = ranker(backbone, args, tokenizer).to('cuda')
    return model, tokenizer, collator


def probe():
    from scripts.resources import require_resources
    preflight = require_resources()
    torch, attempts = runtime()
    model, tokenizer, collator = build(torch)
    rows = [dict(source='What is two plus two?', scores=[[1.], [0.]], costs=[[.1], [1.]]),
            dict(source='Explain a contextual bandit. ' * 180, scores=[[0.], [1.]], costs=[[.1], [1.]])]
    batch = {k: v.to('cuda') for k, v in collator(rows).items()}
    if batch['source_ids'].shape != (2, 128) or not torch.all((batch['source_ids'] == tokenizer.source_prefix_id).sum(1) == 1):
        raise ValueError('Official source-token/fixed128 collation differs')
    if any(parameter.dtype!=torch.float32 for parameter in model.parameters()):
        raise ValueError('Declared FP32 Hybrid parameters required')
    fixed=torch.tensor([[2.,-1.],[-1.,2.]],device='cuda',dtype=torch.float32)
    with torch.autocast('cuda',enabled=False):
        semantic=model._calculate_loss('cuda',fixed,batch['scores'],batch['costs'])
        correct=torch.nn.functional.cross_entropy(fixed,torch.tensor([0,1],device='cuda'))
        wrong=torch.nn.functional.cross_entropy(fixed,torch.tensor([1,0],device='cuda'))
    if not torch.allclose(semantic,correct,atol=1e-6,rtol=1e-6) or torch.allclose(semantic,wrong,atol=1e-6,rtol=1e-6):
        raise ValueError('Official Hybrid label-semantic check/negative control failed')
    model.train()
    before = model.head_layer[1].weight.detach().clone()
    from transformers import Adafactor
    optimizer = Adafactor(model.parameters(), lr=1e-5, scale_parameter=False,
        relative_step=False, warmup_init=False, weight_decay=.1)
    scaler = torch.amp.GradScaler('cuda')
    skipped = 0
    for attempt in range(8):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            output = model(**batch)
            # Preserve the official logits dtype under AMP. Explicitly casting
            # half logits to float changes CE rounding even when the returned
            # loss tensor is float32 (confirmed by the real dtype diagnostic).
            expected = torch.nn.functional.cross_entropy(output['preds'], torch.tensor([0, 1], device='cuda'))
            promoted_reference=torch.nn.functional.cross_entropy(output['preds'].float(),torch.tensor([0,1],device='cuda'))
        if not torch.allclose(output['loss'].float(), expected, atol=1e-5, rtol=1e-4):
            raise ValueError('Official deterministic Hybrid loss disagrees with independently constructed labels')
        scaler.scale(output['loss']).backward()
        previous_scale = scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        if scaler.get_scale() < previous_scale:
            skipped += 1
        else:
            break
    else:
        raise ValueError('Hybrid compatibility probe never obtained finite scaled gradients')
    if torch.equal(before, model.head_layer[1].weight):
        raise ValueError('Real Hybrid optimizer probe made no update')
    model.eval()
    with torch.inference_mode():
        with torch.autocast('cuda', dtype=torch.float16):
            official = model.predict(batch['source_ids'], batch['source_attention_mask'])['preds']
            hidden = model.pretrained_model(input_ids=batch['source_ids'], attention_mask=batch['source_attention_mask'], output_hidden_states=True).hidden_states[-1]
            indices = torch.where(batch['source_ids'] == tokenizer.source_prefix_id)
            manual = model.head_layer(hidden[indices[0], indices[1], :])
        difference = float((official.float() - manual.float()).abs().max())
    if difference > 1e-5 or attempts:
        raise ValueError('Manual source-token replay or network-denial probe failed')
    proof = dict(verified=True, real_backbone=True, official_source_token_head=True,
        deterministic_loss_verified=True, actual_backward_and_optimizer_update=True,
        flipped_label_negative_control_rejected=True,all_parameters_float32=True,
        amp_loss_dtype=str(output['loss'].dtype),amp_logits_dtype=str(output['preds'].dtype),
        amp_loss=float(output['loss'].detach()),promoted_logits_reference_loss=float(promoted_reference.detach()),
        gradscaler_overflow_skipped_steps=skipped, final_gradient_scale=scaler.get_scale(),
        source_token_manual_replay_max_difference=difference, microbatch=2, max_length=128,
        tokenizer_class=type(tokenizer).__name__,use_fast=True,
        fp32_parameters_fp16_autocast=True, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        preflight=preflight, source=pin(Path(__file__)), dependencies=sources(),
        backbone=pin(FOLLOWUP / 'deberta-asset-verification.json'), network_attempts=len(attempts), paid_api_calls=0)
    proof['environment'] = environment(torch)
    once(BASE / 'probe.json', proof)
    print(json.dumps(proof), flush=True)


def cohorts(kind):
    from scripts.research_pipeline import load_routing
    _, queries, splits, source_hashes = load_routing(ROOT / 'data')
    metadata = queries.set_index('query_id')
    groups = metadata.query_hash.to_dict()
    ids = {phase: list(splits.select(phase).query_ids) for phase in ('learning', 'tuning', 'deployment')}
    if kind == 'binary':
        partitions = partition_classifier_training(ids['learning'], ids['tuning'], ids['deployment'], groups)
        stream = None
    elif kind == 'drift':
        stream = make_prompt_disjoint_stream(ids['learning'], metadata.task.to_dict(), groups,
            warm_count=5000, before_count=5000, after_count=5000)
        candidate = FOLLOWUP / 'drift-prompt-disjoint-candidate.json'
        if candidate.exists() and read(candidate)['stream'] != stream:
            raise ValueError('Prospective drift cohort differs from the audited label-free candidate')
        validation = [q for q in ids['tuning'] if metadata.loc[q, 'task'].startswith('mmlu-')]
        partitions = partition_classifier_training(stream['warm'], validation, stream['before'] + stream['after'], groups)
        expected_tuning=read(ROOT/'outputs/openai-core-20261031/protocol.json')['primary_plan']['frozen_counts']['mmlu']['tuning']
        if len(validation) != expected_tuning:
            raise ValueError('Original MMLU tuning population changed')
    else:
        raise ValueError('Unknown Hybrid cohort')
    return partitions, stream, source_hashes


def audit(kind):
    from scripts.research_pipeline import load_routing
    from scripts.resources import require_resources
    preflight=require_resources(); torch,attempts=runtime()
    from transformers import AutoTokenizer
    partitions,stream,source_hashes=cohorts(kind)
    _,queries,_,_=load_routing(ROOT/'data'); texts=queries.set_index('query_id')['query'].to_dict()
    assets=read(FOLLOWUP/'deberta-asset-verification.json'); local=Path(assets['local_path'])
    for name in ('config.json','tokenizer_config.json','spm.model'):
        if sha(local/name)!=assets['files'][name]:
            raise ValueError('Hybrid tokenizer asset changed before label-free audit')
    tokenizer=AutoTokenizer.from_pretrained(local,local_files_only=True,use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError('Declared Hybrid fast tokenizer unavailable during input audit')
    official=sources(); namespace=dict(torch=torch,List=List,Dict=Dict)
    encode=reviewed_definitions(official['llm-blender-collator.py'],{'encode_texts'},namespace)['encode_texts']
    collator_cls=reviewed_definitions(official['hybrid_llm/pair_ranker/collator.py'],{'NClassCollator'},
        dict(namespace,PreTrainedTokenizer=object,encode_texts=encode,SOURCE_PREFIX='<|source|>'))['NClassCollator']
    collator=collator_cls(128,tokenizer,128); phases={}
    for phase,ids in partitions.items():
        lengths=[]; token_hash=hashlib.sha256(); mask_hash=hashlib.sha256(); unknown_tokens=0; unknown_rows=0
        for offset in range(0,len(ids),64):
            questions=[texts[q] for q in ids[offset:offset+64]]
            rendered=[tokenizer.source_prefix+t.replace(tokenizer.sep_token or tokenizer.eos_token,' ') for t in questions]
            lengths.extend(len(v) for v in tokenizer(rendered,truncation=False)['input_ids'])
            batch=collator([dict(source=t) for t in questions])
            if not torch.all((batch['source_ids']==tokenizer.source_prefix_id).sum(1)==1):
                raise ValueError('Actual historical question loses/duplicates the source token inside128 tokens')
            unknown=batch['source_ids']==tokenizer.unk_token_id
            unknown_tokens+=int(unknown.sum()); unknown_rows+=int(unknown.any(dim=1).sum())
            token_hash.update(batch['source_ids'].numpy().tobytes()); mask_hash.update(batch['source_attention_mask'].numpy().tobytes())
        phases[phase]=dict(rows=len(ids),clipped=sum(n>128 for n in lengths),max_full_tokens=max(lengths),
            unknown_tokens=unknown_tokens,rows_with_unknown_token=unknown_rows,
            full_token_sum=sum(lengths),token_ids_sha256=token_hash.hexdigest(),attention_mask_sha256=mask_hash.hexdigest(),
            query_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':'),ensure_ascii=False).encode()).hexdigest())
    if attempts:
        raise ValueError('Network attempted during label-free Hybrid input audit')
    proof=dict(verified=True,cohort=kind,partitions=partitions,stream=stream,source_hashes=source_hashes,
        token_audit=phases,source=pin(__file__),official_sources=official,preflight=preflight,
        network_attempts=len(attempts),selection_uses_outcomes=False,paid_api_calls=0)
    once(BASE/kind/'input-audit.json',proof)
    print(json.dumps(dict(cohort=kind,token_audit=phases,verified=True)),flush=True)


def declare(kind):
    from scripts.experiments import verify_pins
    from tools.openai_study import DEFAULT_BASE as CORE, sealed
    parent = sealed(CORE / 'protocol.json')
    verify_pins(parent, ROOT)
    proof = read(BASE / 'probe.json')
    if not proof['verified'] or proof['source']['sha256'] != sha(__file__):
        raise ValueError('Current real Hybrid compatibility probe required')
    partitions, stream, hashes = cohorts(kind)
    audit_proof=read(BASE/kind/'input-audit.json')
    if (not audit_proof['verified'] or audit_proof['source']['sha256']!=sha(__file__)
            or audit_proof['partitions']!=partitions or audit_proof['stream']!=stream or audit_proof['source_hashes']!=hashes):
        raise ValueError('Current complete label-free Hybrid input audit required')
    config = dict(seed=42, epochs=5, max_length=128, microbatch=2, accumulation=64,
        effective_batch=128, lr=1e-5, weight_decay=.1, optimizer='Adafactor',use_fast=True,
        warmup_ratio=.05, scheduler='linear', dropout=.05, fp32_parameters=True,
        autocast='float16', gradient_scaling=True, selection='minimum validation CE, earliest tie',
        deterministic_algorithms=True, workers=0, threads=1)
    protocol = dict(version=1, kind='local-source-token-HybridLLM-' + kind,
        arms=[CHEAP, STRONG], model='microsoft/deberta-v3-large', revision=MODEL_REVISION,
        official_commit=COMMIT, parent=pin(CORE / 'protocol.json'), source_hashes=hashes,
        environment=proof['environment'],
        partitions=partitions, stream=stream, config=config, threshold_grid=[i / 20 for i in range(21)],
        token_audit=audit_proof['token_audit'],
        drift_threshold=.5, source_pins=[pin(__file__), pin(ROOT / 'tools/openai_study_hybrid.py'),
            pin(ROOT / 'tools/openai_study_hybrid_verify.py'),
            pin(ROOT / 'configs/followup_assets.json'), pin(BASE / 'probe.json'),
            pin(ROOT / 'configs/followup-neural-requirements.txt'),
            pin(BASE / kind / 'input-audit.json'),
            pin(FOLLOWUP / 'deberta-asset-verification.json')], official_sources=sources(),
        label_rule='class0 iff historical cheap reward >= strong reward; no new LLM judgments',
        information='supervised both-arm training outcomes; PILOT selected-feedback with human prior',
        differences='verified Llama2 pair; local splits; micro2/accum64; local validation CE; native threshold front; not author-exact',
        paper_equivalent=False, synthetic=False, paid_api_calls=0)
    protocol['manifest_sha256']=digest(protocol)
    once(BASE / kind / 'protocol.json', protocol)
    print(json.dumps(dict(cohort=kind, counts={k:len(v) for k,v in partitions.items()}, frozen=True)), flush=True)


def training_data(protocol):
    import numpy as np
    import pandas as pd
    from scripts.research_pipeline import load_routing
    folder, queries, _, hashes = load_routing(ROOT / 'data')
    if hashes != protocol['source_hashes']:
        raise ValueError('Historical Hybrid input source changed')
    truth = pd.read_parquet(folder / 'outcomes.parquet', columns=['query_id', 'arm_id', 'reward', 'actual_cost'])
    lookup = {(r.query_id, r.arm_id):(float(r.reward), float(r.actual_cost)) for r in truth.itertuples(index=False)
              if r.arm_id in protocol['arms']}
    labels = {}
    # Deliberately do not request test labels for training/checkpoint selection.
    for phase in ('train', 'validation'):
        ids = protocol['partitions'][phase]
        labels[phase] = np.asarray(deterministic_labels([lookup[q, CHEAP][0] for q in ids], [lookup[q, STRONG][0] for q in ids]), dtype=np.int64)
    return queries.set_index('query_id')['query'].to_dict(), labels


def train(kind):
    from scripts.resources import require_resources
    preflight = require_resources()
    import numpy as np
    folder = BASE / kind
    protocol = sealed_protocol(folder / 'protocol.json'); initial_protocol_sha=sha(folder/'protocol.json')
    for record in protocol['source_pins'] + list(protocol['official_sources'].values()):
        if sha(ROOT / record['path']) != record['sha256']:
            raise ValueError('Frozen Hybrid training dependency changed')
    if (folder / 'selection.json').exists():
        raise ValueError('Completed Hybrid fit already exists; verify rather than retrain')
    if shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise ValueError('Hybrid fit requires20GiB free disk for recoverable local checkpoints')
    torch, attempts = runtime()
    if environment(torch)!=protocol['environment']:
        raise ValueError('Declared Hybrid numerical environment changed')
    from transformers import Adafactor, get_linear_schedule_with_warmup
    model, tokenizer, collator = build(torch, protocol['config']['seed'])
    texts, labels = training_data(protocol)
    encoded = {}
    for phase, ids in protocol['partitions'].items():
        chunks = []
        for offset in range(0, len(ids), 64):
            batch = collator([dict(source=texts[q]) for q in ids[offset:offset + 64]])
            if not torch.all((batch['source_ids'] == tokenizer.source_prefix_id).sum(1) == 1):
                raise ValueError('Embedded source-prefix token duplicated or lost in a historical prompt')
            chunks.append((batch['source_ids'], batch['source_attention_mask']))
        encoded[phase] = tuple(torch.cat([v[k] for v in chunks]) for k in (0, 1))
        for index,key in ((0,'token_ids_sha256'),(1,'attention_mask_sha256')):
            if hashlib.sha256(encoded[phase][index].numpy().tobytes()).hexdigest()!=protocol['token_audit'][phase][key]:
                raise ValueError('Hybrid actual training/evaluation token input differs from frozen audit')
    del texts
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if 'bias' in name.lower() or 'layernorm' in name.lower() else decay).append(parameter)
    optimizer = Adafactor([dict(params=decay, weight_decay=.1), dict(params=no_decay, weight_decay=0.)],
        lr=1e-5, scale_parameter=False, relative_step=False, warmup_init=False)
    steps_per_epoch = math.ceil(len(labels['train']) / 128)
    total_steps = steps_per_epoch * 5
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(total_steps * .05), total_steps)
    scaler = torch.amp.GradScaler('cuda')
    state_path = folder / 'training-state.pt'
    history, start_epoch, step_count, applied_steps, overflow_steps = [], 1, 0, 0, 0
    if state_path.exists():
        state = torch.load(state_path, map_location='cpu', weights_only=True)
        if state['protocol_sha256'] != sha(folder / 'protocol.json'):
            raise ValueError('Hybrid resume protocol changed')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
        history, start_epoch, step_count = state['history'], state['next_epoch'], state['steps']
        applied_steps, overflow_steps = state['applied_steps'], state['overflow_steps']
        torch.set_rng_state(state['cpu_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
        gc.collect()
    def atomic_checkpoint(destination, payload):
        temporary = destination.with_suffix('.partial.pt')
        with temporary.open('wb') as stream:
            torch.save(payload, stream)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, destination)
    def predict(phase):
        model.eval(); values = []
        with torch.inference_mode():
            for offset in range(0, len(protocol['partitions'][phase]), 2):
                source, mask = (v[offset:offset + 2].to('cuda') for v in encoded[phase])
                with torch.autocast('cuda', dtype=torch.float16):
                    values.append(model.predict(source, mask)['preds'].float().cpu())
        return torch.cat(values).numpy()
    for epoch in range(start_epoch, 6):
        model.train()
        generator = torch.Generator().manual_seed(protocol['config']['seed'] + epoch)
        permutation = torch.randperm(len(labels['train']), generator=generator)
        loss_sum = 0.
        for block in range(0, len(permutation), 128):
            indices = permutation[block:block + 128]; optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(indices), 2):
                take = indices[offset:offset + 2]
                source, mask = (v[take].to('cuda') for v in encoded['train'])
                selected = torch.tensor(labels['train'][take.numpy()], device='cuda')
                # The official forward consumes observed scores; these encode
                # exactly the same deterministic class without reading test rows.
                scores = torch.stack((1 - selected, selected), dim=1).float().unsqueeze(-1)
                with torch.autocast('cuda', dtype=torch.float16):
                    loss = model(source, mask, scores, None)['loss']
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite Hybrid training loss')
                loss_sum += float(loss.detach()) * len(take)
                scaler.scale(loss * (len(take) / len(indices))).backward()
            previous_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
            if scaler.get_scale() < previous_scale:
                # A skipped scaled-gradient overflow is normal GradScaler
                # behavior. Match HF's scheduler skip and disclose the count.
                overflow_steps += 1
            else:
                scheduler.step(); applied_steps += 1
            step_count += 1
            if (block // 128) % 20 == 0:
                print(f'Hybrid {kind} epoch{epoch}/5 optimizer{step_count}/{total_steps}', flush=True)
        logits = predict('validation')
        if sha(folder/'protocol.json')!=initial_protocol_sha:
            raise ValueError('Hybrid protocol changed during the active fit')
        validation_loss = float(torch.nn.functional.cross_entropy(torch.from_numpy(logits).double(), torch.from_numpy(labels['validation'])))
        history.append(dict(epoch=epoch, train_loss=loss_sum / len(labels['train']), validation_loss=validation_loss))
        candidate = folder / f'candidate-epoch-{epoch}.pt'
        if candidate.exists():
            raise ValueError('Uncommitted existing epoch candidate requires inspection')
        with candidate.open('xb') as stream:
            torch.save(model.state_dict(), stream)
        atomic_checkpoint(state_path, dict(protocol_sha256=sha(folder / 'protocol.json'), model=model.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
            history=history, next_epoch=epoch + 1, steps=step_count,
            applied_steps=applied_steps, overflow_steps=overflow_steps,
            cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))
        print(f'Hybrid {kind} epoch{epoch}: validationCE={validation_loss:.8f}', flush=True)
    selected_epoch = select_validation_checkpoint([r['validation_loss'] for r in history])
    selected_path = folder / f'candidate-epoch-{selected_epoch}.pt'
    model.load_state_dict(torch.load(selected_path, map_location='cpu', weights_only=True))
    predictions = folder / 'predictions.npz'
    if predictions.exists():
        raise ValueError('Uncommitted predictions already exist; inspect before resume')
    validation_logits = predict('validation'); test_logits = predict('test')
    with predictions.open('xb') as stream:
        np.savez(stream, validation_ids=np.asarray(protocol['partitions']['validation']),
            validation_logits=validation_logits, test_ids=np.asarray(protocol['partitions']['test']), test_logits=test_logits)
    selection = dict(verified_training_complete=True, selected_epoch=selected_epoch, history=history,
        protocol_sha256=sha(folder / 'protocol.json'), checkpoint=pin(selected_path), predictions=pin(predictions),
        supervised_label_sha256={phase:hashlib.sha256(values.tobytes()).hexdigest() for phase,values in labels.items()},
        epoch_checkpoints=[pin(folder / f'candidate-epoch-{epoch}.pt') for epoch in range(1, 6)],
        attempted_optimizer_steps=step_count, optimizer_steps=applied_steps, overflow_skipped_steps=overflow_steps,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        preflight=preflight, network_attempts=len(attempts), paid_api_calls=0)
    if attempts:
        raise ValueError('Unexpected network attempt during local Hybrid fit')
    if sha(folder/'protocol.json')!=initial_protocol_sha:
        raise ValueError('Hybrid protocol changed during final inference')
    once(folder / 'selection.json', selection)
    print(json.dumps(dict(cohort=kind, selected_epoch=selected_epoch, optimizer_steps=step_count, complete=True)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['probe', 'audit', 'declare', 'train'])
    parser.add_argument('--cohort', choices=['binary', 'drift'])
    args = parser.parse_args()
    if args.action == 'probe':
        probe()
    elif not args.cohort:
        parser.error('--cohort is required')
    else:
        globals()[args.action](args.cohort)


if __name__ == '__main__':
    main()

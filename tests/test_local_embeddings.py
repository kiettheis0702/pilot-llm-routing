import hashlib
import json

import numpy as np
import pytest


def rows():
    return [dict(query_id=f'q{i}',text=f'text {i}',prompt_hash=hashlib.sha256(f'text {i}'.encode()).hexdigest()) for i in range(5)]


def recipe():
    return dict(model='BAAI/bge-m3',revision='5617a9f61b028005a4858fdac845db406aefb181',
                dimension=1024,max_length=512,pooling='cls',normalization='l2',
                inference_dtype='float16',prompt_construction='verbatim-no-instruction')


class Encoder:
    """Only neural computation is replaced; real writer/resume/validation execute."""
    def __init__(self, fail_after=None):self.calls=0;self.fail_after=fail_after
    def __call__(self,texts):
        if self.fail_after is not None and self.calls>=self.fail_after:raise RuntimeError('encoder interrupted')
        self.calls+=1
        return np.full((len(texts),1024),1/32,dtype=np.float32),[600 if t.endswith('0') else 5 for t in texts]


def test_interrupted_resume_matches_complete_run_and_reports_truncation(tmp_path):
    from scripts.local_embeddings import encode_rows
    out=tmp_path/'interrupted'
    with pytest.raises(RuntimeError,match='interrupted'):
        encode_rows(rows(),out,recipe(),Encoder(fail_after=1),batch_size=2)
    assert not (out/'manifest.json').exists()
    cache=encode_rows(rows(),out,recipe(),Encoder(),batch_size=2,resume=True)
    other=encode_rows(rows(),tmp_path/'full',recipe(),Encoder(),batch_size=2)
    np.testing.assert_array_equal(cache.vectors,other.vectors)
    assert cache.metadata['truncated_queries']==1
    assert cache.metadata['max_observed_tokens']==600
    assert encode_rows(rows(),out,recipe(),Encoder(fail_after=0),resume=True).fingerprint()==cache.fingerprint()


def test_resume_rejects_changed_recipe_inputs_and_tampered_prefix(tmp_path):
    from scripts.local_embeddings import encode_rows
    out=tmp_path/'cache'
    with pytest.raises(RuntimeError):encode_rows(rows(),out,recipe(),Encoder(1),batch_size=2)
    changed=recipe();changed['max_length']=1024
    with pytest.raises(ValueError,match='resume'):encode_rows(rows(),out,changed,Encoder(),resume=True)
    modified=rows();modified.reverse()
    with pytest.raises(ValueError,match='resume'):encode_rows(modified,out,recipe(),Encoder(),resume=True)
    vectors=np.load(out/'vectors.npy',mmap_mode='r+');vectors[0,0]=99;vectors.flush();del vectors
    with pytest.raises(ValueError,match='checksum'):encode_rows(rows(),out,recipe(),Encoder(),resume=True)


def test_encoder_bad_output_and_wrong_prompt_hash_fail_closed(tmp_path):
    from scripts.local_embeddings import encode_rows
    bad=rows();bad[0]['text']='changed'
    with pytest.raises(ValueError,match='hash'):encode_rows(bad,tmp_path/'bad-input',recipe(),Encoder())
    with pytest.raises(ValueError):encode_rows(rows(),tmp_path/'bad-output',recipe(),lambda texts:(np.full((len(texts),1024),np.nan),[4]*len(texts)))
    assert not (tmp_path/'bad-output/manifest.json').exists()


def test_resource_gate_denies_low_ram_conflicts_and_invalid_override():
    from scripts.resources import require_resources
    with pytest.raises(RuntimeError,match='RAM'):
        require_resources(snapshot={'available_gib':7.9,'heavy_jobs':[]})
    with pytest.raises(RuntimeError,match='job'):
        require_resources(snapshot={'available_gib':12.,'heavy_jobs':[{'pid':99}]})
    assert require_resources(snapshot={'available_gib':8.,'heavy_jobs':[]})['available_gib']==8


def test_resume_rejects_symlink_before_modifying_external_vectors(tmp_path):
    from scripts.local_embeddings import encode_rows
    out=tmp_path/'cache'
    with pytest.raises(RuntimeError):encode_rows(rows(),out,recipe(),Encoder(1),batch_size=2)
    external=tmp_path/'external.npy'
    (out/'vectors.npy').rename(external)
    try:
        (out/'vectors.npy').symlink_to(external)
    except OSError as exc:
        pytest.skip(f'OS does not permit test symlinks: {exc}')
    before=external.read_bytes()
    with pytest.raises(ValueError,match='path'):
        encode_rows(rows(),out,recipe(),Encoder(),resume=True)
    assert external.read_bytes()==before


def test_interrupted_manifest_write_can_resume_without_reencoding(tmp_path,monkeypatch):
    from scripts.local_embeddings import encode_rows
    import scripts.embedding_store as store
    out=tmp_path/'cache'
    def interrupted_write(path,payload):
        path.write_text('{"partial":')
        raise OSError('disk full')
    with monkeypatch.context() as patch:
        patch.setattr(store,'_write_json',interrupted_write)
        with pytest.raises(OSError,match='disk full'):
            encode_rows(rows(),out,recipe(),Encoder())
    assert not (out/'manifest.json').exists()
    cache=encode_rows(rows(),out,recipe(),Encoder(fail_after=0),resume=True)
    assert len(cache.ids)==5


def test_resume_checks_vector_path_before_encoder_or_write(tmp_path,monkeypatch):
    from pathlib import Path
    from scripts.local_embeddings import encode_rows
    out=tmp_path/'cache'
    with pytest.raises(RuntimeError):encode_rows(rows(),out,recipe(),Encoder(1),batch_size=2)
    before=(out/'vectors.npy').read_bytes()
    original=Path.is_symlink
    monkeypatch.setattr(Path,'is_symlink',lambda path: path==out/'vectors.npy' or original(path))
    encoder=Encoder()
    with pytest.raises(ValueError,match='path'):
        encode_rows(rows(),out,recipe(),encoder,resume=True)
    assert encoder.calls==0
    assert (out/'vectors.npy').read_bytes()==before


def test_windows_preflight_does_not_count_own_venv_launcher():
    import os
    import subprocess
    import sys
    if os.name!='nt':pytest.skip('Windows venv launcher regression')
    code=('import os,json; from scripts.resources import resource_snapshot; '
          'print(json.dumps(dict(parent=os.getppid(),snapshot=resource_snapshot())))')
    result=subprocess.run([sys.executable,'-B','-c',code,'--local_embeddings-preflight-probe'],
                          check=True,capture_output=True,text=True)
    report=json.loads(result.stdout)
    assert report['parent'] not in {job['pid'] for job in report['snapshot']['heavy_jobs']}


def test_grouped_checkpoints_resume_from_last_published_prefix(tmp_path):
    from scripts.local_embeddings import encode_rows
    inputs = [dict(query_id=f'q{i}', text=f'text {i}',
                   prompt_hash=hashlib.sha256(f'text {i}'.encode()).hexdigest()) for i in range(9)]
    out = tmp_path/'cache'
    with pytest.raises(RuntimeError, match='interrupted'):
        encode_rows(inputs, out, recipe(), Encoder(fail_after=3), batch_size=2, checkpoint_batches=2)
    state = json.loads((out/'progress.json').read_text())
    assert state['completed'] == 4
    assert [(c['start'], c['end']) for c in state['chunks']] == [(0,4)]
    # Bytes written after the last checkpoint are deliberately not trusted.
    vectors = np.load(out/'vectors.npy', mmap_mode='r+')
    vectors[4:6] = 99; vectors.flush(); del vectors
    resumed_encoder = Encoder()
    actual = encode_rows(inputs, out, recipe(), resumed_encoder, batch_size=2,
                         checkpoint_batches=2, resume=True)
    expected = encode_rows(inputs, tmp_path/'reference', recipe(), Encoder(), batch_size=2)
    assert resumed_encoder.calls == 3
    np.testing.assert_array_equal(actual.vectors, expected.vectors)
    assert actual.metadata['total_original_tokens'] == expected.metadata['total_original_tokens'] == 640
    assert actual.metadata['truncated_queries'] == expected.metadata['truncated_queries'] == 1
    state = json.loads((out/'progress.json').read_text())
    assert state['completed'] == 9
    assert [(c['start'], c['end']) for c in state['chunks']] == [(0,4), (4,8), (8,9)]


def test_grouped_checkpoint_limit_is_bounded_before_writing(tmp_path):
    from scripts.local_embeddings import encode_rows
    for value in (0, 65, True, 1.5):
        with pytest.raises(ValueError, match='Checkpoint'):
            encode_rows(rows(), tmp_path/'bad', recipe(), Encoder(), checkpoint_batches=value)
        assert not (tmp_path/'bad').exists()

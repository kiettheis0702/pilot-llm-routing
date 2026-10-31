import hashlib
import importlib
import numpy as np
import pytest
from scripts.pretraining import _digest
from tools.openai_study_instructor_journal import EmbeddingJournal


def module():return importlib.import_module('tools.openai_study_instructor_tail')


def test_tail_overlay_preserves_representation_and_declares_batch_padding_change():
    protocol=dict(unique_texts=3,dimension=2,revision='pinned',instruction='instruction',max_length=512,dtype='float32')
    proof=dict(committed_rows=1,remaining_rows=2,dimension=2,protocol_sha256='a'*64,prefix_vectors_sha256='b'*64)
    overlay=module().make_overlay(protocol,proof,'c'*64)
    assert overlay['start_row']==1 and overlay['end_row']==3
    assert overlay['batch_size']==1 and overlay['padding']=='max_length'
    assert overlay['dtype']=='float32' and overlay['max_length']==512
    assert overlay['original_protocol_sha256']=='a'*64
    assert overlay['source_prefix_sha256']=='b'*64 and overlay['producer_sha256']=='c'*64
    assert overlay['inference_api_calls']==0
    proof['remaining_rows']=9
    with pytest.raises(ValueError):module().make_overlay(protocol,proof,'c'*64)


def test_tail_readback_detects_any_change_in_approved_committed_prefix(tmp_path):
    ids=['q0','q1'];hashes=[hashlib.sha256(q.encode()).hexdigest() for q in ids]
    recipe=dict(model='synthetic',version='test',source='unit',normalization='L2',synthetic=True)
    values=np.array([[1.,0.]],dtype=np.float32)
    with EmbeddingJournal(tmp_path/'cache',ids=ids,prompt_hashes=hashes,provenance=recipe,dimension=2,metadata={}) as journal:
        journal.commit(0,values)
        proof=dict(committed_rows=1,dimension=2,prefix_vectors_sha256=hashlib.sha256(values.tobytes()).hexdigest())
        assert module().verify_original_prefix(tmp_path/'cache',proof)==proof['prefix_vectors_sha256']
    matrix=np.load(tmp_path/'cache/vectors.npy',mmap_mode='r+');matrix[0]=[0.,1.];matrix.flush();del matrix
    with pytest.raises(ValueError,match='prefix'):module().verify_original_prefix(tmp_path/'cache',proof)

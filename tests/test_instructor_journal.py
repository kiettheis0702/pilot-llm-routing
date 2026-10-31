"""A partial embedding cache must never masquerade as completed work."""
import hashlib
import importlib
import numpy as np
import pytest


def journal(path,**overrides):
    values=dict(ids=['q0','q1','q2'],prompt_hashes=[hashlib.sha256(q.encode()).hexdigest() for q in ['q0','q1','q2']],
        provenance=dict(model='synthetic-instructor',version='test',source='unit test',normalization='L2',synthetic=True),
        dimension=2,metadata={'recipe_sha':'one'})
    values.update(overrides)
    return importlib.import_module('tools.openai_study_instructor_journal').EmbeddingJournal(path,**values)


def test_resume_keeps_committed_prefix_and_overwrites_uncommitted_tail(tmp_path):
    path=tmp_path/'cache'
    with journal(path) as writer:
        writer.commit(0,np.array([[1.,0.]],dtype=np.float32))
        with pytest.raises(ValueError,match='incomplete'):writer.finish()
    matrix=np.load(path/'vectors.npy',mmap_mode='r+')
    matrix[1:]=99.;matrix.flush();del matrix
    with journal(path) as writer:
        assert writer.next_row==1
        writer.commit(1,np.array([[0.,1.],[.6,.8]],dtype=np.float32))
        cache=writer.finish()
        np.testing.assert_array_equal(cache.get('q0'),[1.,0.])
        np.testing.assert_allclose(cache.get('q2'),[.6,.8])
    with journal(path) as writer:
        assert writer.next_row==3
        assert writer.finish().fingerprint()==cache.fingerprint()
        with pytest.raises(FileExistsError):writer.commit(3,np.array([[1.,0.]],dtype=np.float32))


def test_resume_rejects_changed_committed_vector(tmp_path):
    path=tmp_path/'cache'
    with journal(path) as writer:writer.commit(0,np.array([[1.,0.]],dtype=np.float32))
    matrix=np.load(path/'vectors.npy',mmap_mode='r+');matrix[0]=[0.,1.];matrix.flush();del matrix
    with pytest.raises(ValueError,match='checksum'):journal(path)
    assert not (path/'manifest.json').exists()


def test_recipe_change_refuses_resume_instead_of_mixing_encoders(tmp_path):
    path=tmp_path/'cache'
    with journal(path):pass
    with pytest.raises(ValueError,match='identity'):journal(path,metadata={'recipe_sha':'different'})
    # A failed open must release the actual OS lock.
    with journal(path) as writer:assert writer.next_row==0


@pytest.mark.parametrize('vectors',[np.array([[np.nan,0.]]),np.array([[0.,0.]]),np.array([[2.,0.]]),np.ones((1,3))])
def test_invalid_vector_cannot_be_committed_or_published(tmp_path,vectors):
    path=tmp_path/'cache'
    with journal(path) as writer:
        with pytest.raises(ValueError):writer.commit(0,vectors)
        assert writer.next_row==0
    assert not (path/'manifest.json').exists()


def test_noncontiguous_commit_rejected_before_writing(tmp_path):
    with journal(tmp_path/'cache') as writer:
        with pytest.raises(ValueError,match='contiguous'):writer.commit(1,np.array([[1.,0.]],dtype=np.float32))
        assert writer.next_row==0


def test_second_writer_cannot_modify_same_cache_and_lock_is_reusable(tmp_path):
    path=tmp_path/'cache'
    with journal(path):
        with pytest.raises(RuntimeError,match='writer'):journal(path)
    with journal(path) as writer:assert writer.next_row==0

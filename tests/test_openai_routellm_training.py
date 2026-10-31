"""Guard source isolation and calibration-only checkpoint selection."""
import hashlib
import importlib

import pytest


def module():
    return importlib.import_module('tools.openai_study_routellm_train')


def test_source_loader_never_executes_module_level_network_or_other_side_effects(tmp_path):
    marker=tmp_path/'must-not-exist'
    source=tmp_path/'train.py'
    source.write_text(f"raise RuntimeError('API client import executed')\nopen({str(marker)!r}, 'w').write('bad')\nclass MFModel_Train:\n    def score(self):\n        return 17\n",encoding='utf-8')
    cls=module().load_training_class(source,hashlib.sha256(source.read_bytes()).hexdigest(),{})
    assert cls().score()==17
    assert not marker.exists()


def test_source_loader_rejects_modified_or_decorated_classes(tmp_path):
    source=tmp_path/'train.py'
    source.write_text('class MFModel_Train:\n    pass\n',encoding='utf-8')
    with pytest.raises(ValueError,match='checksum'):
        module().load_training_class(source,'0'*64,{})
    source.write_text('@dangerous()\nclass MFModel_Train:\n    pass\n',encoding='utf-8')
    with pytest.raises(ValueError,match='decorator'):
        module().load_training_class(source,hashlib.sha256(source.read_bytes()).hexdigest(),{})


def test_model_id_loader_parses_literals_without_importing_paid_client(tmp_path):
    source=tmp_path/'model.py'
    source.write_text("raise RuntimeError('paid client')\nMODEL_IDS={'small':0,'large':1}\n",encoding='utf-8')
    assert module().load_model_ids(source,hashlib.sha256(source.read_bytes()).hexdigest())=={'small':0,'large':1}


def test_checkpoint_selection_ignores_any_test_scores_and_breaks_loss_tie_by_earlier_epoch():
    history=[dict(epoch=1,calibration_loss=.4,test_accuracy=1.),
             dict(epoch=2,calibration_loss=.3,test_accuracy=0.),
             dict(epoch=3,calibration_loss=.3,test_accuracy=.9)]
    assert module().best_epoch(history)==2
    for row in history:row['test_accuracy']=1-row['test_accuracy']
    assert module().best_epoch(history)==2
    with pytest.raises(ValueError):module().best_epoch([dict(epoch=1,calibration_loss=float('nan'))])


def test_partition_refuses_shared_question_across_train_and_test():
    rows=[dict(query_id='q',winner='a',loser='b',split='train'),
          dict(query_id='q2',winner='b',loser='a',split='test')]
    with pytest.raises(ValueError,match='overlap'):
        module().partition_pairs(rows,{'q':'same-hash','q2':'same-hash'},{'a':0,'b':1})


def test_partition_preserves_split_and_model_identity_and_rejects_unknown_model():
    rows=[dict(query_id='q',winner='a',loser='b',split='train'),
          dict(query_id='q2',winner='b',loser='a',split='calibration'),
          dict(query_id='q3',winner='a',loser='b',split='test')]
    groups=module().partition_pairs(rows,{'q':'h1','q2':'h2','q3':'h3'},{'a':8,'b':24})
    assert groups['train']==[('q',8,24)]
    assert groups['calibration']==[('q2',24,8)]
    assert groups['test']==[('q3',8,24)]
    rows[0]['winner']='unverified'
    with pytest.raises(ValueError):module().partition_pairs(rows,{'q':'h1','q2':'h2','q3':'h3'},{'a':8,'b':24})

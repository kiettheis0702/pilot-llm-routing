import ast
from pathlib import Path

import pytest

from tools.openai_study_hybrid_default_batch import adapt_trainer_source


def test_batch_adapter_preserves_model_data_split_label_and_selection_functions():
    path=Path(__file__).resolve().parents[1]/'tools/openai_study_hybrid_train.py'
    original=path.read_bytes();before=ast.parse(original)
    after,audit=adapt_trainer_source(original)
    for name in ('build','runtime','environment','cohorts','training_data','sealed_protocol'):
        first=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name==name)
        second=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(first)==ast.dump(second)
    assert audit['train_microbatch']==8 and audit['effective_batch']==128
    assert path.read_bytes()==original


def test_batch_adapter_rejects_a_changed_parent_before_running_any_code():
    path=Path(__file__).resolve().parents[1]/'tools/openai_study_hybrid_train.py'
    with pytest.raises(ValueError,match='source'):
        adapt_trainer_source(path.read_bytes()+b'\n# altered parent\n')

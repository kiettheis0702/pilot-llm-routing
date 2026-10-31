import pytest

from tools.openai_followup_tracking import make_record, validate_record


def test_same_condition_is_namespaced_by_embedding_recipe_before_upload():
    config=dict(domain='routerbench',seed=42,method='pilot',variant='main')
    metrics=dict(requested=10,served=10,quality_fraction_gpt4=.9,cost_fraction_gpt4=.25,coverage=1.)
    first=make_record('openai','core','same-condition',config,metrics,'0'*64)
    second=make_record('instructor','core','same-condition',config,metrics,'0'*64)
    assert first['id']!=second['id']
    assert make_record('openai','core','same-condition',config,metrics,'0'*64)==first
    assert validate_record(first)==first


def test_public_tracking_rejects_private_fields_nonfinite_values_and_stale_digest():
    with pytest.raises(ValueError):
        make_record('openai','core','id',dict(prompt_text='private'),dict(coverage=1.),'0'*64)
    with pytest.raises(ValueError):
        make_record('openai','core','id',{},dict(coverage=float('nan')),'0'*64)
    for private_path in ('C:/Users/ADMIN/secret',r'C:\Users\ADMIN\secret'):
        with pytest.raises(ValueError):
            make_record('openai','core','id',dict(description=private_path),dict(coverage=1.),'0'*64)
    good=make_record('openai','core','id',{},dict(coverage=1.),'0'*64)
    good['metrics']['coverage']=.5
    with pytest.raises(ValueError):
        validate_record(good)

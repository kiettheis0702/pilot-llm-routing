import csv
import json

import pytest

from tools.course import prepare_presentation


def links_file(path, ids, **changes):
    rows=[dict(id=i,url=f'https://wandb.ai/team/pilot/runs/{i}',public_verified=True,
               verified_at='2026-10-31T01:00:00+00:00',verification_method='signed-out-browser') for i in ids]
    if changes:rows[0].update(changes)
    path.write_text(json.dumps(dict(version=1,runs=rows)))
    return path


def test_verified_publication_is_preserved_on_repeated_rebuild(tmp_path):
    from tools.course.publication import publication_state
    records=[{'id':'pilot-a'},{'id':'pilot-b'}]
    file=links_file(tmp_path/'public.json',['pilot-a','pilot-b'])
    state=publication_state(records,file)
    assert state['complete'] and state['verified_count']==2
    assert state['project_url']=='https://wandb.ai/team/pilot'
    assert publication_state(records,file)==state
    assert not publication_state(records,None)['complete']


@pytest.mark.parametrize('change',[{'public_verified':False},{'id':'unknown'},
    {'url':'https://example.com/team/pilot/runs/pilot-a'},
    {'verified_at':'2026-10-31T01:00:00'}])
def test_unverified_unknown_or_invalid_public_links_fail(tmp_path,change):
    from tools.course.publication import publication_state
    file=links_file(tmp_path/'links.json',['pilot-a'],**change)
    with pytest.raises(ValueError):publication_state([{'id':'pilot-a'}],file)


def test_real_mapping_rebuild_keeps_verified_link(tmp_path):
    from tools.course.tracking import build_payloads
    records=build_payloads(prepare_presentation.PACKAGE,prepare_presentation.REPORT/'analysis')
    file=links_file(tmp_path/'public.json',[records[0]['id']])
    out=tmp_path/'content'
    args=['--output-dir',str(out),'--public-links',str(file)]
    prepare_presentation.main(args)
    before=(out/'experiment-map.csv').read_bytes()
    prepare_presentation.main(args)
    assert (out/'experiment-map.csv').read_bytes()==before
    with (out/'experiment-map.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
    assert rows[0]['wandb_url']==f"https://wandb.ai/team/pilot/runs/{records[0]['id']}"
    assert rows[0]['status']=='verified public'
    assert rows[1]['wandb_url']==''

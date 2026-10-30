"""Independent OOXML and numerical readback of the actual course deliverables."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from tools.course.analyze import ROOT, sha256, verify_package, write_json

NS={'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
    'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
    'c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}


def need(condition, message):
    """Use explicit failures; verification must also work under Python optimization."""
    if not condition:
        raise ValueError(message)


def numbers(node, key):
    """Read a native chart's cached numeric sequence in document order."""
    return [float(n.text) for n in node.findall(f'.//c:{key}//c:pt/c:v', NS)]


def inspect_pptx(file, count, size, chart_keys, content):
    """Verify actual ZIP/XML text, notes, dimensions, charts and package sizes."""
    with ZipFile(file) as archive:
        names=archive.namelist()
        slides=sorted(n for n in names if re.fullmatch(r'ppt/slides/slide\d+\.xml', n))
        notes=sorted(n for n in names if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml',n))
        need(len(slides)==count==len(notes),'Slide/notes count mismatch')
        dimensions=ET.fromstring(archive.read('ppt/presentation.xml')).find('p:sldSz',NS)
        need(tuple(int(dimensions.get(k)) for k in ('cx','cy'))==size,'Wrong canvas size')
        word_counts=[]
        for member in notes:
            note=ET.fromstring(archive.read(member))
            text=' '.join(n.text or '' for n in note.findall('.//a:t',NS))
            wc=len(text.split());need(20<=wc<=500,'Missing/oversized speaker notes: '+member)
            word_counts.append(wc)
        for member in slides:
            slide=ET.fromstring(archive.read(member))
            text=' '.join(n.text or '' for n in slide.findall('.//a:t',NS))
            need(bool(re.search(r'\[[1-5]\]',text)),'Missing numbered citation: '+member)
        charts=sorted((n for n in names if re.search(r'/charts/chart\d+\.xml$',n)),
                      key=lambda n:int(re.search(r'chart(\d+)\.xml',n).group(1)))
        need(len(charts)==len(chart_keys),'Chart inventory mismatch')
        for member,key in zip(charts,chart_keys):
            root=ET.fromstring(archive.read(member));expected=content['charts'][key]
            series=root.findall('.//c:ser',NS)
            need(len(series)==len(expected['series']),'Series count mismatch')
            for actual, exp in zip(series,expected['series']):
                xs,ys=numbers(actual,'xVal'),numbers(actual,'yVal')
                need(xs==expected['x'],'Chart x-values differ from source')
                need(len(ys)==len(exp['points']),'Chart point count mismatch')
                need(all(abs(a-b['mean'])<=5.1e-9 for a,b in zip(ys,exp['points'])),
                     'Chart y-values differ from source: '+key)
        if count==33:
            slide=ET.fromstring(archive.read('ppt/slides/slide27.xml'))
            cells=[''.join(t.text or '' for t in n.findall('.//a:t',NS)) for n in slide.findall('.//a:tc',NS)]
            need(cells==['Variant','RouterBench','MMLU']+[v for row in content['ablations'] for v in row],
                 'Ablation table differs from source')
    need(file.stat().st_size<20_000_000,'File exceeds course size limit')
    return dict(file=file.relative_to(ROOT).as_posix(),sha256=sha256(file),bytes=file.stat().st_size,
                slides=count,notes_min_words=min(word_counts),notes_max_words=max(word_counts),
                native_charts=len(charts),size_emu=size,native_powerpoint_opened=False)


def verify_all(folder):
    """Recompute chart aggregates and supplement accounting independently of builders."""
    report=ROOT/'reports/course-submission-2026-10-30'
    frozen=ROOT/'reports/full-experiments-2026-10-29'
    integrity=verify_package(frozen);verify_package(report/'analysis')
    content=json.loads((report/'slide-content.json').read_text(encoding='utf-8'))
    need(content['source_csv_sha256']==sha256(frozen/'aggregate/per-seed.csv'),'Wrong source CSV hash')
    with (frozen/'aggregate/per-seed.csv').open(encoding='utf-8',newline='') as stream:
        rows=list(csv.DictReader(stream))
    byid={r['id']:r for r in rows}
    checked=0
    for key,chart in content['charts'].items():
        if key in ('cost','magnitude'):continue
        metric='learning_empirical_oracle_regret' if key.endswith('regret') else 'service_utility'
        for series in chart['series']:
            for point in series['points']:
                source=[byid[i] for i in point['source_ids']]
                need(len(source)==3 and {int(r['seed']) for r in source}=={7,42,2026},'Missing seed cell')
                values=[float(r[metric]) for r in source]
                need(abs(math.fsum(values)/3-point['mean'])<1e-10,'Mean differs from raw CSV')
                need(min(values)==point['minimum'] and max(values)==point['maximum'],'Range differs')
                checked+=1
    extra=json.loads((report/'analysis/analysis.json').read_text(encoding='utf-8'))
    for case in extra['stress']:
        for policy in case['policies']:
            trace=policy['trace'];rewards=[.2 if r['arm']=='low' else .8 for r in trace]
            need(len(trace)==case['rounds'],'Wrong stress horizon')
            need(all(r['reward']==v for r,v in zip(trace,rewards)),'Stress reward differs from selected arm')
            need(abs(math.fsum(.8-v for v in rewards)-policy['regret'])<1e-8,'Stress regret mismatch')
    for label,cases in extra['cost_sensitivity'].items():
        group=rows if label=='all conditions' else [r for r in rows if r['variant']=='main' and r['method']=='pilot' and float(r['learning_fraction'])==1]
        for c in cases:
            excess=[max(0,c['factor']*float(r['actual_recorded_cost'])-float(r['budget'])) for r in group]
            need(sum(v>1e-9 for v in excess)==c['over_budget_count'],'Cost excess count mismatch')
            need(abs(math.fsum(excess)-c['total_excess'])<1e-8,'Cost total mismatch')
    folder=Path(folder).resolve()
    artifacts=[inspect_pptx(folder/'midterm/8.pptx',33,(12192000,6858000),
        ['routerbench-budget','mmlu-budget','routerbench-learning','routerbench-regret','magnitude','cost'],content),
        inspect_pptx(folder/'poster/8.pptx',1,(15120000,21384000),['magnitude','cost'],content)]
    return dict(passed=True,original_package=integrity,checked_aggregate_points=checked,
        artifacts=artifacts,source_csv_sha256=content['source_csv_sha256'],
        paid_api_calls=0,public_wandb_urls_verified=False)


def main():
    """Check a pair of PPTX files and write a new verification receipt."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,default=ROOT/'deliverables')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=verify_all(args.folder);write_json(args.output,result)
    print(json.dumps(result))


if __name__=='__main__':main()

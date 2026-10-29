"""Independent CSV-to-figure and PDF readback; needs pdfplumber, not the plotter."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import pdfplumber

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/research-experiments-20261029'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(name):
    with (BASE / 'aggregate' / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10), (a, b)


def select(rows, **criteria):
    return [r for r in rows if all(str(r[k]) == str(v) for k, v in criteria.items())]


individual = read_csv('per-seed.csv')
learning = read_csv('learning-prefixes.csv')
assert len(individual) == 1098 and len(learning) == 336
figures = json.loads((BASE / 'figures/figure-data.json').read_text())
for name, expected in figures['files'].items():
    assert sha(BASE / 'figures' / name) == expected
for name, expected in figures['source_files'].items():
    assert sha(BASE / 'aggregate' / name) == expected
assert len(figures['series']) == 6
checked = 0
for name, series in figures['series'].items():
    assert len(series) == (10 if name == 'ablations' else 26)
    for line in series:
        for i, x in enumerate(line['x']):
            if name == 'ablations':
                values = []
                for seed in ('7', '42', '2026'):
                    rows = select(individual, domain=line['domain'], seed=seed, method='pilot')
                    rows = [r for r in rows if float(r['learning_fraction']) == 1 and float(r['budget_fraction']) == x]
                    variant = select(rows, variant=line['variant'])
                    control = select(rows, variant='main')
                    assert len(variant) == len(control) == 1
                    values.append(float(variant[0]['service_utility']) - float(control[0]['service_utility']))
                close(line['low'][i], min(values))
                close(line['high'][i], max(values))
            else:
                source = learning if name == 'learning-regret' else individual
                rows = select(source, domain=line['domain'], method=line['method'], variant='main')
                if name in ('learning-regret', 'learning-size'):
                    rows = [r for r in rows if float(r['learning_fraction']) == x]
                    if name == 'learning-size':
                        rows = [r for r in rows if float(r['budget_fraction']) == .5]
                else:
                    rows = [r for r in rows if float(r['learning_fraction']) == 1 and float(r['budget_fraction']) == x]
                assert len(rows) == 3 and {r['seed'] for r in rows} == {'7', '42', '2026'}
                metric = line['source_metric'].removesuffix('_mean')
                values = [float(r[metric]) for r in rows]
            close(line['y'][i], mean(values))
            checked += 1

documents = json.loads((BASE / 'deliverables/manifest.json').read_text())
assert documents['aggregate_manifest_sha256'] == sha(BASE / 'aggregate/manifest.json')
assert documents['figure_manifest_sha256'] == sha(BASE / 'figures/figure-data.json')
pdfs = {}
for name, expected in documents['files'].items():
    path = BASE / 'deliverables' / name
    assert sha(path) == expected
    with pdfplumber.open(path) as pdf:
        count = 9 if 'report' in name else 1
        assert len(pdf.pages) == count
        content = '\n'.join(page.extract_text() or '' for page in pdf.pages)
        assert all(word in content for word in ('PILOT', '1,098', 'LinUCB', 'ExploreOnly'))
        assert 'limits' in content.lower() or 'limitations' in content.lower()
        outside = []
        for page_index, page in enumerate(pdf.pages):
            for word in page.extract_words():
                if word['x0'] < 0 or word['x1'] > page.width or word['top'] < 0 or word['bottom'] > page.height:
                    outside.append((page_index + 1, word['text']))
        assert not outside, outside
        pdfs[name] = dict(pages=count, text_characters=len(content), words_outside_page=outside, sha256=expected)

# The packaged verifier must reproduce the original numerical report exactly.
old = json.loads((BASE / 'verification/routerbench-seed7-pilot-main.json').read_text())
new = json.loads((BASE / 'verification-reproduction/routerbench-seed7-pilot-main.json').read_text())
for report in (old, new):
    report.pop('script_sha256')
    report.pop('preflight')
assert old == new

result = dict(verified=True, figures=6, plotted_points=checked, source_rows=len(individual),
              pdfs=pdfs, independent_packaged_replay_matches=True,
              synthetic=False, paper_equivalent=False, script_sha256=sha(Path(__file__)))
with (BASE / 'deliverable-checks.json').open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result))

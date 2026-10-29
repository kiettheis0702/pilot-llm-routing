import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from scripts.resources import require_resources

BASE = ROOT / 'outputs/final-reproduction-20261029'
BASE.mkdir()
SOURCE = BASE / 'source'
SOURCE.mkdir()
files = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT).decode('utf-8').split('\0')
hashes = {}
for name in sorted(set(files) - {''}):
    src = ROOT / name
    if not src.is_file():
        continue
    target = SOURCE / name
    assert target.resolve().is_relative_to(SOURCE.resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)
    hashes[name] = hashlib.sha256(src.read_bytes()).hexdigest()
    assert hashlib.sha256(target.read_bytes()).hexdigest() == hashes[name]
(BASE / 'source-files.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
BASELINE = BASE / 'baseline'
BASELINE.mkdir()
archive = subprocess.check_output(['git', 'archive', '--format=zip', 'HEAD'], cwd=ROOT)
with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
    for name in zipped.namelist():
        assert (BASELINE/name).resolve().is_relative_to(BASELINE.resolve())
    zipped.extractall(BASELINE)

env = dict(os.environ)
env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           UV_CONCURRENT_INSTALLS='1', UV_CONCURRENT_BUILDS='1', UV_CONCURRENT_DOWNLOADS='1')
UV = r'C:\Users\ADMIN\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe'
PYTHON = r'C:\Users\ADMIN\AppData\Roaming\uv\python\cpython-3.11.18-windows-x86_64-none\python.exe'
report = dict(kind='clean-source-snapshot-reproduction', scope='uncommitted source snapshot; not a Git clone or a full re-embedding',
              source_files=len(hashes), source_manifest_sha256=hashlib.sha256((BASE/'source-files.json').read_bytes()).hexdigest(),
              base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), steps=[])

def run(name, cmd, cwd):
    preflight = require_resources()
    start = time.perf_counter()
    with (BASE/(name+'.log')).open('w',encoding='utf-8') as log:
        result = subprocess.run(cmd,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
    item = dict(name=name, command=cmd, cwd=str(cwd), exit_code=result.returncode,
                seconds=time.perf_counter()-start, preflight=preflight, log=name+'.log')
    report['steps'].append(item)
    (BASE/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(item),flush=True)
    print((BASE/(name+'.log')).read_text(encoding='utf-8')[-1800:],flush=True)
    if result.returncode:
        raise RuntimeError(name)

run('uv-sync', [UV,'sync','--locked','--extra','cost','--offline','--python',PYTHON,'--project',str(SOURCE)], SOURCE)
fresh = str(SOURCE/'.venv/Scripts/python.exe')
run('baseline-pytest',[fresh,'-m','pytest','-q','--basetemp',str(BASE/'baseline-tmp'),'--junitxml',str(BASE/'baseline.xml')],BASELINE)
run('snapshot-pytest',[fresh,'-m','pytest','-q','--basetemp',str(BASE/'snapshot-tmp'),'--junitxml',str(BASE/'snapshot.xml')],SOURCE)
run('toy-smoke',[fresh,'-m','scripts.run_toy_pipeline','--output-dir',str(BASE/'toy-smoke')],SOURCE)
run('baseline-smoke',[fresh,'-m','scripts.run_baselines','--source','toy','--rows','100','--output-dir',str(BASE/'baseline-smoke')],SOURCE)
report['verified']=True
report['synthetic_smokes']=True
report['full_real_experiment_rerun']=False
(BASE/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('REPRODUCTION COMPLETE',flush=True)

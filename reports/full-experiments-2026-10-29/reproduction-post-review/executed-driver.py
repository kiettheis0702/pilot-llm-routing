"""Fresh source and environment after the final verifier guard fix."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from scripts.resources import require_resources
BASE=ROOT/'outputs/final-reproduction-post-review-20261029';BASE.mkdir()
SOURCE=BASE/'source';SOURCE.mkdir()
hashes={}
files=subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=ROOT).decode().split('\0')
for name in sorted(set(files)-{''}):
    src=ROOT/name
    if not src.is_file():continue
    dest=SOURCE/name;assert dest.resolve().is_relative_to(SOURCE.resolve())
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    hashes[name]=hashlib.sha256(src.read_bytes()).hexdigest()
    assert hashlib.sha256(dest.read_bytes()).hexdigest()==hashes[name]
(BASE/'source-files.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',UV_CONCURRENT_INSTALLS='1',UV_CONCURRENT_BUILDS='1',UV_CONCURRENT_DOWNLOADS='1')
UV=r'C:\Users\ADMIN\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe'
PYTHON=r'C:\Users\ADMIN\AppData\Roaming\uv\python\cpython-3.11.18-windows-x86_64-none\python.exe'
fresh=str(SOURCE/'.venv/Scripts/python.exe')
report={'kind':'post-review-clean-source-reproduction','base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_files':len(hashes),'steps':[],'assertions_enabled':__debug__}
def run(name,command,cwd):
    preflight=require_resources();start=time.perf_counter()
    with (BASE/(name+'.log')).open('w',encoding='utf-8') as f:result=subprocess.run(command,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
    report['steps'].append(dict(name=name,command=command,cwd=str(cwd),exit_code=result.returncode,seconds=time.perf_counter()-start,preflight=preflight))
    (BASE/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(name,result.returncode,flush=True);print((BASE/(name+'.log')).read_text()[-1400:],flush=True)
    if result.returncode:raise RuntimeError(name)
run('uv-sync',[UV,'sync','--locked','--extra','cost','--offline','--python',PYTHON,'--project',str(SOURCE)],SOURCE)
run('pytest',[fresh,'-m','pytest','-q','--basetemp',str(BASE/'pytest-tmp'),'--junitxml',str(BASE/'pytest.xml')],SOURCE)
run('toy-smoke',[fresh,'-m','scripts.run_toy_pipeline','--output-dir',str(BASE/'toy-smoke')],SOURCE)
run('baseline-smoke',[fresh,'-m','scripts.run_baselines','--source','toy','--rows','100','--output-dir',str(BASE/'baseline-smoke')],SOURCE)
run('guarded-real-replay',[fresh,str(ROOT/'tools/research/verify_runs.py'),'--domain','routerbench','--seed','7','--method','pilot','--variant','main','--output','outputs/research-experiments-20261029/verification-reproduction/routerbench-seed7-pilot-main-guarded.json'],ROOT)
report['verified']=True
report['scope']='fresh source snapshot and locked CPU environment; full real group replay uses existing immutable inputs, not a full-grid or GPU re-embedding rerun'
(BASE/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('POST-REVIEW REPRODUCTION COMPLETE',flush=True)

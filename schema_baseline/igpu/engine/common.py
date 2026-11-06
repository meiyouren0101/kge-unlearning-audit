"""Portable file helpers and pinned upstream inventory, no execution service."""
from pathlib import Path
import csv,json,math,os
import io_utils as io
ROOT=Path(__file__).resolve().parent
def ROOT_OFFICIAL():
    return Path(os.environ.get('KGE_GRAPHDPO_SOURCE', ROOT.parents[1]/'.upstream/GraphDPO')).resolve()
def ROOT_VENDOR():
    return Path(os.environ.get('KGE_ADVERTORCH_SOURCE', ROOT.parents[1]/'.upstream/advertorch')).resolve()
SEEDS=(0,42,123)
METHODS=('M','M_no_gate','M_no_repair','M_no_real')
GRAPH_ARMS=('GraphDPO_A','GraphDPO_B','T_full',*METHODS)
def write(path,obj):io.write(path,obj)
def csvwrite(path,rows):
    rows=list(rows);keys=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with Path(path).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def info(folder):return io.read(Path(folder)/'segment.json')
def official_check():
    r=io.read(ROOT/'official_receipt.json')
    for n,v in r['files'].items():io.require(io.sha(ROOT_OFFICIAL()/n)==v['sha256'],'Changed pinned upstream source '+n)
    return r
def stage(folder,name,full=False):
    f=Path(folder)/'stages'/name;m=io.read(f/'commit.json')
    for n,h in m['files'].items():io.require(io.sha(f/n)==h,'Changed reference fixture file')
    return io.read(f/'result.json'),f,m
def seal(folder):
    folder=Path(folder);write(folder/'commit.json',dict(files={f.name:io.sha(f) for f in folder.iterdir() if f.is_file() and f.name!='commit.json'}))

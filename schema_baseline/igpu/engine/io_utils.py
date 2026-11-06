"""Small stdlib IO; never imports ML or opens assets on import."""
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import sys
import uuid


def require(ok,message):
    if not ok:
        raise ValueError(message)


def sha(path):
    path=Path(path)
    require(path.is_file() and not path.is_symlink(),'Missing/symlinked file: '+str(path))
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):
            h.update(b)
    return h.hexdigest()


def child(root,name):
    require(type(name) is str and name and '\\' not in name and ':' not in name
            and not name.startswith('/') and PurePosixPath(name).as_posix()==name
            and all(x not in ('','.','..') for x in name.split('/')),'Unsafe path')
    out=Path(root).resolve(strict=True)
    for x in name.split('/'):
        out/=x;require(not out.is_symlink(),'Internal symlink')
    return out


def decode(raw):
    require(len(raw)<=8*1024**2,'Small JSON bound')
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,'Duplicate JSON key');out[k]=v
        return out
    def number(value):
        result=float(value)
        require(math.isfinite(result),'Nonfinite JSON number')
        return result
    return json.loads(raw.decode('utf-8'),object_pairs_hook=pairs,parse_float=number,
                      parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))


def read(path):
    path=Path(path)
    require(path.is_file() and not path.is_symlink(),'Small input missing/symlinked')
    return decode(path.read_bytes())


def write(path,obj):
    path=Path(path);require(not path.is_symlink(),'Output symlink')
    raw=(json.dumps(obj,ensure_ascii=True,allow_nan=False,indent=2)+'\n').encode()
    require(len(raw)<=8*1024**2,'Small output bound')
    with path.open('xb') as f:
        f.write(raw)


def load(name,path):
    before=sys.path[:]
    try:
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:]=before


def stamp():
    return dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ_')+uuid.uuid4().hex[:8]


def delivery(root):
    root=Path(root)
    m=read(root/'PACKAGE_SHA256.json')
    require({p.name for p in root.iterdir() if p.is_file() and p.suffix in ('.py','.sh')}<=set(m['files']),
            'Unlisted executable')
    for n,d in m['files'].items():
        require(sha(child(root,n))==d,'Package changed: '+n)
    return read(root/'protocol.json')

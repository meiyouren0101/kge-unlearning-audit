"""Fetch dependencies separately from this distribution; no upstream code is bundled."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'igpu/engine'
SOURCES = {
    'GraphDPO': ('https://github.com/ljj-007/GraphDPO.git',
                 '2b0aa9597fc0c9a4b122f21d2935265ad60da1f7', ENGINE / 'official_receipt.json'),
    'ZOWFKGIF': ('https://github.com/NKUShaw/ZOWFKGIF.git',
                 '9672e98756985576024818e8ac7bbee56f24bb89', ROOT / 'baselines/zowfkgif_pins.json'),
}

def git(folder, *args):
    return subprocess.check_output(['git', '-C', str(folder), '-c', 'core.autocrlf=false', *args])

def verify(folder, receipt):
    files = receipt['files']
    rows = files.items() if isinstance(files, dict) else ((v['path'], v) for v in files)
    for name, row in rows:
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Upstream file differs: ' + name)

def fetch(root, name):
    out = root / name
    out.mkdir(parents=True, exist_ok=False)
    if name in SOURCES:
        url, commit, pin = SOURCES[name]
        git(out, 'init', '--quiet')
        git(out, 'remote', 'add', 'origin', url)
        git(out, 'config', 'remote.origin.promisor', 'true')
        git(out, 'config', 'remote.origin.partialclonefilter', 'blob:none')
        git(out, 'fetch', '--quiet', '--depth=1', '--filter=blob:none', 'origin', commit)
        if git(out, 'rev-parse', 'FETCH_HEAD').decode().strip() != commit:
            raise ValueError('Unexpected fetched commit')
        receipt = json.loads(pin.read_text('utf-8'))
        files = receipt['files']
        names = list(files) if isinstance(files, dict) else [v['path'] for v in files]
        # Materialize only pinned files; no datasets, notebooks or checkpoints.
        for path in names:
            dest = out / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(git(out, 'show', commit + ':' + path))
    else:
        receipt = json.loads((ENGINE / 'vendor_receipt.json').read_text('utf-8'))
        with urllib.request.urlopen(receipt['url'], timeout=120) as response:
            raw = response.read()
        if hashlib.sha256(raw).hexdigest() != receipt['distribution_sha256']:
            raise ValueError('Unexpected advertorch distribution hash')
        with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
            for name in receipt['files']:
                member = archive.getmember('advertorch-0.2.3/' + name)
                if not member.isfile():
                    raise ValueError('Expected regular dependency file')
                target = out / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
    verify(out, receipt)
    print('Verified', out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('names', nargs='+', choices=(*SOURCES, 'advertorch'))
    ap.add_argument('--dest', type=Path, default=ROOT / '.upstream')
    args = ap.parse_args()
    for name in args.names:
        fetch(args.dest, name)

if __name__ == '__main__':
    main()


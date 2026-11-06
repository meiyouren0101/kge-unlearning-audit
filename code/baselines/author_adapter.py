"""Run the author's exact CUDA functions with a GraphDPO TransE score adapter."""
from igpu import bootstrap
import ast
import os
import ctypes
import shutil
import subprocess
from pathlib import Path

import common as c
import graph_data as gd
import io_utils as io

SOURCE = Path(os.environ.get('KGE_ZOWFKGIF_SOURCE', Path(__file__).resolve().parents[1]/'.upstream/ZOWFKGIF')).resolve()
PINS = Path(__file__).with_name('zowfkgif_pins.json')
DEFAULTS = dict(epsilon=1e-5, gamma=1.0, iteration=100, damp=0.0, scale=10)
NAMES = {'calculate_zo_gradients', 'woodfisher_hvps', 'GIF_unleanring'}


def verify():
    for row in io.read(PINS)['files']:
        io.require(io.sha(io.child(SOURCE, row['path'])) == row['sha256'], 'Author source changed')


def functions(cpu_fixture=False):
    import torch
    verify()
    path = SOURCE / 'ZO.py'
    nodes = [n for n in ast.parse(path.read_text('utf-8')).body
             if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    io.require({n.name for n in nodes} == NAMES, 'Official function roster')
    tree = ast.Module(body=nodes, type_ignores=[])
    if cpu_fixture:
        class CpuDevices(ast.NodeTransformer):
            def visit_Call(self, node):
                node = self.generic_visit(node)
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'cuda':
                    io.require(not node.args and not node.keywords, 'Unexpected CUDA call in fixture')
                    return node.func.value
                return node

            def visit_Constant(self, node):
                if node.value == 'cuda':
                    return ast.copy_location(ast.Constant('cpu'), node)
                return node
        tree = CpuDevices().visit(tree)
    ns = dict(torch=torch)
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), ns)
    return ns['GIF_unleanring']


def build(out):
    verify()
    runtime = out / 'openke'
    for row in io.read(PINS)['files']:
        name = row['path']
        if name.startswith('openke/'):
            dst = out / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE / name, dst)
    (runtime / 'release').mkdir()
    binary = runtime / 'release/Base.so'
    subprocess.run(['g++', str(runtime/'base/Base.cpp'), '-fPIC', '-shared',
                    '-o', str(binary), '-pthread', '-O3'], cwd=out, check=True, timeout=600)
    return dict(path=str(binary),sha256=io.sha(binary))


def make_loaders(binary, out, data, phase, seed):
    io.require(io.sha(binary['path']) == binary['sha256'], 'Native sampler changed')
    runtime = Path(binary['path']).parents[1]
    cls = io.load('author_sampler', runtime / 'data/TrainDataLoader.py').TrainDataLoader
    libc = ctypes.CDLL(None)
    libc.srand.argtypes = [ctypes.c_uint]
    libc.srand(seed)
    prior = data['triples'] if phase == 0 else gd.at(data, phase - 1)['retained']
    retained = gd.at(data, phase)['retained']
    for name, facts in (('before', prior), ('retained', retained)):
        with (out / (name+'.txt')).open('x', encoding='ascii', newline='\n') as f:
            f.write(f'{2*len(facts)}\n')
            for h, r, t in gd.queries(facts):
                f.write(f'{h} {t} {r}\n')
    for name, n in (('entities', data['counts']['entities']), ('relations', 2*data['counts']['relations'])):
        (out / (name+'.txt')).write_text(f'{n}\n' + ''.join(f'{i}\t{i}\n' for i in range(n)), encoding='ascii')
    options = dict(in_path=None, ent_file=str(out/'entities.txt'), rel_file=str(out/'relations.txt'),
                   nbatches=100, threads=8, sampling_mode='normal', bern_flag=1, filter_flag=1,
                   neg_ent=25, neg_rel=0)
    return tuple(cls(tri_file=str(out/(n+'.txt')), **options) for n in ('before','retained'))


def apply(model, first, second, cpu_fixture=False):
    import torch
    import resource_bridge as bridge
    import legacy_runtime as rt

    class Adapter(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.calls = 0

        def model(self, batch):
            self.calls += 1
            if bridge.ACTIVE is not None:
                bridge.ACTIVE.anchor(f'author_score/{self.calls}', rt.tensor_tree_hash((batch, self.base.state_dict())))
            ent, rel = self.base.embedding()
            return self.base.score_fun(ent[batch['batch_h']], rel[batch['batch_r']], ent[batch['batch_t']])

    adapter = Adapter(model)
    with torch.no_grad():
        values = functions(cpu_fixture)(adapter, first, second, **DEFAULTS)
        parameters = tuple(adapter.parameters())
        io.require(len(parameters) == len(values) == 2, 'Official broadcast parameter roster')
        errors, norms = [], []
        for parameter, value in zip(parameters, values):
            delta = value - parameter
            errors.append(float((delta - delta[:1]).abs().max()))
            norms.append(float(delta[0].norm()))
            parameter.copy_(value)
    return dict(defaults=DEFAULTS, score_calls=adapter.calls, max_broadcast_error=errors, row_update_l2=norms)

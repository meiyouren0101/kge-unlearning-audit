"""Run paper settings serially and emit one final-request CSV per table/setting."""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0,42,123)
BASE = ('P','R','GraphDPO','IGPU')
ABL = ('IGPU','IGPU-no-gate','IGPU-no-repair','IGPU-no-real','IGPU-push-only')
FULL = ('P','R','Finetune','FT-lr1e-4','RL','NG','Fisher-stream','BS','ADV-IMP','SSD','GraphDPO-code','GraphDPO','T_full','IGPU')
FAIR = (*BASE,'IGPU-gate-c15','IGPU-uniform15','GraphDPO+repair','B5','B10')
MATCH = tuple(f'GraphDPO+repair-{n}epoch' for n in (1,2,3,5))
TABLES = ('5','7','8','10','11','C1','C2','C3','C4','C5','C6','C7','C-bootstrap','C-rotate-repair','G','YAGO')
FAIR_TRUTH = 'request_mean_over_true_answers'

def table_plan(t):
    if t == '11': return [('FB','RotatE',BASE),('WN','RotatE',(*BASE,'IGPU-calibrated')),('CL','TransE',BASE)]
    if t == 'C-rotate-repair': return [('FB','RotatE',(*BASE,'IGPU-half-repair','IGPU-no-repair'))]
    if t == 'YAGO': return [('YA','TransE',BASE)]
    if t == 'C4': return [('WN','TransE',('P','R','IGPU','IGPU-tail-degree','IGPU-shuffled','IGPU-no-gate','IGPU-gate-c20'))]
    if t == 'C5': return [(g,'TransE',('P','R')) for g in ('FB','WN','CL')]
    basic = {
        '5':('P','R','FT-lr1e-4','RL','Fisher-stream','SSD','GraphDPO-code','GraphDPO'),
        '7':BASE,'8':FAIR,'10':('P','R',*ABL),'C1':BASE,'C2':FULL,'C3':tuple(dict.fromkeys((*FULL,*ABL[:-1]))),
        'C6':(*BASE,'GraphDPO-code','GraphDPO-native','GraphDPO-code-native'),
        'C7':(*BASE,'IGPU-gate-c05','IGPU-gate-c15','IGPU-repair-x05','IGPU-repair-x2'),
        'C-bootstrap':(*BASE,'GraphDPO+repair'),
        'G':(*FULL,*[m for m in ABL if m!='IGPU'],'GraphDPO+repair'),
    }[t]
    return [(g,'TransE',basic+(MATCH if t in ('8','C-bootstrap') and g=='WN' else
                             ('IGPU-gate-c20',) if t=='C7' and g=='WN' else ())) for g in ('FB','WN')]

def invoke(module, *args):
    subprocess.run([sys.executable,'-B','-m',module,*map(str,args)],cwd=ROOT,check=True)

def settings(backbone, method):
    return ROOT/'configs'/('RotatE' if backbone=='RotatE' else '')/(method+'.json')

def run_path(work, graph, backbone, method, seed):
    return work/'runs'/graph/backbone/method/str(seed)

def ensure(work, graph, backbone, method, seed, data, device, config=None, calibration_only=False):
    dest = run_path(work,graph,backbone,method,seed)
    if (dest/'COMPLETE.json').is_file():
        info = json.loads((dest/'run.json').read_text('utf-8'))
        if (info['dataset'],info['backbone'],info['method'],info['seed'],info['device']) != (graph,backbone,method,seed,device):
            raise ValueError('Unexpected cached run metadata: '+str(dest))
        chosen = config or settings(backbone,method)
        expected = json.loads(chosen.read_text('utf-8'))
        if graph in ('CL','YA') or backbone=='RotatE':
            expected['evaluation']='sampled'
        if graph in ('CL','YA'):
            expected['query_batch_size']=4
        if json.loads((dest/'config.json').read_text('utf-8')) != expected:
            raise ValueError('Changed cached configuration: '+str(dest))
        return dest
    cmd = ['--config',config or settings(backbone,method),'--data',data,'--seed',seed,'--device',device,'--out',dest]
    if method!='P':
        cmd += ['--pretrained', run_path(work,graph,backbone,'P',seed)/'pretrained.pt']
    if backbone=='RotatE' or graph in ('CL','YA'):
        cmd += ['--evaluation','sampled']
    if calibration_only:
        cmd += ['--calibration-only']
    invoke('igpu.run',*cmd)
    return dest

def metrics(folder, phase):
    rows=json.loads((folder/f'p{phase}/evaluation.json').read_text('utf-8'))['metrics']
    return next(r['MRR'] for r in rows if (r['filter'],r['cohort'],r['direction'])==('complete','cumulative_forget','combined'))

def calibrated(work, data, device):
    target=metrics(run_path(work,'WN','RotatE','GraphDPO',0),0)
    values={c:metrics(ensure(work,'WN','RotatE',f'IGPU-c{c}',0,data,device,calibration_only=True),0) for c in (2,3,4)}
    passing=[c for c in values if values[c]<=target]
    chosen=min(passing) if passing else 4
    directory=work/'calibration';directory.mkdir(exist_ok=True)
    selected=directory/'IGPU-calibrated.json'
    cfg=json.loads(settings('RotatE',f'IGPU-c{chosen}').read_text('utf-8'));cfg['name']='IGPU-calibrated'
    (directory/'selection.json').write_text(json.dumps(dict(target=target,candidates=values,selected=chosen,fallback=not passing,R_used=False),indent=2)+'\n','utf-8')
    selected.write_text(json.dumps(cfg,indent=2)+'\n','utf-8')
    return selected

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def csvwrite(path, rows):
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)

def table_csv(out, table, folders, graph):
    audit=out/'audit'
    task={r['method']:r for r in read(audit/'task_metrics_mean.csv') if (r['phase'],r['endpoint'],r['filter'],r['direction'])==('3','primary','complete','combined')}
    audit_pref={r['method']:r['MRR'] for r in read(audit/'preferred_mean.csv') if (r['phase'],r['endpoint'],r['population'],r['direction'])==('3','primary','unobserved_original_preferred','combined')}
    if table=='8':
        pfile=out/'probes'/'preferred_probe_mean.csv'
        fair_pref={r['method']:r['MRR'] for r in read(pfile) if (r['phase'],r['population'],r['direction'])==('3','unobserved_original_preferred','combined')}
        truth={r['method']:r['DeltaMRR_vs_P'] for r in read(out/'probes'/'true_answers_mean.csv') if (r['phase'],r['direction'],r['weighting'])==('3','combined',FAIR_TRUTH)}
    else:
        fair_pref={};truth={}
    d={}
    for r in read(audit/'cross_bins_mean.csv'):
        if (r['phase'],r['endpoint'],r['direction'],r['deletion_phase'])==('3','primary','combined','all'):
            d[r['method'],r['cohort'],r['R_bin']]=r
    epoch={}
    if table=='C6':
        for f in folders:
            method=f.parent.name
            epoch.setdefault(method,[])
            for p in range(4):
                training=json.loads((f/f'p{p}/training.json').read_text('utf-8'))
                v=training.get('selected_epoch')
                if v is None and training.get('trace'):
                    v=training['trace'][-1]['epoch']
                if v is not None:epoch[method].append(v)
    match,match_valid=None,False
    if graph=='WN' and table in ('8','C-bootstrap'):
        roster=[*MATCH,'GraphDPO+repair']
        diffs={m:abs(float(task[m]['deleted_MRR'])-float(task['IGPU']['deleted_MRR'])) for m in roster}
        match=min(diffs,key=diffs.get);match_valid=diffs[match]<=.01
        (out/'matched_deletion_selection.json').write_text(json.dumps(dict(arm=match,within_tolerance=match_valid,tolerance=.01,
           absolute_MRR_differences=diffs,selection_uses_only_deleted_MRR=True),indent=2)+'\n','utf-8')
    rows=[]
    for m,r in task.items():
        if table=='10' and m in ('P','R'):
            continue
        if table=='8' and m in MATCH and (m!=match or not match_valid):
            continue
        key='R_control' if m=='R' else m
        vals=dict(method=key,deleted_MRR=r['deleted_MRR'],retained_MRR=r['retained_MRR'],MRR_Avg=r['MRR_Avg'],MRR_F1=r['MRR_F1'])
        for name,cohort,bin_name in [('high','cumulative_forget','R_le10'),('middle','cumulative_forget','R_11to100'),
              ('low','cumulative_forget','R_gt100'),('focus','cumulative_forget','focus_Pb_le10_Ra_gt100'),
              ('all_deleted','cumulative_forget','all'),('all_retained','all_retained','all')]:
            for metric in ('abs','signed'):
                vals[f'{name}_{metric}']=d[key,cohort,bin_name][metric]
        if m=='P' or m.startswith(('GraphDPO','IGPU')) or m in ('T_full','B5','B10'):
            vals['substitute_MRR']=fair_pref.get(m,audit_pref.get(m))
            if table=='11' and audit_pref.get('P') and audit_pref.get(m):
                denom=float(audit_pref['P'])
                vals['substitute_ratio_to_P']=float(audit_pref[m])/denom if denom else None
        if table=='8':
            vals['true_answer_DeltaMRR_vs_P']=truth.get(m)
            if m in MATCH:vals['substitute_population']='matched control current-request preferred roster'
            elif m in fair_pref:vals['substitute_population']='fair-control matched-context sample'
        if epoch.get(m):
            e=epoch[m];vals.update(selected_epoch_mean=math.fsum(e)/len(e),selected_epoch_min=min(e),selected_epoch_max=max(e))
        rows.append(vals)
    csvwrite(out/'table.csv',rows)
    return match if match_valid else None

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--table',required=True,choices=TABLES)
    ap.add_argument('--dataset',choices=('FB','WN','CL','YA'))
    ap.add_argument('--data-root',type=Path,default=Path('prepared'))
    ap.add_argument('--work',type=Path,default=Path('outputs'))
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--plan',action='store_true')
    a=ap.parse_args()
    t=a.table; plan=[p for p in table_plan(t) if not a.dataset or p[0]==a.dataset]
    if not plan:
        ap.error('Requested table has no such dataset')
    if a.plan:
        print(json.dumps(dict(table=t,settings=plan,seeds=SEEDS,requests=4),indent=2))
        return
    work=a.work.resolve()
    for graph,backbone,methods in plan:
        data=(a.data_root/graph).resolve()
        dest=work/'tables'/t/graph/backbone;dest.mkdir(parents=True,exist_ok=False)
        folders=[]
        for method in methods:
            cfg=calibrated(work,data,a.device) if method=='IGPU-calibrated' else None
            for seed in SEEDS:
                folders.append(ensure(work,graph,backbone,method,seed,data,a.device,cfg))
        invoke('igpu.audit','--runs',*folders,'--data',data,'--out',dest/'audit')
        if t=='8':
            probe_folders=[f for f in folders if f.parent.name not in MATCH]
            invoke('igpu.probes','--runs',*probe_folders,'--data',data,'--out',dest/'probes','--device',a.device)
        selected=table_csv(dest,t,folders,graph)
        if t in ('C3','C5'):
            from igpu import data as dat
            if t=='C3':
                from audit import rebound
                rebound.compute(folders,dest)
            else:
                from audit import appendix_auc
                appendix_auc.compute(folders,dat.load(data),dest)
        if t=='G':
            invoke('igpu.deletion_attack','--runs',*folders,'--out',dest/'inference')
        if t=='C-bootstrap':
            invoke('igpu.paired_bootstrap','--runs',*folders,'--data',data,'--out',dest/'bootstrap_equal_repair')
            if graph=='WN' and selected is not None:
                invoke('igpu.paired_bootstrap','--runs',*folders,'--data',data,'--right',selected,'--out',dest/'bootstrap_matched_deletion')
        print('Wrote',dest,flush=True)

if __name__=='__main__':
    sys.path.insert(0,str(ROOT))
    main()


"""Fixed-rule FB p3 illustrations, restricted to the common fair probe subset."""
from . import bootstrap
from pathlib import Path
import argparse,csv,json,math
import numpy as np
import common as c,case_kernels as ck
from . import audit,data as dat

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--probes',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--labels');a=p.parse_args()
    data=dat.load(a.data)
    if data['dataset']!='FB':raise ValueError('Fixed case design is FB only')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);aliases={'IGPU':'M','GraphDPO':'GraphDPO_B','GraphDPO+repair':'B_repair'};reg={}
    for f in map(Path,a.runs):md=audit.verify(f);reg[aliases.get(md['method'],md['method']),md['seed']]=f
    raw=Path(a.data)/'all_data.txt';entities={};relations={}
    for line in raw.read_text('utf-8').splitlines():
        h,r,t=line.split('\t')
        for x in (h,t):
            if x not in entities:entities[x]=len(entities)
        if r not in relations:relations[r]=2*len(relations)
    es={v:k for k,v in entities.items()};rs={v:k for k,v in relations.items()};labels=json.loads(Path(a.labels).read_text('utf-8')) if a.labels else {}
    rows=[];truth=set(data['triples']);fids={q:i for i,q in enumerate(data['triples'])}
    for seed in (0,42,123):
        required=(*ck.METHODS,'P')
        if any((m,seed) not in reg for m in required):raise ValueError('Missing a common required method/seed')
        loc=Path(a.probes)/f's{seed}_p3';positions=list(csv.DictReader((loc/'positions.csv').open(encoding='utf-8-sig')))
        cached={}
        for m in required:
            keys,dm,values,e=audit.cache(reg[m,seed],3,'primary');lookup={tuple(k):j for j,k in enumerate(keys.tolist())}
            actual_name=json.loads((reg[m,seed]/'run.json').read_text('utf-8'))['method']
            ranks=np.fromfile(loc/actual_name/'actual.u32le',dtype='<u4').reshape(-1,5);cached[m]=(lookup,values,ranks)
        for pos in positions:
            i=int(pos['request_index']);fact=data['phases'][3][i];h,r,t=fact;head=data['is_heads'][3][i];q=(t,r+1,h) if head else fact
            row=dict(seed=seed,fact_id=fids[fact],direction='inverse' if head else 'forward',subject_id=es[h],relation_id=rs[r],object_id=es[t],
                subject_name=labels.get(es[h]),object_name=labels.get(es[t]),source_population='fixed fair p3 common probe subset, <=2048 requests/seed; not all preferred')
            for m in required:
                lookup,values,ranks=cached[m];j=int(pos['anchor_id'])
                if q not in lookup:raise ValueError('Case fact absent in common full task cache')
                row[m+'_rank']=int(values[lookup[q]]);row[m+'_preferred_rank']=int(ranks[j,4])
                if m=='GraphDPO_B':
                    aq=tuple(map(int,ranks[j,:3]));canonical=(aq[2],aq[1]-1,aq[0]) if aq[1]%2 else aq
                    row['preferred_unobserved']=canonical not in truth;row['preferred_entity_id']=es[aq[2]];row['preferred_entity_name']=labels.get(es[aq[2]])
            rows.append(row)
    first,second,counts=ck.select(rows);selected=[dict(panel='deleted',**r) for r in first]+[dict(panel='preferred',**r) for r in second]
    c.csvwrite(out/'case_table.csv',selected)
    (out/'selection.json').write_text(json.dumps(dict(counts,thresholds='R<=10,M<=30,B+repair>100,B>100; sort log(B)-log(R) descending; distinct facts; never relax thresholds',
        note='Outcome-selected illustrations, not representative samples. Missing human labels remain null.'),indent=2)+'\n')
if __name__=='__main__':main()

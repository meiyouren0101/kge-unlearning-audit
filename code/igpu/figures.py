"""Figures generated only from supplied audit CSVs; no manuscript values embedded."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,json,math
def read(folder,name):
    with (Path(folder)/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
    p=argparse.ArgumentParser();p.add_argument('--audit');p.add_argument('--figure',choices=('overview','tradeoff','stratified','substitutes','sensitivity'),required=True);p.add_argument('--out',required=True);a=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);source=[]
    with plt.rc_context({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42}):
        if a.figure=='overview':
            fig,ax=plt.subplots(figsize=(9,2.8),layout='constrained');ax.axis('off')
            labels=['Current retained graph\ntrue targets + five features','S3 and q per direction\nw = 2(1-q)','20-epoch deletion\ncontinuous deletion Adam','Retained-only repair\nfresh repair Adam']
            for i,label in enumerate(labels):
                ax.text(.12+.25*i,.55,label,ha='center',va='center',transform=ax.transAxes,bbox=dict(boxstyle='round',facecolor='#e7f0f7',edgecolor='#24587a'))
                if i<3:ax.annotate('',xy=(.28+.25*i,.55),xytext=(.21+.25*i,.55),xycoords='axes fraction',arrowprops=dict(arrowstyle='->'))
            ax.text(.5,.12,'Next request uses repaired parameters and previous deletion-Adam state; rolling teacher = actual request start.',ha='center',transform=ax.transAxes)
        elif a.figure=='tradeoff':
            fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
            for row in read(a.audit,'task_metrics_mean.csv'):
                if (row['phase'],row['endpoint'],row['filter'],row['direction'])!=('3','primary','complete','combined') or not row['retained_MRR']:continue
                x=float(row['deleted_MRR']);y=float(row['retained_MRR']);ax.scatter(x,y,marker='o',s=25);ax.annotate(row['method'],(x,y),fontsize=7);source.append(row)
            ax.set(xlabel='Deleted-fact MRR (ranking proxy)',ylabel='Retained-fact MRR',title='Final endpoint: forgetting versus retained performance')
        elif a.figure in ('stratified','sensitivity'):
            fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');bins=['all','R_le10','R_11to100','R_gt100','focus_Pb_le10_Ra_gt100']
            values=read(a.audit,'cross_bins_mean.csv');roles=read(a.audit,'cross_bins_by_combination.csv');methods=['R_control','IGPU','GraphDPO+repair','B5','GraphDPO'] if a.figure=='stratified' else ['IGPU','IGPU-gate-c05','IGPU-gate-c15','IGPU-repair-x05','IGPU-repair-x2']
            for ax,cohort in zip(axes,('cumulative_forget','all_retained')):
                for j,method in enumerate(methods):
                    lookup={r['R_bin']:r for r in values if (r['method'],r['phase'],r['endpoint'],r['cohort'],r['direction'],r['deletion_phase'])==(method,'3','primary',cohort,'combined','all')}
                    if not lookup:continue
                    for k,b in enumerate(bins):
                        if b not in lookup or not lookup[b]['abs_log_mean']:
                            ax.text(k+(j-2)*.15,.02,'NA',transform=ax.get_xaxis_transform(),rotation=90,ha='center',fontsize=5);continue
                        v=float(lookup[b]['abs_log_mean']);x=k+(j-2)*.15
                        rr=[float(r['abs_log_mean']) for r in roles if (r['method'],r['phase'],r['endpoint'],r['cohort'],r['direction'],r['deletion_phase'],r['R_bin'])==(method,'3','primary',cohort,'combined','all',b) and r['abs_log_mean']]
                        ax.bar(x,v,width=.14,label=method if k==0 else None,color='C'+str(j),hatch=['','//','..','xx','\\\\'][j])
                        if rr:ax.errorbar(x,v,yerr=[[max(v-min(rr),0)],[max(max(rr)-v,0)]],color='#303030',capsize=2,fmt='none',lw=.6)
                        source.append(dict(lookup[b],role_min=min(rr) if rr else None,role_max=max(rr) if rr else None,uncertainty='six dependent role min/max, not CI'))
                ax.set(xticks=range(len(bins)),xticklabels=['All','R<=10','11-100','R>100','Focus'],ylabel='Mean per-query |log rank gap|',title=cohort,ylim=(0,None));ax.tick_params(axis='x',labelrotation=25);ax.legend(fontsize=6)
        else:
            fig,(ax,gap)=plt.subplots(2,1,figsize=(7,7),layout='constrained');groups=defaultdict(list)
            for row in read(a.audit,'preferred_mean.csv'):
                if (row['endpoint'],row['population'],row['direction'])==('primary','unobserved_original_preferred','combined') and row['MRR']:groups[row['method']].append(row)
            for i,(method,rows) in enumerate(sorted(groups.items())):
                rows=sorted(rows,key=lambda r:int(r['phase']));ax.plot([int(r['phase'])+1 for r in rows],[float(r['MRR']) for r in rows],marker=['o','s','^','D','v'][i%5],label=method);source.extend(rows)
            ax.set(xlabel='Deletion request',ylabel='Original preferred MRR (designated side)',yscale='log',xticks=[1,2,3,4]);ax.legend(fontsize=6)
            bins=read(a.audit,'cross_bins_mean.csv')
            for cohort,binname,label,marker in (('cumulative_forget','R_le10','Deleted R_a<=10','o'),('all_retained','all','All retained','s')):
                xs=[];ys=[]
                for phase in range(4):
                    pair={r['method']:r for r in bins if (r['phase'],r['endpoint'],r['cohort'],r['direction'],r['deletion_phase'],r['R_bin'])==(str(phase),'primary',cohort,'combined','all',binname) and r['method'] in ('IGPU','GraphDPO+repair')}
                    if len(pair)==2 and all(r['abs_log_mean'] for r in pair.values()):
                        y=float(pair['GraphDPO+repair']['abs_log_mean'])-float(pair['IGPU']['abs_log_mean']);xs.append(phase+1);ys.append(y)
                        source.append(dict(panel='gap',cohort=cohort,R_bin=binname,phase=phase,gap=y))
                    else:xs.append(phase+1);ys.append(float('nan'))
                if any(math.isfinite(y) for y in ys):gap.plot(xs,ys,marker=marker,label=label)
            if not gap.lines:gap.text(.5,.5,'Required paired values unavailable',transform=gap.transAxes,ha='center',va='center')
            else:gap.legend(fontsize=7)
            gap.axhline(0,color='#aaaaaa',lw=.6);gap.set(xlabel='Deletion request',ylabel='B+repair minus IGPU absolute gap',xticks=[1,2,3,4])
        if a.figure!='overview':
            for ax in fig.axes:ax.grid(axis='y',alpha=.2)
        for ext in ('pdf','png'):fig.savefig(out/(a.figure+'.'+ext),dpi=300)
        plt.close(fig)
    keys=list(dict.fromkeys(k for r in source for k in r)) or ['status']
    with (out/'plotted_values.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(source)
    (out/'figure_spec.json').write_text(json.dumps(dict(panel=a.figure,source=a.audit,general_draft=True,publisher_format_not_certified=True,missing='Missing cells omitted, never filled with0',uncertainty='No CI inferred from dependent roles',original_values='read supplied CSV only'),indent=2)+'\n')
if __name__=='__main__':main()

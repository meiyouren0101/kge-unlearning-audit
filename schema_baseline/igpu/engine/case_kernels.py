import math
METHODS=("R","M","B_repair","GraphDPO_B")

def select(rows):
    complete=[r for r in rows if all(r.get(a+'_rank') is not None and r.get(a+'_preferred_rank') is not None for a in METHODS)]
    eligible=[r for r in complete if r['R_rank']<=10 and r['B_repair_rank']>100 and r['GraphDPO_B_rank']>100 and r['M_rank']<=30]
    key=lambda r:(r['seed'],r['fact_id'],r['direction'])
    first=sorted(eligible,key=lambda r:(-(math.log(r['GraphDPO_B_rank'])-math.log(r['R_rank'])),*key(r)))
    def unique(values,n,excluded=()):
        used=set(excluded);out=[]
        for r in values:
            if r['fact_id'] not in used:out.append(r);used.add(r['fact_id'])
            if len(out)==n:break
        return out
    first=unique(first,5)
    promoted=[r for r in complete if r['preferred_unobserved'] and r['GraphDPO_B_preferred_rank']<r['R_preferred_rank']
              and r.get('P_preferred_rank') is not None and r['GraphDPO_B_preferred_rank']<r['P_preferred_rank']]
    second=sorted(promoted,key=lambda r:(r['GraphDPO_B_preferred_rank'],*key(r)))
    second=unique(second,3,[r['fact_id'] for r in first])
    return first,second,dict(common_complete_rows=len(complete),eligible_rows=len(eligible),
        eligible_distinct_facts=len({r['fact_id'] for r in eligible}),selected_threshold_cases=len(first),
        promoted_distinct_facts=len({r['fact_id'] for r in promoted}),selected_preferred_cases=len(second))

import math
METHODS=("M","M_no_gate","M_no_repair","M_no_real","M_repair05","M_norepair")

def budget(retained):return (238*math.ceil(2*retained/1024)+50)//100

def has_repair(arm):return arm in METHODS and arm not in ('M_no_repair','M_norepair')

def repair_budget(arm,retained):
    value=budget(retained)
    return (value+1)//2 if arm=='M_repair05' else value if has_repair(arm) else 0

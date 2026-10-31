"""Explicit views of the verified shared prior for binary routing."""
import copy
import numpy as np


def restrict_prior(parent,arms):
    if (not isinstance(arms,(list,tuple)) or len(arms)!=2 or len(set(arms))!=2
            or any(a not in parent['arms'] for a in arms)):
        raise ValueError('Binary view requires two distinct verified parent arms')
    result=copy.deepcopy(parent);positions=[parent['arms'].index(a) for a in arms]
    result['arms']=list(arms);result['theta']=np.asarray(parent['theta'])[positions].copy()
    validation=result['metrics']['validation']
    validation['per_arm']={a:validation['per_arm'][a] for a in arms}
    validation['blockers']={a:v for a,v in validation.get('blockers',{}).items() if a in arms}
    validation['all_lambdas_ready']=not validation['blockers']
    return result

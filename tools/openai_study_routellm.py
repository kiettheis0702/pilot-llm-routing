"""Cache-only RouteLLM MF mathematics; source architecture pinned separately."""
import numpy as np


def mf_scores(query,model_embeddings,projection,classifier):
    x=np.asarray(query,dtype=float);models=np.asarray(model_embeddings,dtype=float)
    W=np.asarray(projection,dtype=float);head=np.asarray(classifier,dtype=float).reshape(-1)
    if (x.ndim!=1 or models.ndim!=2 or W.ndim!=2 or W.shape[1]!=len(x)
            or models.shape[1]!=W.shape[0] or head.shape!=(W.shape[0],)
            or not all(np.isfinite(v).all() for v in (x,models,W,head))):
        raise ValueError('Invalid MF dimensions or nonfinite values')
    norms=np.linalg.norm(models,axis=1,keepdims=True)
    if np.any(norms<=0):raise ValueError('MF model directions must be nonzero')
    return (models/norms) @ ((W@x)*head)

"""Instructor-XL local representation recipe; no inference API integration."""
import numpy as np

MODEL='hkunlp/instructor-xl'
REVISION='ce48b213095e647a6c3536364b9fa00daf57f436'
INSTRUCTION='Represent the user question for routing to a suitable language model:'
MAX_LENGTH=512


def format_instructor_input(instruction,text):
    if not isinstance(instruction,str) or not isinstance(text,str) or not text.strip():
        raise ValueError('Instructor input requires nonempty question text')
    # Official INSTRUCTOR.tokenize strips each element and joins without separator.
    return instruction.strip()+text.strip()


def pool_dense_normalize(token_embeddings,attention_mask,instruction_lengths,weight):
    tokens=np.asarray(token_embeddings,dtype=float);mask=np.asarray(attention_mask,dtype=float).copy()
    dense=np.asarray(weight,dtype=float);lengths=np.asarray(instruction_lengths)
    if (tokens.ndim!=3 or mask.shape!=tokens.shape[:2] or lengths.shape!=(tokens.shape[0],)
            or dense.ndim!=2 or dense.shape[1]!=tokens.shape[2] or lengths.dtype.kind not in 'iu'
            or np.any(lengths<0) or np.any(lengths>tokens.shape[1])
            or not all(np.isfinite(v).all() for v in (tokens,mask,dense)) or not np.isin(mask,[0,1]).all()):
        raise ValueError('Invalid Instructor pooling dimensions, masks or values')
    for i,n in enumerate(lengths):mask[i,:int(n)]=0
    counts=mask.sum(axis=1,keepdims=True)
    if np.any(counts<=0):raise ValueError('No text tokens remain after instruction masking')
    pooled=(tokens*mask[...,None]).sum(axis=1)/counts
    result=pooled@dense.T;norm=np.linalg.norm(result,axis=1,keepdims=True)
    if not np.isfinite(result).all() or np.any(norm<=0):raise ValueError('Invalid projected Instructor vector')
    return result/norm

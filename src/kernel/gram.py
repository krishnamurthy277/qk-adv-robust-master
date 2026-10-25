"""Content-keyed atomic Gram caches; raw metrics remain available to callers."""
import hashlib
import json
import os
import uuid
from pathlib import Path
import numpy as np
import torch
from src.kernel.fidelity import compute_kernel_matrix, KERNEL_VERSION
from src.kernel.psd_repair import make_psd
from src.utils.hashing import stable_hash


def tensor_digest(X):
    return hashlib.sha256(X.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def get_gram_matrix(X1,X2=None,reps=2,c=1.,p=0.,cache_dir=None,config_dict=None,repair=True):
    key = dict(version=KERNEL_VERSION,X1=tensor_digest(X1),
               X2=None if X2 is None else tensor_digest(X2),shape=list(X1.shape),
               reps=reps,c=c,p=p,config=config_dict or {})
    cache_path = None
    if cache_dir is not None:
        cache_dir=Path(cache_dir)
        cache_dir.mkdir(parents=True,exist_ok=True)
        cache_path=cache_dir/f'raw_{stable_hash(key)}.npy'
    if cache_path is not None and cache_path.exists():
        raw=torch.as_tensor(np.load(cache_path),dtype=torch.float64,device=X1.device)
    else:
        with torch.no_grad():
            raw=compute_kernel_matrix(X1,X2,reps,c,p)
        if cache_path is not None:
            temp=cache_path.with_name(cache_path.name+'.'+uuid.uuid4().hex+'.tmp')
            with temp.open('wb') as f: np.save(f,raw.cpu().numpy())
            os.replace(temp,cache_path)
            cache_path.with_suffix('.json').write_text(json.dumps(key,indent=2),encoding='utf-8')
    if X2 is None and repair and torch.linalg.eigvalsh(raw).min() < -1e-8:
        return make_psd(raw,epsilon=0)
    return raw

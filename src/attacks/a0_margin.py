"""A0 local linearized distance estimate (not a certified lower bound)."""
import numpy as np
import torch


def a0_linearized_margin(X_test,y_test,X_sv,alpha_y,intercept,kernel_fn):
    if len(X_test)>8:
        parts=[a0_linearized_margin(X_test[i:i+8],np.asarray(y_test)[i:i+8],
               X_sv,alpha_y,intercept,kernel_fn) for i in range(0,len(X_test),8)]
        return {key:np.concatenate([part[key] for part in parts]) for key in parts[0]}
    X=X_test.detach().clone().requires_grad_(True)
    f=kernel_fn(X,X_sv)@alpha_y+intercept
    gradient,=torch.autograd.grad(f.sum(),X)
    norms=gradient.norm(dim=1)
    y=torch.as_tensor(np.where(np.asarray(y_test)<=0,-1.,1.),dtype=X.dtype,device=X.device)
    correct=y*f>0
    margins=f.abs()/norms.clamp_min(1e-12)
    return dict(margins=margins[correct].detach().cpu().numpy(),
                grad_norms=norms[correct].detach().cpu().numpy(),
                correctly_classified=correct.detach().cpu().numpy())

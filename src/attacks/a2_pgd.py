"""Batched PGD-L2 with best clean/initial/intermediate/final iterate retention."""
import numpy as np
import torch
from src.attacks.base import clip_to_domain


def _pgd_batch(X, y, X_sv, alpha_y, intercept, kernel_fn, eps, alpha, T, R):
    if eps < 0 or T < 1 or R < 1:
        raise ValueError('Require eps >= 0, T >= 1, R >= 1')
    with torch.no_grad():
        best_loss = y * (kernel_fn(X, X_sv) @ alpha_y + intercept)
    best = X.clone()
    for restart in range(R):
        # Clean start makes R=1 a genuine no-random-restart comparison.
        x = X.clone()
        if restart:
            delta = torch.randn_like(X)
            delta = delta / delta.norm(dim=1,keepdim=True).clamp_min(1e-12)
            delta = delta * eps * torch.rand((len(X),1),dtype=X.dtype,device=X.device)
            x = clip_to_domain(X + delta)
        for step in range(T+1):
            x = x.detach().requires_grad_(True)
            loss = y * (kernel_fn(x,X_sv) @ alpha_y + intercept)
            with torch.no_grad():
                improved = loss < best_loss
                best = torch.where(improved[:,None],x,best)
                best_loss = torch.minimum(best_loss,loss)
            if step == T:
                break
            grad, = torch.autograd.grad(loss.sum(),x)
            with torch.no_grad():
                x = x - alpha*grad/grad.norm(dim=1,keepdim=True).clamp_min(1e-12)
                delta = x-X
                factor = (eps/delta.norm(dim=1,keepdim=True).clamp_min(1e-12)).clamp(max=1)
                x = clip_to_domain(X+delta*factor)
    return best.detach(),best_loss.detach()


def pgd_l2_single(x0,y_label,X_sv,alpha_y,intercept,kernel_fn,eps,alpha,T=40,R=3):
    adv,loss = _pgd_batch(x0[None],torch.tensor([y_label],dtype=x0.dtype),
                          X_sv,alpha_y,intercept,kernel_fn,eps,alpha,T,R)
    return adv[0],float(loss[0])


def a2_pgd_attack(X_test,y_test,X_sv,alpha_y,intercept,kernel_fn,eps,T=40,R=3):
    y = torch.as_tensor(np.where(np.asarray(y_test)<=0,-1.,1.),dtype=X_test.dtype,device=X_test.device)
    with torch.no_grad():
        initial = y*(kernel_fn(X_test,X_sv)@alpha_y+intercept)>0
    adv = X_test.clone()
    correct = initial.clone()
    indices = torch.where(initial)[0]
    # Bound autograd memory under the 12-worker production configuration.
    for chunk in indices.split(8):
        if not len(chunk):
            continue
        selected,loss = _pgd_batch(X_test[chunk],y[chunk],X_sv,alpha_y,intercept,
                                  kernel_fn,eps,eps/10,T,R)
        adv[chunk] = selected
        correct[chunk] = loss>0
    init_np,correct_np = initial.cpu().numpy(),correct.cpu().numpy()
    return dict(X_adv=adv,robust_acc=float(correct_np.mean()),
                attack_success_rate=float((init_np & ~correct_np).sum()/max(init_np.sum(),1)),
                initially_correct=init_np,after_attack_correct=correct_np)

"""Transfer from noiseless surrogate, scored against TARGET clean predictions."""
import numpy as np
import torch
from src.attacks.a2_pgd import a2_pgd_attack


def a1_transfer_attack(X_test,y_test,X_sv_surrogate,alpha_y_surrogate,
                       intercept_surrogate,kernel_fn_surrogate,X_sv_noisy,
                       alpha_y_noisy,intercept_noisy,kernel_fn_noisy,eps,T=40,R=3):
    surrogate = a2_pgd_attack(X_test,y_test,X_sv_surrogate,alpha_y_surrogate,
                             intercept_surrogate,kernel_fn_surrogate,eps,T,R)
    adv = surrogate['X_adv']
    y = np.where(np.asarray(y_test)<=0,-1.,1.)
    with torch.no_grad():
        clean_f = (kernel_fn_noisy(X_test,X_sv_noisy)@alpha_y_noisy+intercept_noisy).cpu().numpy()
        adv_f = (kernel_fn_noisy(adv,X_sv_noisy)@alpha_y_noisy+intercept_noisy).cpu().numpy()
    initial = y*clean_f>0
    correct = initial & (y*adv_f>0)
    return dict(X_adv=adv,robust_acc=float(correct.mean()),
                attack_success_rate=float((initial & ~correct).sum()/max(initial.sum(),1)),
                initially_correct=initial,after_attack_correct=correct)

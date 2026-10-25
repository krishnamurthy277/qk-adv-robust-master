"""Isolated float64/complex128 CPU/CUDA equivalence and timing check.

Install the matching CUDA torch wheel into .gpu_packages, not the running
environment. This script imports that wheel before any project modules.
No production caches or result tables are modified.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'.gpu_packages'))
import json
import time
import statistics
import torch
from src.kernel.fidelity import feature_density,density_overlap


def main():
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError(f'CUDA unavailable with torch {torch.__version__}')
    torch.manual_seed(42)
    references=torch.rand(60,5,dtype=torch.float64)*torch.pi
    points=torch.rand(60,5,dtype=torch.float64)*torch.pi
    coefficients=torch.randn(60,dtype=torch.float64)
    rows=[]
    for reps in (1,3):
        for batch in (8,60):
            computed={}
            for device in ('cpu','cuda'):
                x0=points[:batch].to(device)
                weights=coefficients.to(device)
                with torch.no_grad():
                    fixed=feature_density(references.to(device),reps,1.,.03)
                elapsed=[];cpu_times=[]
                if device=='cuda':torch.cuda.reset_peak_memory_stats()
                for repeat in range(5):
                    x=x0.detach().clone().requires_grad_(True)
                    if device=='cuda':torch.cuda.synchronize()
                    start=time.perf_counter();cpu_start=time.process_time()
                    values=density_overlap(feature_density(x,reps,1.,.03),fixed)
                    gradient,=torch.autograd.grad((values@weights).sum(),x)
                    if device=='cuda':torch.cuda.synchronize()
                    if repeat:
                        elapsed.append(time.perf_counter()-start)
                        cpu_times.append(time.process_time()-cpu_start)
                computed[device]=(values.detach().cpu(),gradient.detach().cpu())
                rows.append(dict(reps=reps,batch=batch,device=device,
                    median_wall_s=statistics.median(elapsed),min_wall_s=min(elapsed),
                    max_wall_s=max(elapsed),median_process_cpu_s=statistics.median(cpu_times),
                    peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0))
            kernel_error=float((computed['cpu'][0]-computed['cuda'][0]).abs().max())
            gradient_relative_error=float((computed['cpu'][1]-computed['cuda'][1]).norm()/computed['cpu'][1].norm())
            for row in rows[-2:]:
                row.update(kernel_max_abs_error=kernel_error,gradient_relative_error=gradient_relative_error)
            if kernel_error>=1e-8 or gradient_relative_error>=1e-5:
                raise AssertionError((kernel_error,gradient_relative_error))
    result=dict(torch=torch.__version__,cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0),n_qubits=5,noise_p=.03,
                dtype='float64/complex128',concurrent_cpu_grid=True,measurements=rows)
    output=ROOT/'results/profile_gpu.json'
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

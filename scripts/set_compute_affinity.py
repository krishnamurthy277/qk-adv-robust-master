"""Limit only this project's active Joblib workers, without restarting cells."""
import argparse
import json
from pathlib import Path
import time
import psutil


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',type=int,required=True)
    parser.add_argument('--logical-cpus',type=int,default=10)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent.parent
    parent=psutil.Process(args.parent)
    command=' '.join(parent.cmdline())
    if 'scripts/run_experiment.py' not in command or Path(parent.cwd()).resolve()!=root:
        raise RuntimeError('Refusing to modify processes outside this project experiment')
    allowed=parent.cpu_affinity()[:args.logical_cpus]
    if len(allowed)!=args.logical_cpus or not allowed:
        raise ValueError('Invalid logical CPU count')
    workers=[p for p in parent.children(recursive=True)
             if 'popen_loky_win32' in ' '.join(p.cmdline())]
    before=[dict(pid=p.pid,created=p.create_time(),affinity=p.cpu_affinity(),priority=int(p.nice())) for p in workers]
    record=root/'results/logs'/f'affinity_{time.time_ns()}.json'
    record.write_text(json.dumps(dict(parent=args.parent,requested_affinity=allowed,before=before),indent=2))
    for worker in workers:
        worker.cpu_affinity(allowed)
        worker.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    print(json.dumps(dict(workers=len(workers),affinity=allowed,priority='below normal',record=str(record))))


if __name__=='__main__':main()

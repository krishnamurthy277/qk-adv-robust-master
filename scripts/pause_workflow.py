"""User-requested stop preserving completed checkpoints; no automatic resume."""
from datetime import datetime,timezone
import json
from pathlib import Path
import os
import psutil
import pandas as pd


def main():
    root=Path(__file__).resolve().parent.parent
    targets={}
    # Stop the supervisor first so it cannot launch another experiment.
    for pid_file,script in [('workflow.pid','scripts/complete_workflow.py'),
                            ('tier_a_v2.pid','scripts/run_experiment.py')]:
        path=root/'results/logs'/pid_file
        if not path.exists():continue
        try:
            process=psutil.Process(int(path.read_text().strip()))
            if script not in process.cmdline() or Path(process.cwd()).resolve()!=root:
                raise RuntimeError(f'Refusing unexpected process {process.pid}')
            process.suspend()
            descendants=process.children(recursive=True)
            for child in descendants:
                if child.name().lower()=='python.exe':
                    child.suspend()
                    targets[child.pid]=child
            targets[process.pid]=process
        except psutil.NoSuchProcess:
            continue
    for process in targets.values():
        try:process.terminate()
        except psutil.NoSuchProcess:pass
    gone,alive=psutil.wait_procs(list(targets.values()),timeout=10)
    if alive:raise RuntimeError(f'Processes still alive: {[p.pid for p in alive]}')
    status_path=root/'results/workflow_status.json'
    state=json.loads(status_path.read_text())
    run=root/state['tier_a_run']
    frames=[pd.read_csv(path) for path in (run/'cells').glob('*.csv')]
    if frames:
        table=pd.concat(frames,ignore_index=True)
        if table.duplicated(['cell_hash','eps_rel']).any():
            raise RuntimeError('Duplicate saved checkpoint rows')
        output=root/'results/tables/tier_a_results.csv'
        temporary=output.with_suffix('.pause.tmp')
        table.to_csv(temporary,index=False)
        os.replace(temporary,output)
    else:table=pd.DataFrame()
    state.update(status='paused',stage='paused_by_user',
                 updated_utc=datetime.now(timezone.utc).isoformat(),
                 completed_cells=len(frames),saved_rows=len(table),
                 stopped_processes=sorted(targets),
                 resume_note='Relaunch Tier A and supervisor explicitly; unfinished cells rerun.')
    status_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
    (root/'results/pause_status.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    with (root/'LOG.md').open('a',encoding='utf-8') as log:
        log.write(f'\n\n## {state["updated_utc"]} — User-requested pause\n\n'
                  f'Stopped experiment and follow-up supervisor at the user\'s explicit request. '
                  f'Preserved {len(frames)} complete model cells / {len(table)} rows and rebuilt '
                  'the aggregate from atomic checkpoints. In-flight cells will rerun on resume. '
                  'No further experiments or analyses are authorized until the user resumes.\n')
    print(json.dumps(state,indent=2))


if __name__=='__main__':main()

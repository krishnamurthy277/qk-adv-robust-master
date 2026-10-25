"""Continue an already launched Tier A run through the authorized later stages.

State is persisted after every stage. This supervisor never starts Tier B.
Run from the project root, using the same interpreter as the experiments.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tier-a-run',required=True)
    args=parser.parse_args()
    run_dir=Path(args.tier_a_run)
    state_path=Path('results/workflow_status.json')
    logs=Path('results/logs');logs.mkdir(parents=True,exist_ok=True)
    state=dict(tier_a_run=str(run_dir),completed_stages=[],status='running')
    def record(stage,**extra):
        state.update(stage=stage,updated_utc=datetime.now(timezone.utc).isoformat(),**extra)
        temporary=state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state,indent=2))
        os.replace(temporary,state_path)
        print(json.dumps(state),flush=True)
    def execute(stage,arguments):
        record(stage)
        log=logs/(stage+'.log')
        with log.open('w',encoding='utf-8') as out:
            result=subprocess.run([sys.executable,*arguments],stdout=out,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{stage} exited {result.returncode}; see {log}')
        state['completed_stages'].append(stage)
        record(stage+'_complete')
    try:
        record('waiting_for_tier_a')
        while not (run_dir/'status.json').exists():
            time.sleep(30)
        status=json.loads((run_dir/'status.json').read_text())
        if not status.get('complete'):
            raise RuntimeError(f'Tier A incomplete: {status}')
        execute('phase5_raw_summary',['scripts/analyze_results.py','--phase5-only'])
        execute('phase6_iso_grid',['scripts/run_experiment.py','--config','configs/iso_concentration.yaml'])
        execute('phase6_classical_controls',['scripts/run_classical_controls.py'])
        execute('phase6_analysis',['scripts/analyze_results.py'])
        execute('phase6_tests',['-m','pytest','tests','-q','--junitxml=results/validation_final.xml'])
        execute('phase6_smoke',['scripts/run_experiment.py','--config','configs/smoke_test.yaml'])
        # This is deliberately not labelled complete certification: fresh install
        # and visual review are distinct from successful numerical/script checks.
        record('artifacts_ready_for_review',status='needs_final_review',
               remaining=['Inspect generated figures and hypothesis reports',
                          'Check iso-concentration overlap and extend if needed',
                          'Clean dependency-install verification remains unperformed'])
    except Exception as error:
        record('failed',status='failed',error=str(error))
        raise


if __name__=='__main__':main()

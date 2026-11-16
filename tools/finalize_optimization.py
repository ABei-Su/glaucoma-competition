"""Run final checks and produce artifacts after validation-only candidate comparison."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils import save_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--node',required=True)
    args=parser.parse_args()
    deadline=time.monotonic()+10800
    status_path=ROOT/'reports/finalization_status.json'
    try:
        save_json(status_path,dict(phase='waiting_for_candidate_comparison'))
        while True:
            status=json.loads((ROOT/'reports/optimization_status.json').read_text(encoding='utf-8'))
            if status['phase']=='failed': raise RuntimeError('Candidate pipeline failed: '+status.get('error',''))
            if status['phase']=='candidates_compared': break
            if time.monotonic()>deadline: raise TimeoutError('Candidate comparison timed out')
            time.sleep(10)
        commands=[('verification',[sys.executable,'tools/verify.py']),
                  ('selection',[sys.executable,'-u','tools/select_optimized.py']),
                  ('uncertainty',[sys.executable,'-u','tools/bootstrap_comparison.py']),
                  ('frozen_test_inference',[sys.executable,'-u','predict.py','--checkpoint','output/optimized/selected_config.json','--output','output/optimized/submission.csv']),
                  ('declaration',[args.node,'tools/build_checklist.mjs','--optimized']),
                  ('packaging',[sys.executable,'-u','tools/package_optimized.py']),
                  ('packaged_runtime',[sys.executable,'-u','tools/verify_packaged_runtime.py'])]
        for phase,command in commands:
            save_json(status_path,dict(phase=phase))
            print(phase,flush=True)
            with (ROOT/'reports'/('final_'+phase+'.log')).open('w',encoding='utf-8') as stream:
                subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        save_json(status_path,dict(phase='complete',official_T4_verified=False))
        print('OPTIMIZATION ARTIFACTS COMPLETE',flush=True)
    except BaseException as error:
        save_json(status_path,dict(phase='failed',error=str(error)))
        raise


if __name__=='__main__': main()

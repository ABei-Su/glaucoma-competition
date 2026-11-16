"""Finalize v3 with strict v2 comparison; preserve prior delivered versions."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils import save_json


def main(version='v3',reference_name='fixed v2',wait_seconds=18000,provenance_relative='reports/pretrain_provenance_v3.json'):
    parser=argparse.ArgumentParser();parser.add_argument('--node',required=True);args=parser.parse_args()
    state=ROOT/'reports'/(version+'_finalization_status.json')
    deadline=time.monotonic()+wait_seconds
    created_prediction=False
    try:
        save_json(state,dict(phase='waiting_for_candidates'))
        while True:
            status_path=ROOT/'reports'/(version+'_status.json')
            status=json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {'phase':'not_started'}
            if status['phase']=='failed': raise RuntimeError(version+' training/comparison failed: '+status.get('error',''))
            if status['phase']=='not_promoted':
                save_json(state,dict(phase='not_promoted',reason=status.get('reason','Internal validation did not meet the declared gate')))
                return
            if status['phase']=='candidates_compared': break
            if time.monotonic()>deadline: raise TimeoutError(version+' comparison wait exceeded '+str(wait_seconds)+' seconds')
            time.sleep(10)
        commands=[('verification',[sys.executable,'tools/verify.py']),
                  ('selection',[sys.executable,'-u','tools/select_'+version+'.py']),
                  ('bootstrap',[sys.executable,'-u','tools/bootstrap_'+version+'.py']),
                  ('test_inference',[sys.executable,'-u','predict.py','--checkpoint','output/'+version+'/selected_config.json','--output','output/'+version+'/submission.csv']),
                  ('declaration',[args.node,'tools/build_checklist.mjs','--optimized','--output','output/'+version,'--provenance',provenance_relative]),
                  ('packaging',[sys.executable,'-u','tools/package_'+version+'.py'])]
        for phase,command in commands:
            save_json(state,dict(phase=phase));print(phase,flush=True)
            with (ROOT/'reports'/(version+'_'+phase+'.log')).open('w',encoding='utf-8') as stream:
                completed=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            if completed.returncode:
                if phase=='selection' and (ROOT/'output'/version/'selection_rejected.json').exists():
                    rejection=json.loads((ROOT/'output'/version/'selection_rejected.json').read_text(encoding='utf-8'))
                    save_json(state,dict(phase='not_promoted',reason=rejection.get('rejection_reason','Actual validation did not improve on '+reference_name)))
                    return
                raise RuntimeError(phase+' exited '+str(completed.returncode))
            if phase=='test_inference':created_prediction=True
        save_json(state,dict(phase='complete',official_T4_verified=False))
        print(version.upper()+' ARTIFACTS COMPLETE',flush=True)
    except BaseException as error:
        failure=dict(phase='failed',error=str(error))
        if created_prediction:
            output=ROOT/'output'/version
            source=output/'submission.csv'
            if source.exists():
                quarantine=output/'submission.unverified.csv'
                if quarantine.exists():quarantine=output/('submission.unverified.'+str(time.time_ns())+'.csv')
                source.rename(quarantine)
                manifest=output/'submission.manifest.json'
                if manifest.exists():manifest.rename(quarantine.with_suffix('.manifest.json'))
                failure['unverified_prediction']=str(quarantine)
        save_json(state,failure);raise


if __name__=='__main__': main()

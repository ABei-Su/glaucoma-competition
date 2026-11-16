"""Run local preparation-dependent training then frozen inference; never upload."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def status(phase, **extra):
    value=dict(phase=phase,**extra)
    (ROOT/'reports'/'pipeline_status.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(value),flush=True)


def run(command, log_name):
    with (ROOT/'reports'/log_name).open('a',encoding='utf-8') as logfile:
        result=subprocess.run([sys.executable,'-u']+command,cwd=ROOT,stdout=logfile,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Command failed; see reports/'+log_name)


def main():
    started=time.monotonic()
    try:
        status('waiting_for_data_audit')
        deadline=started+3600
        while not (ROOT/'data'/'cache224'/'READY.json').is_file():
            audit=ROOT/'reports'/'data_audit.json'
            if audit.is_file() and json.loads(audit.read_text(encoding='utf-8'))['cross_split_exact_duplicate_hashes']:
                raise RuntimeError('Train/validation exact duplicates need resolution')
            if time.monotonic()>deadline:
                raise TimeoutError('Data preparation did not finish within one hour')
            time.sleep(10)
        if not (ROOT/'reports'/'pretrain_provenance.json').is_file() or not (ROOT/'pretrain_model'/'resnet18-f37072fd.pth').is_file():
            raise RuntimeError('Pretraining provenance or weight missing')
        status('binding_audit')
        run(['tools/bind_audit.py'],'audit_binding.log')
        status('verification')
        run(['tools/verify.py'],'verification.log')
        status('training')
        run(['train.py'],'training.log')
        status('frozen_test_inference')
        run(['predict.py'],'inference.log')
        status('packaging')
        run(['tools/package_results.py'],'packaging.log')
        status('baseline_ready',elapsed_seconds=time.monotonic()-started,
               submission='output/submission.csv',materials='output/baseline_materials.zip',
               note='User upload only; official T4 timing remains unverified')
    except BaseException as error:
        status('failed',error=str(error))
        raise


if __name__=='__main__':
    main()

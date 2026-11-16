"""Exercise the actual packaged timing CLI on validation-only 100+5 images."""
import shutil
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data import image_index,load_rows
from provenance import inference_fingerprint
from utils import save_json


def main(output=None):
    output=Path(output) if output is not None else ROOT/'output/optimized'
    package=output/'materials'
    raw=ROOT/'data/raw/glaucoma_first'
    rows=load_rows(raw,'valid')
    index=image_index(raw/'valid')
    probe=Path(tempfile.mkdtemp(prefix=output.name+'_runtime_',dir=ROOT/'tmp'))
    for name,subset in [('warmup',rows[:5]),('measured',rows[5:105])]:
        directory=probe/name
        directory.mkdir()
        for row in subset:
            source=index[row['Image_name'].casefold()]
            shutil.copy2(source,directory/source.name)
    subprocess.run([sys.executable,str(package/'cost.py'),str(output/'validation_predictions.csv'),
                    str(probe/'measured'),str(probe/'warmup'),
                    '--report',str(output/'packaged_local_runtime_check.json')],cwd=package,check=True,
                   env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    report=json.loads((output/'packaged_local_runtime_check.json').read_text(encoding='utf-8'))
    report['model_and_config']=inference_fingerprint(package/'model/config.json')
    files=[package/'cost.py']+list((package/'code').glob('*.py'))
    report['code_sha256']={file.relative_to(package).as_posix():hashlib.sha256(file.read_bytes()).hexdigest() for file in files}
    save_json(output/'packaged_local_runtime_check.json',report)
    print('PACKAGED CLI VERIFIED ON LOCAL VALIDATION IMAGES; NOT OFFICIAL T4 TEST',flush=True)


if __name__=='__main__': main()

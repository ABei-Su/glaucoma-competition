"""Fixed official ImageNet-1K V2 download; preserve v1/v2 provenance."""
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils import save_json


def main():
    url='https://download.pytorch.org/models/resnet50-11ad3fa6.pth'
    target=ROOT/'pretrain_model/resnet50-11ad3fa6.pth'
    source=(ROOT/'licenses/torchvision-resnet-source.py.txt').read_text(encoding='utf-8')
    if 'resnet50-11ad3fa6.pth' not in source:
        raise ValueError('Official archived model source does not confirm this weight URL')
    valid=target.exists() and hashlib.sha256(target.read_bytes()).hexdigest().startswith('11ad3fa6')
    if not valid:
        temporary=target.with_suffix('.download')
        for attempt in range(3):
            try:
                started=time.monotonic();received=0
                with urllib.request.urlopen(url,timeout=30) as response,temporary.open('wb') as stream:
                    while True:
                        chunk=response.read(1024*1024)
                        if not chunk: break
                        stream.write(chunk);received+=len(chunk)
                        if received%(10*1024*1024)==0: print('Downloaded MB',received//(1024*1024),flush=True)
                        if time.monotonic()-started>300: raise TimeoutError('Official weight download exceeded 5 minutes')
                if not hashlib.sha256(temporary.read_bytes()).hexdigest().startswith('11ad3fa6'):
                    raise ValueError('Official weight hash prefix mismatch')
                os.replace(temporary,target)
                break
            except Exception:
                if attempt==2: raise
                time.sleep(2**attempt)
    provenance=json.loads((ROOT/'reports/pretrain_provenance.json').read_text(encoding='utf-8'))
    for item in provenance:
        if hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Existing provenance mismatch')
    item=dict(path=target.relative_to(ROOT).as_posix(),url=url,bytes=target.stat().st_size,
              sha256=hashlib.sha256(target.read_bytes()).hexdigest(),architecture='resnet50',
              pretraining_data='ImageNet-1K general natural images',variant='IMAGENET1K_V2',
              model_documentation='https://docs.pytorch.org/vision/0.19/models/generated/torchvision.models.resnet50.html')
    save_json(ROOT/'reports/pretrain_provenance_v3.json',provenance+[item])
    print(json.dumps(item),flush=True)


if __name__=='__main__': main()

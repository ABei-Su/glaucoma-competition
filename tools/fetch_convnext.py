"""Fetch a fixed public torchvision ImageNet weight, never use medical weights."""
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


def fetch(url,target,expected_prefix=None):
    if target.exists() and (expected_prefix is None or hashlib.sha256(target.read_bytes()).hexdigest().startswith(expected_prefix)):
        return
    temporary=target.with_suffix('.download')
    with urllib.request.urlopen(url,timeout=30) as response,temporary.open('wb') as stream:
        started=time.monotonic()
        while True:
            chunk=response.read(1024*1024)
            if not chunk:break
            stream.write(chunk)
            if time.monotonic()-started>600:raise TimeoutError('Fixed resource download exceeded 10 minutes')
    if expected_prefix and not hashlib.sha256(temporary.read_bytes()).hexdigest().startswith(expected_prefix):
        raise ValueError('Official weight hash prefix mismatch')
    os.replace(temporary,target)


def main():
    source_url='https://raw.githubusercontent.com/pytorch/vision/v0.19.1/torchvision/models/convnext.py'
    source=ROOT/'licenses/torchvision-convnext-source.py.txt'
    fetch(source_url,source)
    filename='convnext_tiny-983f1562.pth'
    if filename not in source.read_text(encoding='utf-8'):raise ValueError('Archived source does not name weight')
    url='https://download.pytorch.org/models/'+filename
    target=ROOT/'pretrain_model'/filename
    fetch(url,target,'983f1562')
    provenance=json.loads((ROOT/'reports/pretrain_provenance_v3.json').read_text(encoding='utf-8'))
    for item in provenance:
        if hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Existing provenance changed')
    for path,link in [(source,source_url),(target,url)]:
        provenance.append(dict(path=path.relative_to(ROOT).as_posix(),url=link,bytes=path.stat().st_size,
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest(),architecture='convnext_tiny',
                               pretraining_data='ImageNet-1K general natural images',variant='IMAGENET1K_V1',
                               model_documentation='https://docs.pytorch.org/vision/0.19/models/generated/torchvision.models.convnext_tiny.html'))
    save_json(ROOT/'reports/pretrain_provenance_v5.json',provenance)
    print(json.dumps(provenance[-1]),flush=True)


if __name__=='__main__':main()

"""Archive and verify the fixed torchvision EfficientNetV2-S ImageNet resource."""
import hashlib
import json
import sys
from pathlib import Path
from fetch_convnext import fetch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils import save_json


def main():
    source_url='https://raw.githubusercontent.com/pytorch/vision/v0.19.1/torchvision/models/efficientnet.py'
    source=ROOT/'licenses/torchvision-efficientnet-source.py.txt';fetch(source_url,source)
    filename='efficientnet_v2_s-dd5fe13b.pth'
    if filename not in source.read_text(encoding='utf-8'):raise ValueError('Archived source does not confirm weight')
    url='https://download.pytorch.org/models/'+filename;target=ROOT/'pretrain_model'/filename
    fetch(url,target,'dd5fe13b')
    provenance=json.loads((ROOT/'reports/pretrain_provenance_v5.json').read_text())
    for item in provenance:
        if hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Existing provenance changed')
    for path,link in [(source,source_url),(target,url)]:
        provenance.append(dict(path=path.relative_to(ROOT).as_posix(),url=link,bytes=path.stat().st_size,
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest(),architecture='efficientnet_v2_s',
                               pretraining_data='ImageNet-1K general natural images',variant='IMAGENET1K_V1',
                               model_documentation='https://docs.pytorch.org/vision/0.19/models/generated/torchvision.models.efficientnet_v2_s.html'))
    save_json(ROOT/'reports/pretrain_provenance_v7.json',provenance)
    print(json.dumps(provenance[-1]),flush=True)


if __name__=='__main__':main()

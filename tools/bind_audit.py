"""Bind a completed fresh raw-image audit to labels and every derived cache file."""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils import save_json
from data import image_index


def bind():
    raw=ROOT/'data'/'raw'/'glaucoma_first'
    cache=ROOT/'data'/'cache224'
    manifest_path=ROOT/'reports'/'train_valid_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    report=json.loads((ROOT/'reports'/'data_audit.json').read_text(encoding='utf-8'))
    if report['cross_split_exact_duplicate_hashes']:
        raise ValueError('Leakage audit not clean')
    paths={split:image_index(raw/split) for split in ('train','valid')}
    for item in manifest:
        cached=cache/item['split']/(item['name']+'.png')
        item['cache_sha256']=hashlib.sha256(cached.read_bytes()).hexdigest()
        item['disk_name']=paths[item['split']][item['name'].casefold()].name
    save_json(manifest_path,manifest)
    report.update(source_root=str(raw.resolve()),size=224,preprocess_version=1,manifest_path=str(manifest_path.resolve()),
                  csv_sha256={split:hashlib.sha256((raw/(split+'.csv')).read_bytes()).hexdigest() for split in ('train','valid')},
                  manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    save_json(cache/'READY.json',report)
    save_json(ROOT/'reports'/'data_audit.json',report)
    print('AUDIT BOUND',report['manifest_sha256'],flush=True)


if __name__=='__main__':
    bind()

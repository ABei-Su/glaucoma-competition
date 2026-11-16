"""Create a new resolution cache without changing the audited baseline cache."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data import preprocess_image
from utils import save_json
from cache_paths import cache_path


def check_space(directory, minimum=2_000_000_000):
    if shutil.disk_usage(directory).free < minimum:
        raise OSError('Less than 2GB free; cache preparation stopped safely')


def cached_pixels_match(target, processed):
    if not target.exists():
        return False
    try:
        with Image.open(target) as existing:
            return existing.mode=='RGB' and existing.size==processed.size and existing.tobytes()==processed.tobytes()
    except OSError:
        # Only derived cache files are regenerated; source failures still raise.
        return False


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--size',type=int,default=384)
    parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args()
    if args.size not in (320,384,448):
        raise ValueError('Use a declared resolution')
    base=ROOT/'data'/'cache224'/'READY.json'
    report=json.loads(base.read_text(encoding='utf-8'))
    original=Path(report['manifest_path'])
    content=original.read_bytes()
    if hashlib.sha256(content).hexdigest()!=report['manifest_sha256']:
        raise ValueError('Original audit fingerprint mismatch')
    manifest=json.loads(content)
    raw=Path(report['source_root'])
    cache=cache_path(args.size)
    if (cache/'READY.json').exists():
        raise FileExistsError('Resolution already prepared; preserve it')
    for split in ('train','valid'):
        (cache/split).mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    def process(item):
        check_space(cache)
        source=raw/item['split']/item['disk_name']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Source changed after baseline audit')
        with Image.open(source) as image:
            processed=preprocess_image(image,args.size)
        target=cache/item['split']/(item['name']+'.png')
        if cached_pixels_match(target,processed):
            return dict(item,cache_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        temporary=target.with_suffix('.tmp')
        processed.save(temporary,format='PNG',compress_level=6)
        os.replace(temporary,target)
        return dict(item,cache_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n,item in enumerate(pool.map(process,manifest),1):
            results.append(item)
            if n%2000==0:
                print(json.dumps({'size':args.size,'cached':n,'total':len(manifest),'seconds':round(time.monotonic()-started)}),flush=True)
    manifest_path=cache/'manifest.json'
    save_json(manifest_path,results)
    report.update(size=args.size,manifest_path=str(manifest_path),
                  manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  preprocess=f'RGB, EXIF transpose, per-image border >12, aspect pad bilinear {args.size}; lossless PNG',
                  seconds=time.monotonic()-started)
    save_json(cache/'READY.json',report)
    print('RESOLUTION CACHE READY',args.size,flush=True)


if __name__=='__main__':
    main()

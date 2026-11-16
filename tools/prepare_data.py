"""Audit and cache train/valid only. Does not open test images or labels."""
import collections
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import load_rows, preprocess_image, image_index
from utils import save_json


def main():
    if not (ROOT/'reports'/'extraction.json').is_file():
        raise RuntimeError('Extraction must complete first')
    raw = ROOT/'data'/'raw'/'glaucoma_first'
    cache = ROOT/'data'/'cache224'
    records=[]
    paths={split:image_index(raw/split) for split in ('train','valid')}
    for split, expected in [('train',51396),('valid',5849)]:
        rows=load_rows(raw,split)
        if len(rows)!=expected:
            raise ValueError('Unexpected split count')
        for row in rows:
            path=paths[split][row['Image_name'].casefold()]
            if not path.is_file():
                raise FileNotFoundError(path)
            records.append((split,row))
        (cache/split).mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    def process(record):
        split,row=record
        source=paths[split][row['Image_name'].casefold()]
        content=source.read_bytes()
        digest=hashlib.sha256(content).hexdigest()
        with Image.open(source) as image:
            size=image.size
            processed=preprocess_image(image,224)
        target=cache/split/(row['Image_name']+'.png')
        temporary=target.with_suffix('.tmp')
        processed.save(temporary,format='PNG',compress_level=1)
        os.replace(temporary,target)
        return {'split':split,'name':row['Image_name'],'label':int(row['True_Label']),
                'sha256':digest,'width':size[0],'height':size[1]}
    results=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for n,result in enumerate(pool.map(process,records),1):
            results.append(result)
            if n%2000==0:
                print(json.dumps({'cached':n,'total':len(records),'seconds':round(time.monotonic()-started)}),flush=True)
    trainhash={r['sha256'] for r in results if r['split']=='train'}
    validhash={r['sha256'] for r in results if r['split']=='valid'}
    overlap=trainhash & validhash
    report={'splits':{},'cross_split_exact_duplicate_hashes':len(overlap),
            'patient_ids_available':False,'preprocess':'RGB, EXIF transpose, per-image border >12, aspect pad bilinear 224; lossless PNG',
            'test_images_read':False,'seconds':time.monotonic()-started}
    for split in ['train','valid']:
        subset=[r for r in results if r['split']==split]
        report['splits'][split]={'rows':len(subset),'labels':dict(collections.Counter(r['label'] for r in subset)),
                                'unique_file_hashes':len({r['sha256'] for r in subset}),
                                'width_range':[min(r['width'] for r in subset),max(r['width'] for r in subset)],
                                'height_range':[min(r['height'] for r in subset),max(r['height'] for r in subset)]}
    save_json(ROOT/'reports'/'data_audit.json',report)
    save_json(ROOT/'reports'/'train_valid_manifest.json',results)
    if overlap:
        raise RuntimeError('Exact train-valid duplicates found; resolve leakage before training')
    # Bind hashes after all PNGs and the raw manifest are fully written.
    from tools.bind_audit import bind
    bind()
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()

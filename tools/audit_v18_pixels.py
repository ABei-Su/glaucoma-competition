"""One-shot all-validation pixel equivalence audit; CPU only, no outcome metrics."""
import concurrent.futures
import io
import json
from pathlib import Path
from PIL import Image
from data import crop_fundus, pad_fundus, image_index, load_rows
from v18_fast_preprocess import crop_fundus_exact
from provenance import digest
from utils import save_json
import hashlib

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/v18/implementation_001'


def main():
    OUT.mkdir(exist_ok=False)
    raw=ROOT/'data/raw/glaucoma_first'
    names=[r['Image_name'] for r in load_rows(raw,'valid')]
    if len(names)!=5849:raise ValueError('Unexpected valid size')
    index=image_index(raw/'valid')
    bound=json.loads((ROOT/'output/v18/finalization/frozen_trial_inputs.json').read_text('utf-8'))['files']
    expected={str(Path(x['path']).resolve()).casefold():x['sha256'] for x in bound}
    def check(name):
        p=index[name.casefold()];blob=p.read_bytes()
        if hashlib.sha256(blob).hexdigest()!=expected[str(p.resolve()).casefold()]:raise ValueError('Raw image changed')
        with Image.open(io.BytesIO(blob)) as im:old=pad_fundus(crop_fundus(im),384)
        with Image.open(io.BytesIO(blob)) as im:new=pad_fundus(crop_fundus_exact(im),384)
        if old.mode!=new.mode or old.size!=new.size or old.tobytes()!=new.tobytes():
            raise ValueError('Non-equivalent pixels: '+name)
        return name
    count=0
    report=dict(passed=False,scope='all5849officialvalid,no test pixels,no model inference',checked=0)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for _ in pool.map(check,names):
                count+=1
                if count%500==0:print('identical_pixels',count,flush=True)
        report.update(passed=True,checked=count,files=[dict(path=str(ROOT/p),sha256=digest(ROOT/p))
            for p in ('data.py','v18_fast_preprocess.py','v18_fast_predict.py','tools/audit_v18_pixels.py','tests/test_v18_fast_preprocess.py')])
    except BaseException as exc:
        report.update(checked=count,error=repr(exc))
        raise
    finally:save_json(OUT/'pixel_equivalence.json',report)
    print('PASS',count,flush=True)


if __name__=='__main__':main()

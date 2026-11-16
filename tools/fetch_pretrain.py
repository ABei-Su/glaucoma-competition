"""Download only fixed public model provenance resources with bounded retries."""
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = {
 'pretrain_model/resnet18-f37072fd.pth': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
 'licenses/torchvision-BSD.txt': 'https://raw.githubusercontent.com/pytorch/vision/v0.19.1/LICENSE',
 'licenses/torchvision-resnet-source.py.txt': 'https://raw.githubusercontent.com/pytorch/vision/v0.19.1/torchvision/models/resnet.py',
 'licenses/torchvision-training-recipe.md': 'https://raw.githubusercontent.com/pytorch/vision/v0.19.1/references/classification/README.md',
}


def main():
    manifest = []
    for relative, url in RESOURCES.items():
        target = ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            content = target.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if target.suffix != '.pth' or digest == 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec':
                manifest.append({'path':relative,'url':url,'bytes':len(content),'sha256':digest})
                print('Verified existing',relative,flush=True)
                continue
        for attempt in range(3):
            try:
                started = time.monotonic()
                print('Fetching', relative, 'attempt', attempt+1, flush=True)
                chunks = []
                received = 0
                with urllib.request.urlopen(url, timeout=30) as response:
                    while True:
                        chunk = response.read(262144)
                        if not chunk:
                            break
                        chunks.append(chunk)
                        received += len(chunk)
                        if received % (1024*1024) == 0:
                            print('received_MB', received//(1024*1024), flush=True)
                        if time.monotonic()-started>240:
                            raise TimeoutError('Download exceeded 4-minute deadline')
                content = b''.join(chunks)
                digest = hashlib.sha256(content).hexdigest()
                if target.suffix == '.pth' and not digest.startswith('f37072fd'):
                    raise ValueError('Official weight hash prefix mismatch')
                temporary = target.with_suffix(target.suffix+'.tmp')
                temporary.write_bytes(content)
                os.replace(temporary,target)
                manifest.append({'path':relative,'url':url,'bytes':len(content),'sha256':digest})
                print(relative, len(content), digest, flush=True)
                break
            except Exception:
                if attempt==2:
                    raise
                time.sleep(2**attempt)
    (ROOT/'reports'/'pretrain_provenance.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()

"""Frozen schema1 prediction with explicit v12 backend and code identity."""
import argparse
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data import image_index
from predict import Predictor,test_names
from provenance import digest,inference_fingerprint
from utils import save_json,write_submission
from v12_protocol import inference_identity


def configure_backend():
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--data',default='data/raw/glaucoma_first')
    args=parser.parse_args();configure_backend();output=Path(args.output)
    if output.exists() or output.with_suffix('.manifest.json').exists():raise FileExistsError('Preserve v12 prediction')
    identity=inference_identity(args.checkpoint);predictor=Predictor(args.checkpoint)
    names=test_names(args.data);paths=image_index(Path(args.data)/'test');values=[]
    for index,name in enumerate(names,1):
        values.append(predictor.predict_one(paths[name.casefold()]))
        if index%500==0:print('predicted',index,'/',len(names),flush=True)
    if identity!=inference_identity(args.checkpoint):raise ValueError('V12 inference identity changed')
    write_submission(output,names,values)
    save_json(output.with_suffix('.manifest.json'),dict(files=inference_fingerprint(args.checkpoint),
        runtime_files=identity,submission_sha256=digest(output),precision='fp16',device='cuda',batch=1,
        cudnn_deterministic=True,cudnn_benchmark=False))
    print('V12 PREDICTION READY',str(output),flush=True)


if __name__=='__main__':main()

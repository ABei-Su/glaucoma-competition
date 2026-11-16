"""Frozen per-image inference shared by submission generation and timing."""
import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image

from data import pad_fundus, tensor_transform, image_index
from v18_fast_preprocess import crop_fundus_exact as crop_fundus
from model import build_model,probabilities_from_logits,fuse_inference_batchnorm
from utils import write_submission,save_json
from provenance import digest,inference_fingerprint


class Predictor:
    def __init__(self, checkpoint, device='cuda', precision='fp16'):
        if precision not in ('fp16','fp32'):
            raise ValueError('Unsupported precision')
        self.device=torch.device(device)
        if self.device.type != 'cuda' and precision=='fp16':
            raise ValueError('Use fp32 for CPU')
        self.precision=precision
        checkpoint=Path(checkpoint)
        if checkpoint.suffix.lower()=='.json':
            spec=json.loads(checkpoint.read_text(encoding='utf-8'))
            entries=spec.get('models',[])
            if spec.get('schema_version')!=1 or not 1<=len(entries)<=2:
                raise ValueError('Inference bundle must contain one or two models')
            specs=[]
            for entry in entries:
                if not isinstance(entry.get('tta',False),bool):
                    raise ValueError('TTA must be boolean')
                if not isinstance(entry.get('fuse_conv_bn',False),bool):
                    raise ValueError('fuse_conv_bn must be boolean')
                specs.append((checkpoint.parent/entry['checkpoint'],entry.get('tta',False),entry.get('fuse_conv_bn',False)))
        else:
            specs=[(checkpoint,False,False)]
        self.members=[]
        for path,tta,fuse in specs:
            saved=torch.load(path,map_location='cpu',weights_only=True)
            model=build_model(architecture=saved.get('architecture','resnet18'))
            model.load_state_dict(saved['model'])
            if fuse:
                if saved.get('architecture')!='efficientnet_v2_s':
                    raise ValueError('Conv/BN fusion is supported only for efficientnet_v2_s')
                model,count=fuse_inference_batchnorm(model.eval())
                if not count:raise ValueError('No inference Conv/BN pairs found')
            model.inference_memory_format=saved.get('memory_format','contiguous')
            if model.inference_memory_format not in ('contiguous','channels_last'):
                raise ValueError('Unsupported checkpoint memory format')
            layout=torch.channels_last if model.inference_memory_format=='channels_last' else torch.contiguous_format
            model.to(self.device,memory_format=layout).eval().requires_grad_(False)
            self.members.append((model,saved['size'],tta))
        # Keep the single-model API for prior code/tests.
        self.model,self.size,_=self.members[0]
        self.transform=tensor_transform(False)

    @torch.inference_mode()
    def predict_one(self,path):
        with Image.open(path) as image:
            cropped=crop_fundus(image)
        tensors={}
        probabilities=[]
        for model,size,tta in self.members:
            key=(size,model.inference_memory_format)
            if key not in tensors:
                layout=torch.channels_last if model.inference_memory_format=='channels_last' else torch.contiguous_format
                tensors[key]=self.transform(pad_fundus(cropped,size)).unsqueeze(0).to(self.device,memory_format=layout)
            tensor=tensors[key]
            with torch.autocast(device_type=self.device.type,enabled=self.precision=='fp16'):
                probability=probabilities_from_logits(model(tensor).flatten())
                if tta:
                    probability=(probability+probabilities_from_logits(model(tensor.flip(-1)).flatten()))/2
            probabilities.append(probability.item())
        return sum(probabilities)/len(probabilities)


def test_names(root):
    with (Path(root)/'test.csv').open(encoding='utf-8-sig',newline='') as file:
        rows=list(csv.DictReader(file))
    names=[r['Image_name'] for r in rows]
    if len(names)!=5855 or len(set(names))!=5855:
        raise ValueError('Expected exactly 5855 unique test filenames')
    index = image_index(Path(root)/'test')
    for name in names:
        if Path(name).name!=name or '/' in name or '\\' in name:
            raise ValueError('Unsafe test filename')
        if name.casefold() not in index:
            raise FileNotFoundError(name)
    return names


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default='data/raw/glaucoma_first')
    parser.add_argument('--checkpoint',default='runs/baseline_r18_224/best.pt')
    parser.add_argument('--output',default='output/submission.csv')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--precision',default='fp16')
    args=parser.parse_args()
    torch.set_num_threads(4)
    fingerprint=inference_fingerprint(args.checkpoint)
    predictor=Predictor(args.checkpoint,args.device,args.precision)
    names=test_names(args.data)
    paths=image_index(Path(args.data)/'test')
    probabilities=[]
    for i,name in enumerate(names,1):
        probabilities.append(predictor.predict_one(paths[name.casefold()]))
        if i%500==0:
            print('predicted',i,'/',len(names),flush=True)
    write_submission(args.output,names,probabilities)
    if inference_fingerprint(args.checkpoint)!=fingerprint:
        raise RuntimeError('Weights/config changed during inference')
    save_json(Path(args.output).with_suffix('.manifest.json'),dict(files=fingerprint,
              submission_sha256=digest(args.output),precision=args.precision,device=args.device,batch=1))
    print('Saved',args.output,flush=True)


if __name__=='__main__':
    main()

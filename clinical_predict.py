"""Schema2 image/Age inference; per-image supplied metadata, never fitted here."""
import argparse
import csv
import json
from pathlib import Path
import torch
from PIL import Image
from clinical import build_age_model,encode_age,validate_age_stats
from data import crop_fundus,pad_fundus,tensor_transform,image_index
from model import build_model,probabilities_from_logits
from predict import test_names
from provenance import digest
from utils import save_json,write_submission


def write_age_metadata(source,target):
    source,target=Path(source),Path(target)
    if target.exists():raise FileExistsError('Preserve existing metadata')
    with source.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream)
        if not {'Image_name','Age'}.issubset(reader.fieldnames or []):raise ValueError('Missing source Age fields')
        rows=[dict(Image_name=row['Image_name'],Age=row['Age']) for row in reader]
    target.parent.mkdir(exist_ok=True,parents=True)
    with target.open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['Image_name','Age']);writer.writeheader();writer.writerows(rows)
    AgeLookup(target)


class AgeLookup:
    def __init__(self,path):
        self.path=Path(path);self.values={}
        with self.path.open(encoding='utf-8-sig',newline='') as stream:
            reader=csv.DictReader(stream)
            if reader.fieldnames!=['Image_name','Age']:raise ValueError('Inference metadata must contain exactly Image_name,Age')
            for row in reader:
                name=row['Image_name']
                if not name or Path(name).name!=name or '/' in name or '\\' in name or name.casefold() in self.values:
                    raise ValueError('Unsafe/duplicate case-insensitive metadata filename')
                self.values[name.casefold()]=row['Age']

    def features(self,name,stats):return encode_age(self.values.get(Path(name).name.casefold()),stats)


def fingerprint(config):
    config=Path(config).resolve();spec=json.loads(config.read_text(encoding='utf-8'))
    paths=[config,(config.parent/spec['age_metadata']).resolve()]
    paths.extend((config.parent/item['checkpoint']).resolve() for item in spec['models'])
    paths.extend(Path(__file__).resolve().parent/name for name in ('clinical_predict.py','clinical.py','model.py','data.py','utils.py','provenance.py','predict.py'))
    return [dict(path=str(path),sha256=digest(path)) for path in dict.fromkeys(paths)]


class Predictor:
    def __init__(self,checkpoint,device='cuda',precision='fp16'):
        if precision not in ('fp16','fp32') or (not device.startswith('cuda') and precision=='fp16'):
            raise ValueError('Unsupported clinical inference precision/device')
        self.device=torch.device(device);self.precision=precision
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        path=Path(checkpoint);spec=json.loads(path.read_text(encoding='utf-8'))
        if spec.get('schema_version')!=2 or not spec.get('age_metadata') or not 1<=len(spec.get('models',[]))<=2:
            raise ValueError('Schema2 bundle requires explicit immutable Age metadata and one/two members')
        if digest(path.parent/spec['age_metadata'])!=spec.get('age_metadata_sha256'):
            raise ValueError('Age metadata frozen hash mismatch')
        self.lookup=AgeLookup(path.parent/spec['age_metadata'])
        self.members=[]
        for entry in spec['models']:
            if type(entry.get('tta',False)) is not bool:raise ValueError('TTA must be boolean')
            if entry.get('fuse_conv_bn',False):raise ValueError('Clinical bundle does not support BN folding')
            if digest(path.parent/entry['checkpoint'])!=entry.get('sha256'):
                raise ValueError('Clinical checkpoint frozen hash mismatch')
            saved=torch.load(path.parent/entry['checkpoint'],map_location='cpu',weights_only=True)
            architecture=saved.get('architecture','resnet18');stats=None
            if architecture=='convnext_tiny_age':
                if saved.get('schema_version')!=2:raise ValueError('Age checkpoint schema mismatch')
                stats=saved['age_stats'];validate_age_stats(stats);model=build_age_model()
            else:model=build_model(architecture=architecture)
            model.load_state_dict(saved['model'])
            layout=saved.get('memory_format','contiguous')
            if layout not in ('contiguous','channels_last'):raise ValueError('Unsupported memory layout')
            memory=torch.contiguous_format if layout=='contiguous' else torch.channels_last
            model.to(self.device,memory_format=memory).eval().requires_grad_(False)
            self.members.append((model,saved['size'],entry.get('tta',False),stats,layout))
        if not any(item[3] is not None for item in self.members):raise ValueError('Schema2 bundle must contain its Age model')
        self.transform=tensor_transform(False)

    @torch.inference_mode()
    def predict_one(self,path):
        path=Path(path)
        with Image.open(path) as image:cropped=crop_fundus(image)
        tensors={};values=[]
        for model,size,tta,stats,layout in self.members:
            key=(size,layout)
            if key not in tensors:
                memory=torch.contiguous_format if layout=='contiguous' else torch.channels_last
                tensors[key]=self.transform(pad_fundus(cropped,size)).unsqueeze(0).to(self.device,memory_format=memory)
            images=tensors[key]
            ages=self.lookup.features(path.name,stats).unsqueeze(0).to(self.device) if stats is not None else None
            with torch.autocast(device_type=self.device.type,enabled=self.precision=='fp16'):
                logits=model(images,ages) if stats is not None else model(images).flatten()
                p=probabilities_from_logits(logits)
                if tta:
                    flipped=model(images.flip(-1),ages) if stats is not None else model(images.flip(-1)).flatten()
                    p=(p+probabilities_from_logits(flipped))/2
            values.append(p.item())
        return sum(values)/len(values)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--data',default='data/raw/glaucoma_first');parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--precision',default='fp16')
    args=parser.parse_args();torch.set_num_threads(4)
    output=Path(args.output)
    if output.exists() or output.with_suffix('.manifest.json').exists():raise FileExistsError('Preserve existing clinical prediction')
    identity=fingerprint(args.checkpoint);predictor=Predictor(args.checkpoint,args.device,args.precision)
    names=test_names(args.data);paths=image_index(Path(args.data)/'test');probabilities=[]
    for index,name in enumerate(names,1):
        probabilities.append(predictor.predict_one(paths[name.casefold()]))
        if index%500==0:print('predicted',index,'/',len(names),flush=True)
    if identity!=fingerprint(args.checkpoint):raise ValueError('Clinical model/config/metadata/code changed during prediction')
    write_submission(output,names,probabilities)
    save_json(output.with_suffix('.manifest.json'),dict(files=identity,submission_sha256=digest(output),
        precision=args.precision,device=args.device,batch=1,age_statistics_source='checkpoint training partition only',
        cudnn_deterministic=True,cudnn_benchmark=False))
    print('Saved',str(output),flush=True)


if __name__=='__main__':main()

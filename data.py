"""Official training/validation boundary and shared deterministic preprocessing."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def audit_inputs(root, cache, size):
    root, cache = Path(root).resolve(), Path(cache).resolve()
    report=json.loads((cache/'READY.json').read_text(encoding='utf-8'))
    if report.get('source_root') != str(root) or report.get('size') != size or report.get('preprocess_version') != 1:
        raise ValueError('Audit does not match source root, size or preprocessing version')
    if report.get('cross_split_exact_duplicate_hashes') != 0 or not report.get('manifest_sha256'):
        raise ValueError('Audit has overlap or missing manifest fingerprint')
    for split in ('train','valid'):
        digest=hashlib.sha256((root/(split+'.csv')).read_bytes()).hexdigest()
        if digest != report.get('csv_sha256',{}).get(split):
            raise ValueError('Label file changed since audit: '+split)
    manifest_path=Path(report['manifest_path'])
    content=manifest_path.read_bytes()
    if hashlib.sha256(content).hexdigest()!=report['manifest_sha256']:
        raise ValueError('Audit manifest was modified')
    for item in json.loads(content):
        source=root/item['split']/item.get('disk_name',item['name'])
        cached=cache/item['split']/(item['name']+'.png')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Raw image changed after audit: '+item['name'])
        if hashlib.sha256(cached.read_bytes()).hexdigest()!=item['cache_sha256']:
            raise ValueError('Cached image changed after audit: '+item['name'])
    return report


def image_index(directory):
    result = {}
    for path in Path(directory).iterdir():
        if path.is_file() and path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            key = path.name.casefold()
            if key in result:
                raise ValueError('Ambiguous case-insensitive filename: ' + path.name)
            result[key] = path
    return result


def load_rows(root, split):
    if split not in ('train', 'valid'):
        raise ValueError('Training loader may only open train or valid')
    with (Path(root) / (split + '.csv')).open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    if not rows or not {'Image_name', 'True_Label'}.issubset(rows[0]):
        raise ValueError('Missing image/label columns')
    names = [r['Image_name'] for r in rows]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate filename')
    for row in rows:
        name = row['Image_name']
        if Path(name).name != name or '/' in name or '\\' in name or row['True_Label'] not in ('0', '1'):
            raise ValueError('Invalid image name or binary label')
    return rows


def crop_fundus(image):
    # Per-image black-border crop, then aspect-preserving pad: never fit on test data.
    if image.getexif().get(274) in (2,3,4,5,6,7,8):
        image = ImageOps.exif_transpose(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    thumb = image.copy()
    thumb.thumbnail((256, 256))
    mask = np.asarray(thumb).max(axis=2) > 12
    ys, xs = np.where(mask)
    if len(xs):
        sx, sy = image.width / thumb.width, image.height / thumb.height
        box = (max(0, int(xs.min()*sx)-2), max(0, int(ys.min()*sy)-2),
               min(image.width, int((xs.max()+1)*sx)+2), min(image.height, int((ys.max()+1)*sy)+2))
        image = image.crop(box)
    else:
        image = image.copy()
    return image


def pad_fundus(image, size):
    return ImageOps.pad(image, (size, size), method=Image.Resampling.BILINEAR, color=(0,0,0))


def preprocess_image(image, size):
    return pad_fundus(crop_fundus(image),size)


def tensor_transform(training=False,augmentation='mild'):
    if augmentation not in ('mild','color_robust'):
        raise ValueError('Unknown training augmentation profile')
    ops = []
    if training:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),transforms.RandomRotation(15)]
        if augmentation=='mild':ops += [transforms.ColorJitter(.12,.12,.10,.02)]
        else:
            ops += [transforms.ColorJitter(.30,.30,.25,.04),transforms.RandomGrayscale(p=.10),
                    transforms.RandomApply([transforms.GaussianBlur(3,sigma=(.1,.8))],p=.10)]
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class FundusDataset(Dataset):
    def __init__(self, root, split, size, cache=None, training=False,augmentation='mild'):
        self.root, self.split, self.size = Path(root), split, size
        self.rows = load_rows(root, split)
        self.paths = image_index(self.root / split)
        self.cache = Path(cache) if cache else None
        self.transform = tensor_transform(training,augmentation)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.paths[row['Image_name'].casefold()]
        cached = self.cache / self.split / (row['Image_name'] + '.png') if self.cache else None
        if cached is not None and cached.is_file():
            with Image.open(cached) as image:
                image = image.convert('RGB')
                if image.size != (self.size, self.size):
                    raise ValueError('Cache size mismatch')
        else:
            with Image.open(path) as image:
                image = preprocess_image(image, self.size)
        return self.transform(image), torch.tensor(float(row['True_Label'])), row['Image_name']

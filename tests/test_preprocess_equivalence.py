import unittest
import numpy as np
from PIL import Image,ImageOps
from data import preprocess_image


def legacy(image,size):
    image=ImageOps.exif_transpose(image).convert('RGB')
    thumb=image.copy();thumb.thumbnail((256,256))
    ys,xs=np.where(np.asarray(thumb).max(axis=2)>12)
    if len(xs):
        sx,sy=image.width/thumb.width,image.height/thumb.height
        box=(max(0,int(xs.min()*sx)-2),max(0,int(ys.min()*sy)-2),min(image.width,int((xs.max()+1)*sx)+2),min(image.height,int((ys.max()+1)*sy)+2))
        image=image.crop(box)
    return ImageOps.pad(image,(size,size),method=Image.Resampling.BILINEAR,color=(0,0,0))


class PreprocessEquivalenceTests(unittest.TestCase):
    def test_exact_pixels_across_modes_and_exif(self):
        rng=np.random.default_rng(123)
        pixels=rng.integers(0,256,(380,620,3),dtype=np.uint8)
        pixels[:40]=0; pixels[:,:70]=0
        for mode in ['RGB','L','RGBA']:
            for orientation in [1,2,3,4,5,6,7,8]:
                image=Image.fromarray(pixels).convert(mode)
                image.getexif()[274]=orientation
                for size in [224,384]:
                    self.assertEqual(preprocess_image(image,size).tobytes(),legacy(image,size).tobytes())
        image=Image.new('RGB',(600,400))
        self.assertEqual(preprocess_image(image,224).tobytes(),legacy(image,224).tobytes())

"""Pixel-equivalent v18-only thumbnail preparation without a full-size copy.

Matches the installed Pillow thumbnail((256,256), BICUBIC, reducing_gap=2)
on data.py's already-decoded RGB copy. Does not use JPEG draft decoding.
Original training/inference modules and old artifact bindings are unchanged.
"""
import math
import numpy as np
from PIL import Image, ImageOps


def crop_fundus_exact(image):
    if image.getexif().get(274) in (2,3,4,5,6,7,8):
        image=ImageOps.exif_transpose(image)
    if image.mode!='RGB':
        image=image.convert('RGB')
    image.load()
    x=y=256
    if x>=image.width and y>=image.height:
        thumb=image
    else:
        aspect=image.width/image.height
        def nearest_aspect(number,key):
            return max(min(math.floor(number),math.ceil(number),key=key),1)
        if x/y>=aspect:
            x=nearest_aspect(y*aspect,lambda n:abs(aspect-n/y))
        else:
            y=nearest_aspect(x/aspect,lambda n:0 if n==0 else abs(aspect-x/n))
        thumb=image.resize((x,y),Image.Resampling.BICUBIC,reducing_gap=2.0)
    mask=np.asarray(thumb).max(axis=2)>12
    ys,xs=np.where(mask)
    if len(xs):
        sx,sy=image.width/thumb.width,image.height/thumb.height
        box=(max(0,int(xs.min()*sx)-2),max(0,int(ys.min()*sy)-2),
             min(image.width,int((xs.max()+1)*sx)+2),min(image.height,int((ys.max()+1)*sy)+2))
        return image.crop(box)
    return image.copy()

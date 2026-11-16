import unittest
import numpy as np
from PIL import Image
from data import crop_fundus, pad_fundus
from v18_fast_preprocess import crop_fundus_exact


class ExactCropTests(unittest.TestCase):
    def test_same_pixels_all_modes_aspects_and_orientation(self):
        rng=np.random.default_rng(18)
        for width,height in [(1,1),(255,256),(257,256),(513,1025),(2049,1537),(384,384)]:
            a=rng.integers(0,256,(height,width,3),dtype=np.uint8)
            for mode in ('RGB','L','RGBA'):
                for orientation in (1,3,6,8):
                    im=Image.fromarray(a).convert(mode)
                    im.getexif()[274]=orientation
                    before=im.tobytes()
                    old=crop_fundus(im)
                    new=crop_fundus_exact(im)
                    self.assertEqual(old.size,new.size)
                    self.assertEqual(old.tobytes(),new.tobytes())
                    self.assertEqual(pad_fundus(old,384).tobytes(),pad_fundus(new,384).tobytes())
                    self.assertEqual(before,im.tobytes())

    def test_empty_and_border_masks(self):
        for value in (0,12,13,255):
            a=np.zeros((1001,1499,3),dtype=np.uint8)
            a[100:900,200:1300]=value
            im=Image.fromarray(a)
            old,new=crop_fundus(im),crop_fundus_exact(im)
            self.assertEqual(old.size,new.size)
            self.assertEqual(old.tobytes(),new.tobytes())

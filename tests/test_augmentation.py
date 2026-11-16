import unittest
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from data import tensor_transform,MEAN,STD


class AugmentationTests(unittest.TestCase):
    def image(self):
        return Image.fromarray(np.arange(64*64*3,dtype=np.uint8).reshape(64,64,3))

    def test_evaluation_is_identical_under_both_training_profiles(self):
        image=self.image()
        a=tensor_transform(False)(image)
        b=tensor_transform(False,augmentation='color_robust')(image)
        self.assertTrue(torch.equal(a,b))

    def test_default_mild_keeps_legacy_seeded_pixels_and_robust_is_reproducible(self):
        image=self.image()
        legacy=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),transforms.ColorJitter(.12,.12,.10,.02),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
        torch.manual_seed(72);expected=legacy(image)
        torch.manual_seed(72);actual=tensor_transform(True)(image)
        self.assertTrue(torch.equal(expected,actual))
        torch.manual_seed(72);robust=tensor_transform(True,augmentation='color_robust')(image)
        torch.manual_seed(72);again=tensor_transform(True,augmentation='color_robust')(image)
        self.assertTrue(torch.equal(robust,again));self.assertTrue(torch.isfinite(robust).all())
        self.assertFalse(torch.equal(actual,robust))
        with self.assertRaises(ValueError):tensor_transform(True,augmentation='unknown')


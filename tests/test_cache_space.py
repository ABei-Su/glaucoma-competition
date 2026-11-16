import unittest
import tempfile
from pathlib import Path
from PIL import Image
from unittest.mock import patch
from types import SimpleNamespace
from tools.prepare_resolution import check_space,cached_pixels_match


class CacheSpaceTests(unittest.TestCase):
    def test_resume_checks_pixels_and_rejects_partial_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'image.png'
            image=Image.new('RGB',(32,32),(1,2,3))
            self.assertFalse(cached_pixels_match(path,image))
            image.save(path,compress_level=1)
            self.assertTrue(cached_pixels_match(path,image))
            self.assertFalse(cached_pixels_match(path,Image.new('RGB',(32,32),(1,2,4))))
            path.write_bytes(b'partial file')
            self.assertFalse(cached_pixels_match(path,image))

    def test_low_space_stops_before_write(self):
        with patch('tools.prepare_resolution.shutil.disk_usage', return_value=SimpleNamespace(free=199)):
            with self.assertRaises(OSError):
                check_space('.', 200)

    def test_sufficient_space_allowed(self):
        with patch('tools.prepare_resolution.shutil.disk_usage', return_value=SimpleNamespace(free=200)):
            check_space('.', 200)

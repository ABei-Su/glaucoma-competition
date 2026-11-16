import unittest
from tools.package_optimized import validate_rows


class PackageTests(unittest.TestCase):
    def test_valid_submission(self):
        validate_rows([dict(Image_name='a',Probability='0.5',Pred_Label='1')],['a'])

    def test_reject_invalid_rows(self):
        for probability,label,name in [('nan','0','a'),('1.1','1','a'),('0.5','0','a'),('0.2','0','b')]:
            with self.assertRaises(ValueError):
                validate_rows([dict(Image_name=name,Probability=probability,Pred_Label=label)],['a'])

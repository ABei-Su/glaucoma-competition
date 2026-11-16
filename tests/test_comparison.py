import unittest
import numpy as np
import torch
from tools.compare_candidates import combine_tta


class ComparisonTests(unittest.TestCase):
    def test_comparison_tta_matches_inference_float32_rounding(self):
        a=torch.tensor([.00012345,.1234567,.9988776],dtype=torch.float32)
        b=torch.tensor([.9934567,.2345678,.0007777],dtype=torch.float32)
        expected=((a+b)/2).double().numpy()
        np.testing.assert_array_equal(combine_tta(a.tolist(),b.tolist(),True),expected)
        np.testing.assert_array_equal(combine_tta(a.tolist(),b.tolist(),False),a.double().numpy())

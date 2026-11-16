import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tools import bootstrap_comparison


class BootstrapLabelsTests(unittest.TestCase):
    def test_explicit_audited_labels_do_not_require_historical_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'output';output.mkdir()
            labels=root/'valid.csv';labels.write_text('Image_name,True_Label\na,0\nb,0\nc,1\nd,1\n')
            (output/'baseline_single_image_validation.csv').write_text('Image_name,Probability\na,.1\nb,.4\nc,.3\nd,.8\n')
            (output/'validation_predictions.csv').write_text('Image_name,Probability\na,.05\nb,.2\nc,.7\nd,.9\n')
            (output/'selection_report.json').write_text(json.dumps(dict(reference_name='fixed')))
            with patch.object(bootstrap_comparison,'ROOT',root),contextlib.redirect_stdout(io.StringIO()):
                bootstrap_comparison.main(output,labels_path=labels)
            result=json.loads((output/'bootstrap_comparison.json').read_text())
            self.assertEqual(result['positive'],2);self.assertEqual(result['negative'],2)
            self.assertEqual(result['auroc_gain'],.25)
            self.assertEqual(result['labels_sha256'],hashlib.sha256(labels.read_bytes()).hexdigest())


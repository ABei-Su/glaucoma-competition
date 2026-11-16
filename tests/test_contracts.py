import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from data import load_rows, preprocess_image, image_index, audit_inputs
from cost import reference_index
from utils import is_better, metrics, write_submission, Lifecycle


class Contracts(unittest.TestCase):
    def test_metrics_known_ranking_and_ties(self):
        result = metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(result['auroc'], 1.0)
        self.assertEqual(result['average_precision'], 1.0)
        self.assertEqual(metrics([0, 1], [0.5, 0.5])['auroc'], 0.5)
        with self.assertRaises(ValueError):
            metrics([0, 0], [0.1, 0.2])

    def test_primary_metric_has_priority(self):
        self.assertTrue(is_better({'auroc': .91, 'average_precision': .2}, {'auroc': .90, 'average_precision': .9}))
        self.assertFalse(is_better({'auroc': .89, 'average_precision': 1.}, {'auroc': .90, 'average_precision': .1}))

    def test_csv_validation_and_exact_filenames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'submission.csv'
            write_submission(path, ['test_1.JPG', 'test_2.png'], [.5, .1])
            with path.open(encoding='utf-8', newline='') as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(list(rows[0]), ['Image_name', 'Probability', 'Pred_Label'])
            self.assertEqual(rows[0]['Image_name'], 'test_1.JPG')
            self.assertEqual(rows[0]['Pred_Label'], '1')
            for names, probs in [(['a','a'], [.1,.2]), (['a'], [float('nan')]), (['a'], [1.1]), (['a'], [])]:
                with self.assertRaises(ValueError):
                    write_submission(path, names, probs)

    def test_preprocessing_deterministic_black_and_wide(self):
        for image in [Image.new('RGB', (700, 300), (120, 70, 20)), Image.new('RGB',(40,50))]:
            a = preprocess_image(image, 224)
            b = preprocess_image(image, 224)
            self.assertEqual(a.size, (224,224))
            np.testing.assert_array_equal(np.array(a), np.array(b))

    def test_training_rejects_test_and_duplicate_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                load_rows(root, 'test')
            (root/'train.csv').write_text('Image_name,Age,True_Label\na.jpg,20,0\na.jpg,20,1\n')
            with self.assertRaises(ValueError):
                load_rows(root, 'train')

    def test_case_insensitive_lookup_preserves_disk_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'test_1.JPG').write_bytes(b'image')
            index=image_index(root)
            self.assertEqual(index['test_1.jpg'].name,'test_1.JPG')

    def test_cost_reference_casefold_and_collision(self):
        self.assertEqual(reference_index([{'Image_name':'a.JPG','Probability':'.2'}])['a.jpg'],.2)
        with self.assertRaises(ValueError):
            reference_index([{'Image_name':'a.JPG','Probability':'.2'},{'Image_name':'a.jpg','Probability':'.2'}])

    def test_audit_rejects_unbound_or_wrong_source(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'READY.json').write_text(json.dumps({'cross_split_exact_duplicate_hashes':0}))
            with self.assertRaises(ValueError):
                audit_inputs(root,root,224)

    def test_audit_detects_changed_label_file(self):
        import hashlib,json
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for split in ['train','valid']:
                (root/(split+'.csv')).write_text('original')
            (root/'manifest.json').write_text('[]')
            fingerprint=hashlib.sha256(b'[]').hexdigest()
            report={'source_root':str(root.resolve()),'size':224,'preprocess_version':1,
                    'cross_split_exact_duplicate_hashes':0,
                    'csv_sha256':{s:hashlib.sha256(b'original').hexdigest() for s in ['train','valid']},
                    'manifest_path':str(root/'manifest.json'),'manifest_sha256':fingerprint}
            (root/'READY.json').write_text(json.dumps(report))
            self.assertEqual(audit_inputs(root,root,224)['manifest_sha256'],fingerprint)
            (root/'valid.csv').write_text('modified')
            with self.assertRaises(ValueError):
                audit_inputs(root,root,224)

    def test_lifecycle_rejects_skip_and_terminal_restart(self):
        state = Lifecycle()
        with self.assertRaises(ValueError):
            state.transition('complete')
        for next_state in ['training', 'validating', 'saving', 'training', 'validating', 'saving', 'complete']:
            state.transition(next_state)
        with self.assertRaises(ValueError):
            state.transition('training')


if __name__ == '__main__':
    unittest.main()

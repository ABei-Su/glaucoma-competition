import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
from tools import finalize_v3


class FinalizerTests(unittest.TestCase):
    def test_explicit_selection_rejection_reason_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'reports').mkdir();output=root/'output/v8';output.mkdir(parents=True)
            (root/'reports/v8_status.json').write_text(json.dumps(dict(phase='candidates_compared')))
            (output/'selection_rejected.json').write_text(json.dumps(dict(rejection_reason='Descriptive interval includes zero')))
            def execute(command,**kwargs):return SimpleNamespace(returncode=1 if 'tools/select_v8.py' in command else 0)
            with patch.object(finalize_v3,'ROOT',root),patch('sys.argv',['finalize','--node','unused']),\
                 patch.object(finalize_v3.subprocess,'run',side_effect=execute) as commands:
                finalize_v3.main('v8','v6')
            self.assertEqual(commands.call_count,2)
            self.assertEqual(json.loads((root/'reports/v8_finalization_status.json').read_text())['reason'],'Descriptive interval includes zero')

    def test_packaged_failure_quarantines_new_prediction_instead_of_leaving_upload_ready_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'reports').mkdir();output=root/'output/v7';output.mkdir(parents=True)
            (root/'reports/v7_status.json').write_text(json.dumps(dict(phase='candidates_compared')))
            def execute(command,**kwargs):
                if 'predict.py' in command:
                    (output/'submission.csv').write_text('new predictions')
                    (output/'submission.manifest.json').write_text('{}')
                return SimpleNamespace(returncode=1 if 'tools/package_v7.py' in command else 0)
            with patch.object(finalize_v3,'ROOT',root),patch('sys.argv',['finalize','--node','unused']),\
                 patch.object(finalize_v3.subprocess,'run',side_effect=execute):
                with self.assertRaisesRegex(RuntimeError,'packaging'):finalize_v3.main('v7','v6')
            self.assertFalse((output/'submission.csv').exists())
            self.assertEqual((output/'submission.unverified.csv').read_text(),'new predictions')
            self.assertTrue((output/'submission.unverified.manifest.json').exists())
            self.assertEqual(json.loads((root/'reports/v7_finalization_status.json').read_text())['phase'],'failed')

    def test_internal_rejection_is_terminal_without_waiting_or_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'reports').mkdir()
            (root/'reports/v7_status.json').write_text(json.dumps(dict(phase='not_promoted',reason='internal gate failed')))
            with patch.object(finalize_v3,'ROOT',root),patch('sys.argv',['finalize','--node','unused']),\
                 patch.object(finalize_v3.time,'sleep',side_effect=AssertionError('Must not wait after terminal rejection')),\
                 patch.object(finalize_v3.subprocess,'run') as command:
                finalize_v3.main('v7','v6',wait_seconds=1)
            result=json.loads((root/'reports/v7_finalization_status.json').read_text())
            self.assertEqual(result['phase'],'not_promoted')
            self.assertEqual(result['reason'],'internal gate failed')
            command.assert_not_called()

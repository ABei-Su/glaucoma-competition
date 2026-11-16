import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from tools import run_v13
from provenance import digest


class RunnerTests(unittest.TestCase):
    def test_failed_child_stops_without_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'reports').mkdir()
            child=MagicMock();child.pid=123;child.wait.return_value=1
            with patch.object(run_v13,'ROOT',root),patch.object(run_v13.subprocess,'Popen',return_value=child):
                with self.assertRaises(RuntimeError):run_v13.run_child(['dummy.py'],'dummy',1e20)
            self.assertFalse(list(root.rglob('*.csv')))
            self.assertEqual(json.loads((root/'reports/v13_child.json').read_text())['pid'],123)

    def test_timeout_terminates_only_owned_tree_and_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'reports').mkdir()
            child=MagicMock();child.pid=123;child.poll.return_value=None
            child.wait.side_effect=[subprocess.TimeoutExpired('dummy',1),0]
            with patch.object(run_v13,'ROOT',root),patch.object(run_v13.subprocess,'Popen',return_value=child),patch.object(run_v13.subprocess,'run') as kill:
                with self.assertRaises(subprocess.TimeoutExpired):run_v13.run_child(['dummy.py'],'dummy',1e20)
                kill.assert_called_once_with(['taskkill','/PID','123','/T','/F'],check=True,capture_output=True)
            self.assertFalse(list(root.rglob('*.csv')))

    def test_deadline_expired_never_launches(self):
        with patch.object(run_v13.subprocess,'Popen') as launch:
            with self.assertRaises(TimeoutError):run_v13.run_child(['dummy'],'x',0)
            launch.assert_not_called()

    def test_evidence_write_failure_terminates_started_child(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'reports').mkdir()
            child=MagicMock();child.pid=124;child.poll.return_value=None
            with patch.object(run_v13,'ROOT',root),patch.object(run_v13.subprocess,'Popen',return_value=child),patch.object(run_v13,'save_json',side_effect=OSError('disk')),patch.object(run_v13.subprocess,'run') as kill:
                with self.assertRaises(OSError):run_v13.run_child(['dummy.py'],'dummy',1e20)
                kill.assert_called_once_with(['taskkill','/PID','124','/T','/F'],check=True,capture_output=True)

    def test_accept_rejects_incomplete_review_and_changed_model(self):
        for changed in (False,True):
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);out=root/'output/v13';final=out/'finalization'
                final.mkdir(parents=True);(root/'reports').mkdir()
                model=final/'model.pt';model.write_bytes(b'model')
                bound=[dict(path=str(model),sha256=digest(model))]
                (final/'frozen_trial_inputs.json').write_text(json.dumps(dict(files=bound)))
                canonical=final/'submission.csv';canonical.write_bytes(b'csv')
                delivery=root/'submission.csv';delivery.write_bytes(b'csv')
                for name in ('submission.manifest.json','validation_report.json','local_runtime_check.json','result.json'):
                    (final/name).write_text('{}')
                result=dict(delivery=str(delivery),sha256=digest(delivery))
                state=dict(phase='awaiting_review',revision=1,history=[],result=result)
                (root/'reports/v13_status.json').write_text(json.dumps(state))
                paths=list(final.iterdir())+[delivery]
                evidence=[dict(path=str(p),sha256=digest(p)) for p in paths] if changed else []
                (root/'reports/v13_delivery_review.json').write_text(json.dumps(dict(approved=True,evidence=evidence,submission_sha256=digest(delivery))))
                if changed:model.write_bytes(b'modified')
                with patch.object(run_v13,'ROOT',root),patch.object(run_v13,'OUT',out),patch.object(run_v13,'frozen'),patch.object(run_v13.sys,'argv',['run_v13','--stage','accept_delivery']):
                    with self.assertRaises(ValueError):run_v13.main()
                self.assertEqual(json.loads((root/'reports/v13_status.json').read_text())['phase'],'awaiting_review')
                self.assertFalse((out/'operation.lock').exists())


if __name__=='__main__':unittest.main()

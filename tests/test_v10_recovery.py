import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import recover_v10
from tools.recover_v10 import archive_incomplete, check_incomplete, process_alive
from v10_integrity import recipe


class V10RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root/'runs/v10_inner_age'
        self.run.mkdir(parents=True)
        (self.run/'config.json').write_text(json.dumps(recipe('runs/v10_inner_age')))
        (self.run/'progress.json').write_text(json.dumps(dict(epoch=1,step=1200)))
        (self.run/'training.lock').write_text('47028')
        self.lock = self.root/'output/v10/operation.lock'
        self.lock.parent.mkdir(parents=True)
        self.lock.write_text('51844')
        self.state = dict(phase='internal_training',resource_passed=True,run='v10_inner_age')

    def test_live_owner_rejects_without_mutating_any_file(self):
        for active in (47028,51844):
            with self.assertRaisesRegex(ValueError,'live'):
                check_incomplete(self.root,self.state,lambda pid:pid==active)
        self.assertTrue(self.lock.exists())
        self.assertTrue(self.run.exists())

    def test_reject_checkpoint_changed_recipe_terminal_or_unknown_content(self):
        for name in ('last.pt','best.pt','history.jsonl','unexpected.txt'):
            path=self.run/name;path.write_text('preserve')
            with self.assertRaises(ValueError):check_incomplete(self.root,self.state,lambda pid:False)
            path.unlink()
        saved=json.loads((self.run/'config.json').read_text())
        saved['seed']+=1;(self.run/'config.json').write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError,'recipe'):check_incomplete(self.root,self.state,lambda pid:False)
        (self.run/'config.json').write_text(json.dumps(recipe('runs/v10_inner_age')))
        with self.assertRaises(ValueError):check_incomplete(self.root,dict(phase='complete'),lambda pid:False)

    def test_archive_preserves_bytes_and_refuses_repeated_or_outside_target(self):
        expected={p.name:p.read_bytes() for p in self.run.iterdir()}
        check=check_incomplete(self.root,self.state,lambda pid:False)
        destination=self.root/'output/v10/recovery_001'
        archive_incomplete(self.root,destination,check,lambda pid:False)
        self.assertFalse(self.run.exists());self.assertFalse(self.lock.exists())
        self.assertEqual({p.name:p.read_bytes() for p in (destination/'interrupted_run').iterdir()},expected)
        self.assertEqual((destination/'operation.lock.interrupted').read_text(),'51844')
        with self.assertRaises((ValueError,FileExistsError)):
            archive_incomplete(self.root,destination,check,lambda pid:False)
        with self.assertRaises(ValueError):
            archive_incomplete(self.root,self.root.parent/'outside',check,lambda pid:False)

    def test_archive_rechecks_owner_and_content_before_move(self):
        check=check_incomplete(self.root,self.state,lambda pid:False)
        with self.assertRaisesRegex(ValueError,'live'):
            archive_incomplete(self.root,self.root/'output/v10/recovery_001',check,lambda pid:True)
        (self.run/'progress.json').write_text(json.dumps(dict(epoch=1,step=1201)))
        with self.assertRaisesRegex(ValueError,'changed'):
            archive_incomplete(self.root,self.root/'output/v10/recovery_001',check,lambda pid:False)
        self.assertTrue(self.run.exists());self.assertTrue(self.lock.exists())

    def test_new_checkpoint_unknown_file_or_directory_after_check_prevents_move(self):
        check=check_incomplete(self.root,self.state,lambda pid:False)
        for name,is_directory in [('last.pt',False),('unknown.txt',False),('subdirectory',True)]:
            with self.subTest(name=name):
                path=self.run/name
                if is_directory:path.mkdir()
                else:path.write_text('new content')
                with self.assertRaises(ValueError):
                    archive_incomplete(self.root,self.root/'output/v10/recovery_001',check,lambda pid:False)
                self.assertTrue(self.run.exists());self.assertTrue(self.lock.exists())
                self.assertFalse((self.root/'output/v10/recovery_001').exists())
                if is_directory:path.rmdir()
                else:path.unlink()

    def test_current_process_is_live_and_invalid_pid_rejected(self):
        self.assertTrue(process_alive(os.getpid()))
        for pid in (0,-1,True,'47028'):
            with self.assertRaises(ValueError):process_alive(pid)

    def test_missing_or_invalid_status_records_failure_and_releases_own_guard(self):
        for content in (None,'invalid JSON'):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);(root/'output/v10').mkdir(parents=True);(root/'reports').mkdir()
                if content is not None:(root/'reports/v10_status.json').write_text(content)
                with patch.object(recover_v10,'ROOT',root):
                    with self.assertRaises((FileNotFoundError,json.JSONDecodeError)):recover_v10.main()
                self.assertFalse((root/'output/v10/recovery.guard').exists())
                self.assertEqual(json.loads((root/'reports/v10_recovery_status.json').read_text())['phase'],'failed')


if __name__=='__main__':unittest.main()

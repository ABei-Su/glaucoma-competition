import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import shutil
from tools import finalize_v10_age
from tools.finalize_v10_age import trial_allowed


class AgeFinalizationTests(unittest.TestCase):
    def test_only_full_completed_resource_compliant_fixed_candidate_can_predict(self):
        self.assertTrue(trial_allowed('full_ready',100.,499_999_999))
        for phase,latency,size in [('internal_training',80.,200_000_000),('not_promoted',80.,200_000_000),
                                   ('failed',80.,200_000_000),('full_ready',100.01,200_000_000),
                                   ('full_ready',80.,500_000_001),('full_ready',float('nan'),200_000_000)]:
            self.assertFalse(trial_allowed(phase,latency,size))

    def test_failed_copy_never_exposes_official_delivery_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source';source.mkdir()
            (source/'submission.csv').write_text('Image_name,Probability,Pred_Label\n')
            (source/'config.json').write_text('{}')
            destination=root/'delivery';original=shutil.copy2;calls=[]
            def failing_copy(src,dst):
                calls.append(src)
                if len(calls)==2:raise OSError('injected copy failure')
                return original(src,dst)
            self.assertTrue(hasattr(finalize_v10_age,'stage_delivery'),'Atomic delivery helper is missing')
            with patch.object(finalize_v10_age.shutil,'copy2',side_effect=failing_copy):
                with self.assertRaisesRegex(OSError,'injected'):
                    finalize_v10_age.stage_delivery(source,destination,
                        {'submission.csv':'submission.csv','config.json':'config.json'},{'rows':0},'note')
            self.assertFalse(destination.exists())
            self.assertTrue((source/'delivery_staging'/'submission.csv').exists())

    def test_complete_delivery_is_published_once_and_old_files_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source';source.mkdir()
            (source/'submission.csv').write_bytes(b'fixed output')
            destination=root/'delivery'
            self.assertTrue(hasattr(finalize_v10_age,'stage_delivery'),'Atomic delivery helper is missing')
            finalize_v10_age.stage_delivery(source,destination,{'submission.csv':'submission.csv'},{'rows':1},'note')
            self.assertEqual((destination/'submission.csv').read_bytes(),b'fixed output')
            self.assertTrue((destination/'交付校验.json').is_file())
            self.assertTrue((destination/'上传说明.txt').is_file())
            with self.assertRaises(FileExistsError):
                finalize_v10_age.stage_delivery(source,destination,{'submission.csv':'submission.csv'},{'rows':1},'note')
            self.assertEqual((destination/'submission.csv').read_bytes(),b'fixed output')


if __name__=='__main__':unittest.main()

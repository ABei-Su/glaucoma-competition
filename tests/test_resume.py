"""Exercise the real training entrypoint across an epoch-boundary interruption."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import train


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self,*args,**kwargs):
        self.rows=[{'True_Label':str(i%2)} for i in range(8)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        label=index%2
        return torch.rand(3,8,8)+label,torch.tensor(float(label)),str(index)


def tiny_model(*args,**kwargs):
    return torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(3,1))


class ResumeTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_and_rejects_worker_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cache=root/'cache'
            cache.mkdir()
            (cache/'READY.json').write_text('{}')
            original_save=train.save_json

            def invoke(run,resume=False,interrupt=False,workers=0,architecture='resnet18',memory_format='contiguous',checkpoint_epoch=0,augmentation='mild'):
                args=['train.py','--cache',str(cache),'--run',str(run),'--epochs','2','--batch','4',
                      '--workers',str(workers),'--size','32','--architecture',architecture,
                      '--memory-format',memory_format,'--checkpoint-epoch',str(checkpoint_epoch),'--augmentation',augmentation]+(['--resume'] if resume else [])
                def save(path,value):
                    original_save(path,value)
                    if interrupt and Path(path).name=='best_metrics.json':
                        raise KeyboardInterrupt('simulated interruption after atomic epoch checkpoint')
                with patch('sys.argv',args),patch.object(train,'FundusDataset',SyntheticDataset),\
                     patch.object(train,'build_model',tiny_model),patch.object(train,'audit_inputs',return_value={'manifest_sha256':'synthetic-v1'}),\
                     patch.object(train.torch.cuda,'is_available',return_value=False),patch.object(train,'save_json',side_effect=save),\
                     contextlib.redirect_stdout(io.StringIO()):
                    train.main()

            full=root/'full'
            resumed=root/'resumed'
            invoke(full)
            with self.assertRaises(KeyboardInterrupt):
                invoke(resumed,interrupt=True)
            self.assertFalse((resumed/'training.lock').exists())
            with self.assertRaisesRegex(ValueError,'architecture'):
                invoke(resumed,resume=True,architecture='resnet50')
            with self.assertRaisesRegex(ValueError,'memory_format'):
                invoke(resumed,resume=True,memory_format='channels_last')
            with self.assertRaisesRegex(ValueError,'workers'):
                invoke(resumed,resume=True,workers=1)
            with self.assertRaisesRegex(ValueError,'checkpoint_epoch'):
                invoke(resumed,resume=True,checkpoint_epoch=1)
            with self.assertRaisesRegex(ValueError,'augmentation'):
                invoke(resumed,resume=True,augmentation='color_robust')
            invoke(resumed,resume=True)
            a=torch.load(full/'last.pt',weights_only=False)
            b=torch.load(resumed/'last.pt',weights_only=False)
            self.assertEqual(a['epoch'],b['epoch'])
            for name in a['model']:
                torch.testing.assert_close(a['model'][name],b['model'][name],rtol=0,atol=0)
            self.assertEqual(a['best']['auroc'],b['best']['auroc'])
            with patch.object(train,'probabilities_from_logits',wraps=train.probabilities_from_logits) as conversion:
                invoke(root/'r50',architecture='resnet50')
                self.assertGreater(conversion.call_count,0)
            r50=torch.load(root/'r50/last.pt',weights_only=False)
            self.assertEqual(r50['config']['validation_probability_precision'],'fp32')
            invoke(root/'fixed',checkpoint_epoch=2)
            fixed=torch.load(root/'fixed/best.pt',weights_only=True)
            self.assertEqual(fixed['metrics']['epoch'],2)
            self.assertEqual(fixed['checkpoint_epoch'],2)


if __name__=='__main__':
    unittest.main()

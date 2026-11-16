import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from PIL import Image
from internal_split import make_manifest
try:
    import focal_loss
    import train_focal
except ModuleNotFoundError:
    focal_loss=None;train_focal=None


def tiny(*args,**kwargs):
    return nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(3,1))


class FocalLossTests(unittest.TestCase):
    def ready(self):self.assertIsNotNone(focal_loss,'Fixed focal loss is missing')

    def test_gamma_zero_matches_bce_and_class_symmetry_without_alpha(self):
        self.ready();z=torch.tensor([-5.,-1.,0.,1.,5.]);y=torch.tensor([0.,1.,0.,1.,0.])
        actual=focal_loss.FocalBinaryLoss(gamma=0)(z,y)
        torch.testing.assert_close(actual,torch.nn.functional.binary_cross_entropy_with_logits(z,y))
        torch.testing.assert_close(focal_loss.FocalBinaryLoss()(z,y),focal_loss.FocalBinaryLoss()(-z,1-y))

    def test_fp32_loss_extreme_logits_finite_gradients_and_hard_examples_weight_more(self):
        self.ready();z=torch.tensor([-80.,-2.,2.,80.],dtype=torch.float16,requires_grad=True)
        y=torch.tensor([1.,0.,1.,0.]);value=focal_loss.FocalBinaryLoss()(z,y)
        self.assertEqual(value.dtype,torch.float32);self.assertTrue(torch.isfinite(value))
        value.backward();self.assertTrue(torch.isfinite(z.grad).all())
        self.assertLess(z.grad[0].item(),0);self.assertGreater(z.grad[-1].item(),0)
        easy=focal_loss.FocalBinaryLoss()(torch.tensor([4.]),torch.tensor([1.]))
        hard=focal_loss.FocalBinaryLoss()(torch.tensor([-4.]),torch.tensor([1.]))
        self.assertGreater(hard.item(),easy.item())

    def test_rejects_invalid_targets_shapes_parameters_and_handles_single_class(self):
        self.ready()
        for gamma,alpha in [(-1,-1),(float('nan'),-1),(2,2),(2,float('inf'))]:
            with self.assertRaises(ValueError):focal_loss.FocalBinaryLoss(gamma=gamma,alpha=alpha)
        for z,y in [(torch.zeros(2),torch.zeros(1)),(torch.zeros(1),torch.tensor([2.])),
                    (torch.tensor([float('nan')]),torch.zeros(1)),(torch.empty(0),torch.empty(0))]:
            with self.assertRaises(ValueError):focal_loss.FocalBinaryLoss()(z,y)
        for label in (0.,1.):self.assertTrue(torch.isfinite(focal_loss.FocalBinaryLoss()(torch.zeros(4),torch.full((4,),label))))

    def test_known_synthetic_signal_is_learned(self):
        self.ready()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(12);model=nn.Linear(1,1);optimizer=torch.optim.SGD(model.parameters(),lr=.5)
            x=torch.tensor([[-2.],[2.]]);y=torch.tensor([0.,1.]);criterion=focal_loss.FocalBinaryLoss()
            for _ in range(80):
                optimizer.zero_grad();loss=criterion(model(x).flatten(),y);loss.backward();optimizer.step()
            p=model(x).flatten().sigmoid().detach()
            self.assertLess(p[0].item(),.2);self.assertGreater(p[1].item(),.8)


class FocalTrainerTests(unittest.TestCase):
    def fixture(self,root):
        raw=root/'raw';(raw/'train').mkdir(parents=True)
        rows=[dict(Image_name=f'train_{i}.png',True_Label=str(i%2)) for i in range(40)]
        with (raw/'train.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['Image_name','True_Label']);writer.writeheader();writer.writerows(rows)
        for row in rows:Image.new('RGB',(32,32),(30+int(row['True_Label'])*170,60,40)).save(raw/'train'/row['Image_name'])
        split=root/'split.json';split.write_text(json.dumps(make_manifest(rows,(raw/'train.csv').read_bytes())))
        cache=root/'cache';cache.mkdir();(cache/'READY.json').write_text('{}')
        public=root/'public.pth';public.write_bytes(b'synthetic public source')
        return raw,split,cache,public

    def invoke(self,paths,run,resume=False,interrupt=False,extra=None):
        raw,split,cache,public=paths
        args=['--data',str(raw),'--cache',str(cache),'--pretrain',str(public),'--split-manifest',str(split),
              '--architecture','convnext_tiny','--run',str(run),'--epochs','2','--batch','8','--workers','0','--size','32']
        args+=['--resume'] if resume else [];args+=extra or []
        original=train_focal.save_json
        def save(path,value):
            original(path,value)
            if interrupt and Path(path).name=='best_metrics.json':raise KeyboardInterrupt('epoch-boundary fixture')
        with patch.object(train_focal,'build_model',tiny),patch.object(train_focal,'audit_inputs',return_value={'manifest_sha256':'synthetic'}),\
             patch.object(train_focal.torch.cuda,'is_available',return_value=False),patch.object(train_focal,'save_json',side_effect=save),\
             contextlib.redirect_stdout(io.StringIO()):train_focal.main(args)

    def test_epoch_resume_matches_uninterrupted_and_persists_actual_loss_contract(self):
        self.assertIsNotNone(train_focal,'Isolated focal trainer is missing')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);paths=self.fixture(root)
            self.invoke(paths,root/'full')
            with self.assertRaises(KeyboardInterrupt):self.invoke(paths,root/'resumed',interrupt=True)
            self.assertFalse((root/'resumed/training.lock').exists());self.invoke(paths,root/'resumed',resume=True)
            a=torch.load(root/'full/last.pt',weights_only=False);b=torch.load(root/'resumed/last.pt',weights_only=False)
            self.assertEqual(a['loss_spec'],focal_loss.loss_spec());self.assertEqual(a['loss_spec'],b['loss_spec'])
            for name,value in a['model'].items():torch.testing.assert_close(value,b['model'][name],rtol=0,atol=0)
            selected=torch.load(root/'resumed/best.pt',weights_only=True)
            self.assertEqual(selected['loss_spec'],focal_loss.loss_spec())
            self.assertEqual(selected['validation_partition'],'internal_train_holdout')
            with self.assertRaises(FileExistsError):self.invoke(paths,root/'resumed',resume=True)

    def test_resume_rejects_loss_or_public_hash_change_and_scope_is_explicit(self):
        self.assertIsNotNone(train_focal,'Isolated focal trainer is missing')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);paths=self.fixture(root);run=root/'run'
            with self.assertRaises(KeyboardInterrupt):self.invoke(paths,run,interrupt=True)
            with self.assertRaisesRegex(ValueError,'focal_gamma|loss'):self.invoke(paths,run,resume=True,extra=['--focal-gamma','1'])
            paths[3].write_bytes(b'changed source')
            with self.assertRaisesRegex(ValueError,'public'):self.invoke(paths,run,resume=True)
        with contextlib.redirect_stderr(io.StringIO()):
            for args in [['--full'],['--split-manifest',''],['--init-checkpoint','exposed.pt'],['--pos-weight','2']]:
                with self.assertRaises(SystemExit):train_focal.parse_args(args)


if __name__=='__main__':unittest.main()

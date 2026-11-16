import copy
import hashlib
import json
import unittest

import numpy as np
import torch

from v13_core import Mixup, mix_batch, soft_bce, validate_folds, promotion


class MixupTests(unittest.TestCase):
    def test_endpoints_and_shared_permutation(self):
        x = torch.arange(12.).reshape(3, 4)
        y = torch.tensor([0., 1., 0.])
        permutation = torch.tensor([1, 2, 0])
        for lam in (0., .3, 1.):
            a, b = mix_batch(x, y, lam, permutation)
            torch.testing.assert_close(a, lam*x+(1-lam)*x[permutation])
            torch.testing.assert_close(b, lam*y+(1-lam)*y[permutation])
        with self.assertRaises(ValueError):
            mix_batch(x, y, .5, torch.tensor([0, 0, 1]))

    def test_rng_isolation_and_resume(self):
        x = torch.arange(12.).reshape(3, 4)
        y = torch.tensor([0., 1., 0.])
        torch.manual_seed(123)
        np.random.seed(123)
        expected = torch.rand(4)
        expected_np = np.random.rand(4)
        torch.manual_seed(123)
        np.random.seed(123)
        mixer = Mixup(99)
        mixer(x, y)
        saved = copy.deepcopy(mixer.state_dict())
        expected_batch = mixer(x, y)
        restored = Mixup(99)
        restored.load_state_dict(saved)
        for a, b in zip(expected_batch, restored(x, y)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(torch.rand(4), expected)
        np.testing.assert_array_equal(np.random.rand(4), expected_np)

    def test_loss_fp32_soft_targets_and_gradients(self):
        logits = torch.tensor([-2., 3.], dtype=torch.float16, requires_grad=True)
        labels = torch.tensor([.2, .8])
        loss = soft_bce(logits, labels)
        self.assertEqual(loss.dtype, torch.float32)
        torch.testing.assert_close(loss, torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), labels))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_complete_epoch_resume_replays_optimizer_and_mixup(self):
        torch.manual_seed(23)
        x = torch.arange(32.).reshape(8,4)/32
        y = torch.arange(8).remainder(2).float()
        model = torch.nn.Linear(4,1)
        optimizer = torch.optim.AdamW(model.parameters(),lr=.01)
        mixer = Mixup(123)
        loader_rng = torch.Generator().manual_seed(45)
        def epoch(net, opt, mix, generator):
            order = torch.randperm(8,generator=generator)
            for indices in order.split(4):
                images, labels = mix(x[indices],y[indices])
                opt.zero_grad();loss=soft_bce(net(images).flatten(),labels)
                loss.backward();opt.step()
        epoch(model,optimizer,mixer,loader_rng)
        state = copy.deepcopy((model.state_dict(),optimizer.state_dict(),mixer.state_dict(),loader_rng.get_state()))
        epoch(model,optimizer,mixer,loader_rng)
        restored = torch.nn.Linear(4,1);restored.load_state_dict(state[0])
        opt = torch.optim.AdamW(restored.parameters(),lr=.01);opt.load_state_dict(state[1])
        mix = Mixup(123);mix.load_state_dict(state[2])
        gen = torch.Generator();gen.set_state(state[3])
        epoch(restored,opt,mix,gen)
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)


class FoldTests(unittest.TestCase):
    def fixture(self):
        rows = [{'Image_name': str(i)+'.jpg', 'True_Label': str(i % 2)} for i in range(6)]
        items = [{'split': 'train', 'name': r['Image_name'], 'sha256': hashlib.sha256(r['Image_name'].encode()).hexdigest()} for r in rows]
        blob = json.dumps(items).encode()
        names = [r['Image_name'] for r in rows]
        manifest = dict(schema_version=2, source_split='train', seed=20261029,
                        method='StratifiedGroupKFold exact byte hash groups', patient_independence_verified=False,
                        train_csv_sha256=hashlib.sha256(b'csv').hexdigest(),
                        image_manifest_sha256=hashlib.sha256(blob).hexdigest(), folds=[])
        for fold in range(3):
            held = names[2*fold:2*fold+2]
            manifest['folds'].append(dict(fold=fold, train_names=[n for n in names if n not in held],
                                         holdout_names=held, train_rows=4, holdout_rows=2, holdout_positive=1))
        return rows, blob, manifest

    def test_valid_partition(self):
        rows, blob, manifest = self.fixture()
        train, valid = validate_folds(rows, b'csv', blob, manifest, 1)
        self.assertEqual([r['Image_name'] for r in valid], ['2.jpg', '3.jpg'])
        self.assertEqual(len(train), 4)

    def test_conflicting_labels_in_same_hash_group_rejected(self):
        rows, blob, manifest = self.fixture()
        items=json.loads(blob);items[1]['sha256']=items[0]['sha256']
        blob=json.dumps(items).encode()
        manifest['image_manifest_sha256']=hashlib.sha256(blob).hexdigest()
        with self.assertRaises(ValueError):validate_folds(rows,b'csv',blob,manifest,0)

    def test_reject_bad_boundaries_and_identity(self):
        for kind in ('duplicate', 'missing', 'extra', 'overlap', 'count', 'label', 'hash', 'csv', 'fold', 'group'):
            rows, blob, manifest = self.fixture()
            fold = 0
            if kind == 'duplicate': manifest['folds'][0]['train_names'][0] = manifest['folds'][0]['train_names'][1]
            if kind == 'missing': manifest['folds'][1]['holdout_names'].pop()
            if kind == 'extra': manifest['folds'][1]['holdout_names'].append('evil.jpg')
            if kind == 'overlap': manifest['folds'][0]['train_names'][0] = '0.jpg'
            if kind == 'count': manifest['folds'][0]['holdout_positive'] = 2
            if kind == 'label': rows[0]['True_Label'] = '2'
            if kind == 'hash': manifest['image_manifest_sha256'] = 'bad'
            if kind == 'csv': manifest['train_csv_sha256'] = '0'*64
            if kind == 'fold': fold = 3
            if kind == 'group':
                items = json.loads(blob)
                items[2]['sha256'] = items[0]['sha256']
                blob = json.dumps(items).encode()
                manifest['image_manifest_sha256'] = hashlib.sha256(blob).hexdigest()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_folds(rows, b'csv', blob, manifest, fold)

    def test_gate_requires_all_folds_and_all_guards(self):
        self.assertTrue(promotion([.001, .001, -.0002], .8, .8))
        for ds, new_ap in (([.01], .9), ([.002, .002, -.002], .9), ([.001, .001, 0.], .79), ([.0001]*3, .9)):
            self.assertFalse(promotion(ds, .8, new_ap))


if __name__ == '__main__':
    unittest.main()

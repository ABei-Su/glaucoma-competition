import unittest
import tempfile
from pathlib import Path
import torch
from train_mixup import parse_args, validation_probability, commit_predictions, resume_result


class TrainerContract(unittest.TestCase):
    def test_fixed_recipe_and_modes(self):
        a = parse_args(['--fold', '1', '--treatment', 'mixup', '--run', 'runs/test'])
        self.assertEqual((a.seed, a.epochs, a.size, a.batch, a.checkpoint_epoch), (20261102, 8, 384, 16, 8))
        a = parse_args(['--full', '--treatment', 'mixup', '--run', 'runs/test'])
        self.assertEqual(a.seed, 20261104)
        for args in (['--fold','3'], ['--full','--treatment','control'], ['--epochs','7'], ['--init-checkpoint','bad.pt']):
            with self.assertRaises(SystemExit):
                parse_args(['--run','runs/test']+args)

    def test_validation_is_fixed_flip_without_mixup(self):
        class Toy(torch.nn.Module):
            def forward(self, x): return x[:, 0, 0, 0]
        x = torch.tensor([[[[1., 3.]]], [[[-1., 2.]]]])
        p = validation_probability(Toy(), x)
        expected = (torch.sigmoid(x[:,0,0,0])+torch.sigmoid(x[:,0,0,-1]))/2
        torch.testing.assert_close(p, expected)

    def test_final_epoch_commit_windows(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'validation_predictions.csv'
            commit_predictions(path,['a','b'],[0.,1.],[.1,.9])
            before=path.read_bytes()
            # Crash after CSV but before last.pt: deterministic epoch replay is idempotent.
            commit_predictions(path,['a','b'],[0.,1.],[.1,.9])
            self.assertEqual(path.read_bytes(),before)
            with self.assertRaises(ValueError):commit_predictions(path,['a','b'],[0.,1.],[.2,.9])
        # Crash after epoch8 last.pt but before status: result restored without another fit.
        result={'epoch':8,'auroc':.9}
        self.assertEqual(resume_result({'epoch':8,'result':result}),result)
        with self.assertRaises(ValueError):resume_result({'epoch':8,'result':None})


if __name__ == '__main__': unittest.main()

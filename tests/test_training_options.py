import unittest
from train import parse_args


class TrainingOptionsTests(unittest.TestCase):
    def test_baseline_defaults_and_finetuning_options(self):
        base=parse_args([])
        self.assertEqual(base.lr,3e-4)
        self.assertIsNone(base.pos_weight)
        fine=parse_args(['--lr','0.0001','--pos-weight','2','--init-checkpoint','previous.pt'])
        self.assertEqual(fine.lr,1e-4)
        self.assertEqual(fine.pos_weight,2.)
        self.assertEqual(fine.init_checkpoint,'previous.pt')

    def test_reject_invalid_training_options(self):
        for args in [['--lr','0'],['--pos-weight','-1'],['--epochs','0'],['--batch','0'],['--lr','nan']]:
            with self.assertRaises(SystemExit):
                parse_args(args)

    def test_internal_split_forbids_exposed_initialization(self):
        with self.assertRaises(SystemExit):
            parse_args(['--split-manifest','split.json','--init-checkpoint','old.pt'])
        with self.assertRaises(SystemExit):
            parse_args(['--checkpoint-epoch','7','--epochs','6'])
        self.assertEqual(parse_args(['--checkpoint-epoch','2']).checkpoint_epoch,2)


if __name__=='__main__':
    unittest.main()

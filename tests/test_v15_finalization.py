import copy
import unittest
from tools.finalize_v15 import check_training, resource_passes, next_phase, repeat_delta


class V15FinalizationTests(unittest.TestCase):
    def test_repeat_delta_detects_accumulated_drift(self):
        self.assertEqual(repeat_delta([0., .75e-7, 1.5e-7]), 1.5e-7)
        self.assertFalse(resource_passes(80, repeat_delta([0., .75e-7, 1.5e-7]), 100))

    def test_fixed_training_contract(self):
        from train import parse_args
        config=vars(parse_args(['--architecture','resnet50','--cache','data/cache384',
            '--run','runs/v15_full_r50_8','--epochs','8','--checkpoint-epoch','8',
            '--size','384','--batch','16','--workers','4','--seed','20261032',
            '--lr','.0001','--pos-weight','1']))
        status={'state':'complete','best':{'epoch':8}}
        checkpoint={'checkpoint_epoch':8,'metrics':{'epoch':8},'architecture':'resnet50',
                    'size':384,'validation_partition':'official_valid'}
        check_training(config,status,checkpoint,list(range(1,9)))
        for key,value in [('checkpoint_epoch',12),('split_manifest','x'),('init_checkpoint','x'),('epochs',16),('seed',9)]:
            bad=copy.deepcopy(config);bad[key]=value
            with self.assertRaises(ValueError):check_training(bad,status,checkpoint,list(range(1,9)))
        with self.assertRaises(ValueError):check_training(config,status,checkpoint,list(range(1,8)))

    def test_resource_boundaries_and_nonfinite(self):
        self.assertTrue(resource_passes(100,1e-7,500000000))
        for values in [(100.001,0,10),(1,1.1e-7,10),(1,0,500000001),(float('nan'),0,1),(1,float('nan'),1)]:
            self.assertFalse(resource_passes(*values))

    def test_terminal_and_out_of_order_rejection(self):
        phase='created'
        for destination in ['resources','validation','prediction','awaiting_review']:
            phase=next_phase(phase,destination)
        with self.assertRaises(ValueError):next_phase(phase,'prediction')
        with self.assertRaises(ValueError):next_phase('created','prediction')
        self.assertEqual(next_phase('resources','failed'),'failed')
        with self.assertRaises(ValueError):next_phase('failed','resources')


if __name__=='__main__':unittest.main()

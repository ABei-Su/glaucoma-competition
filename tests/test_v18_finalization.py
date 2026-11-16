import copy
import unittest
from tools.finalize_v18 import check_training, resource_passes, next_phase, repeat_delta, TRIALS


class V18FinalizationTests(unittest.TestCase):
    def test_only_one_trial(self):
        self.assertEqual(TRIALS, ('pair',))

    def test_fixed_training_contract(self):
        from train import parse_args
        config=vars(parse_args(['--architecture','resnet50','--cache','data/cache384',
            '--run','runs/v18_r50_bce16','--epochs','16','--checkpoint-epoch','16',
            '--size','384','--batch','16','--workers','4','--seed','20261032',
            '--lr','.0001','--pos-weight','1']))
        status={'state':'complete','best':{'epoch':16}}
        checkpoint={'checkpoint_epoch':16,'metrics':{'epoch':16},'architecture':'resnet50',
                    'size':384,'validation_partition':'official_valid'}
        check_training(config,status,checkpoint,list(range(1,17)))
        for key,value in [('pos_weight',2),('checkpoint_epoch',8),('split_manifest','x'),('init_checkpoint','x'),('epochs',8),('seed',9),('resume',True)]:
            bad=copy.deepcopy(config);bad[key]=value
            with self.assertRaises(ValueError):check_training(bad,status,checkpoint,list(range(1,17)))
        with self.assertRaises(ValueError):check_training(config,status,checkpoint,list(range(1,16)))
        checkpoint['metrics']['epoch']=15
        with self.assertRaises(ValueError):check_training(config,status,checkpoint,list(range(1,17)))

    def test_resource_boundaries_and_nonfinite(self):
        self.assertTrue(resource_passes(100,1e-7,500000000))
        for values in [(100.001,0,10),(1,1.1e-7,10),(1,0,500000001),(float('nan'),0,1),(1,float('nan'),1)]:
            self.assertFalse(resource_passes(*values))
        self.assertFalse(resource_passes(80,repeat_delta([0.,.75e-7,1.5e-7]),100))

    def test_terminal_and_out_of_order_rejection(self):
        phase='created'
        for destination in ['resources','validation','prediction','awaiting_review']:
            phase=next_phase(phase,destination)
        with self.assertRaises(ValueError):next_phase(phase,'prediction')
        with self.assertRaises(ValueError):next_phase('created','prediction')
        self.assertEqual(next_phase('resources','failed'),'failed')
        with self.assertRaises(ValueError):next_phase('failed','resources')

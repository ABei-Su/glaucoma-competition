import copy
import unittest
from tools.finalize_v16 import check_training, resource_passes, next_phase, repeat_delta, check_ancestry


class V16FinalizationTests(unittest.TestCase):
    def test_ancestry_rejects_wrong_stage_hash_and_data(self):
        binding={'checkpoint':'runs/v15_full_r50_8/best.pt','sha256':'abc',
                 'kind':'supervised continuation on official train only'}
        initial={'architecture':'resnet50','metrics':{'epoch':8},'checkpoint_epoch':8,
                 'validation_partition':'official_valid','data_fingerprint':'data'}
        final={'data_fingerprint':'data'}
        check_ancestry(binding,'abc',initial,final,'data')
        for key,value in [('sha256','wrong'),('checkpoint','other.pt')]:
            changed=dict(binding);changed[key]=value
            with self.assertRaises(ValueError):check_ancestry(changed,'abc',initial,final,'data')
        with self.assertRaises(ValueError):check_ancestry(binding,'abc',initial,final,'changed')
        changed=copy.deepcopy(initial);changed['metrics']['epoch']=7
        with self.assertRaises(ValueError):check_ancestry(binding,'abc',changed,final,'data')

    def test_repeat_delta_detects_accumulated_drift(self):
        self.assertEqual(repeat_delta([0., .75e-7, 1.5e-7]), 1.5e-7)
        self.assertFalse(resource_passes(80, repeat_delta([0., .75e-7, 1.5e-7]), 100))

    def test_fixed_training_contract(self):
        from train import parse_args
        config=vars(parse_args(['--architecture','resnet50','--cache','data/cache384',
            '--run','runs/v16_r50_refine4','--epochs','4','--checkpoint-epoch','4',
            '--size','384','--batch','16','--workers','4','--seed','20261033',
            '--lr','.00001','--pos-weight','1','--init-checkpoint','runs/v15_full_r50_8/best.pt']))
        status={'state':'complete','best':{'epoch':4}}
        checkpoint={'checkpoint_epoch':4,'metrics':{'epoch':4},'architecture':'resnet50',
                    'size':384,'validation_partition':'official_valid'}
        check_training(config,status,checkpoint,list(range(1,5)))
        for key,value in [('checkpoint_epoch',12),('split_manifest','x'),('init_checkpoint',None),('epochs',16),('seed',9)]:
            bad=copy.deepcopy(config);bad[key]=value
            with self.assertRaises(ValueError):check_training(bad,status,checkpoint,list(range(1,5)))
        with self.assertRaises(ValueError):check_training(config,status,checkpoint,list(range(1,4)))

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

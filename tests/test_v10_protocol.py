import unittest
from tools.run_v10 import transition


class V10ProtocolTests(unittest.TestCase):
    def test_gated_full_path_rejects_unmeasured_resources_and_changed_epoch(self):
        state=dict(phase='ready',revision=0,history=[])
        state=transition(state,'verifying',{})
        with self.assertRaises(ValueError):transition(state,'internal_training',{})
        state=transition(state,'internal_training',dict(resource_passed=True))
        state=transition(state,'internal_comparison',{})
        with self.assertRaises(ValueError):transition(state,'full_training',dict(gate={'passed':False},frozen_epoch=3))
        with self.assertRaises(ValueError):transition(state,'full_training',dict(gate={'passed':True},frozen_epoch=0))
        state=transition(state,'full_training',dict(gate={'passed':True},frozen_epoch=3))
        with self.assertRaises(ValueError):transition(state,'full_ready',dict(frozen_epoch=4))
        state=transition(state,'full_ready',dict(frozen_epoch=3))
        self.assertEqual(state['revision'],5)
        self.assertEqual(len(state['history']),5)
        with self.assertRaises(ValueError):transition(state,'internal_training',dict(resource_passed=True))

    def test_rejection_and_failure_are_terminal_and_status_fields_cannot_be_injected(self):
        initial=dict(phase='ready',revision=0,history=[])
        verifying=transition(initial,'verifying',{})
        rejected=transition(verifying,'not_promoted',dict(reason='resource'))
        failed=transition(verifying,'failed',dict(error='source mismatch'))
        self.assertEqual(initial['history'],[])
        for state in (rejected,failed):
            with self.assertRaises(ValueError):transition(state,'verifying',{})
        with self.assertRaises(ValueError):transition(initial,'verifying',dict(phase='full_ready'))


if __name__=='__main__':unittest.main()

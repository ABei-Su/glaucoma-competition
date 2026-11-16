import unittest
from v13_protocol import transition, remaining_timeout, resource_gate, stage_guard, training_guard


class ProtocolTests(unittest.TestCase):
    def test_legal_and_terminal_transitions(self):
        state = dict(phase='implementation', revision=0, history=[])
        phases = ['preflight','fold_training','comparison','full_training','finalization','awaiting_review','complete']
        for i, phase in enumerate(phases):
            state = transition(state, phase, now=10+i)
            self.assertEqual(state['revision'], i+1)
        for phase in phases+['failed']:
            with self.assertRaises(ValueError): transition(state, phase, now=20)
        for phase in ['failed','budget_exceeded','not_promoted']:
            terminal = transition(dict(phase='preflight',revision=0,history=[]), phase, now=1)
            with self.assertRaises(ValueError): transition(terminal, 'preflight', now=2)

    def test_illegal_and_duplicate(self):
        for target in ('complete','full_training','implementation'):
            with self.assertRaises(ValueError):
                transition(dict(phase='implementation',revision=0,history=[]),target,now=1)

    def test_deadlines_and_resource_boundaries(self):
        self.assertEqual(remaining_timeout(100, 10, 30),30)
        self.assertEqual(remaining_timeout(100, 90, 30),10)
        with self.assertRaises(TimeoutError): remaining_timeout(100,100,30)
        self.assertTrue(resource_gate(4999999999,100,1e-7,500000000))
        for values in ((5000000000,90,0,1),(1,100.1,0,1),(1,90,1e-6,1),(1,90,0,500000001),(1,float('nan'),0,1)):
            self.assertFalse(resource_gate(*values))

    def test_stage_guard_rejects_gate_bypass(self):
        stage_guard('preflight',{'phase':'preflight'},None)
        stage_guard('finalize',{'phase':'finalization'},{'passed':True,'fixed_epoch':8})
        for state,report in (({'phase':'preflight'},{'passed':True,'fixed_epoch':8}),
                             ({'phase':'finalization'},{'passed':False,'fixed_epoch':8}),
                             ({'phase':'finalization'},{'passed':True,'fixed_epoch':7})):
            with self.assertRaises(ValueError):stage_guard('finalize',state,report)

    def test_training_requires_active_owner_canonical_run_and_deadline(self):
        state=dict(phase='fold_training',pid=123,deadline=100)
        training_guard(state,'123',123,'v13_fold0_control',0,'control',False,now=10)
        cases=[(dict(state,phase='complete'),'123',123,'v13_fold0_control',10),
               (state,'124',123,'v13_fold0_control',10),(state,'123',124,'v13_fold0_control',10),
               (state,'123',123,'another_run',10),(state,'123',123,'v13_fold0_control',101)]
        for s,lock,parent,run,now in cases:
            with self.assertRaises(ValueError):training_guard(s,lock,parent,run,0,'control',False,now)
        full=dict(phase='full_training',pid=123,full_deadline=100)
        training_guard(full,'123',123,'v13_full_mixup',0,'mixup',True,now=10)


if __name__ == '__main__': unittest.main()

"""Reject invalid experiment promotion, duplicate events and terminal restarts."""
import unittest
import tempfile
from pathlib import Path
from v9_protocol import advance, require_fresh_experiment


class V9ProtocolTests(unittest.TestCase):
    def test_existing_terminal_or_partial_evidence_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'output';output.mkdir();status=root/'status.json'
            require_fresh_experiment(output,status)
            for content in ('{"phase":"failed"}', '{"phase":"internal_passed"}'):
                status.write_text(content,encoding='utf-8')
                with self.assertRaises(FileExistsError):require_fresh_experiment(output,status)
                self.assertEqual(status.read_text(encoding='utf-8'),content)
            status.unlink()
            for name in ('experiment.lock','frozen_inputs.json','internal_soup.pt','internal_selection.json'):
                path=output/name;path.write_bytes(b'preserve')
                with self.assertRaises(FileExistsError):require_fresh_experiment(output,status)
                self.assertEqual(path.read_bytes(),b'preserve')
                path.unlink()

    def test_legal_path_has_atomic_history_and_idempotent_last_event(self):
        state = dict(phase='ready', revision=0, history=[])
        for event, phase in enumerate(('verifying', 'composing', 'evaluating', 'evaluating', 'not_promoted'), 1):
            state = advance(state, phase, str(event), {'step': str(event)})
        self.assertEqual(state['revision'], 5)
        self.assertEqual(state['phase'], 'not_promoted')
        self.assertEqual(len(state['history']), 5)
        self.assertEqual(advance(state, 'not_promoted', '5', {'step': '5'}), state)
        self.assertEqual(state['step'], '5')
        with self.assertRaises(ValueError): advance(state, 'verifying', '6', {})
        with self.assertRaises(ValueError): advance(state, 'failed', '5', {})

    def test_illegal_promotion_old_event_and_failure_recovery_are_rejected(self):
        initial = dict(phase='ready', revision=0, history=[])
        with self.assertRaises(ValueError): advance(initial, 'internal_passed', '1', {})
        state = advance(initial, 'verifying', '1', {})
        state = advance(state, 'composing', '2', {})
        with self.assertRaises(ValueError): advance(state, 'evaluating', '1', {})
        failed = advance(state, 'failed', '3', {'error': 'invalid source'})
        self.assertEqual(failed['error'], 'invalid source')
        with self.assertRaises(ValueError): advance(failed, 'composing', '4', {})
        with self.assertRaises(ValueError): advance(initial, 'verifying', '', {})
        with self.assertRaises(ValueError): advance(initial, 'verifying', '1', {'phase': 'internal_passed'})


if __name__ == '__main__': unittest.main()

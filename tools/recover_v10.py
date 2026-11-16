"""One explicit recovery of the interrupted, checkpoint-free first v10 epoch."""
import ctypes
import json
import os
from pathlib import Path
import sys
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from provenance import digest
from utils import save_json
from v8_integrity import verify_evidence
from v10_integrity import frozen,recipe,validate_completed_age
from tools.run_v10 import run,transition


def process_alive(pid):
    if type(pid) is not int or pid<=0:raise ValueError('Invalid process owner PID')
    if os.name=='nt':
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
        kernel.OpenProcess.restype=ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:
            error=ctypes.get_last_error()
            if error==87:return False
            if error==5:return True
            raise OSError(error,'Cannot verify process owner')
        try:
            code=ctypes.c_ulong()
            if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):
                raise OSError(ctypes.get_last_error(),'Cannot inspect process owner')
            return code.value==259
        finally:kernel.CloseHandle(handle)
    try:os.kill(pid,0)
    except ProcessLookupError:return False
    except PermissionError:return True
    return True


def check_incomplete(root,state,alive=process_alive):
    root=Path(root).resolve();directory=root/'runs/v10_inner_age';lock=root/'output/v10/operation.lock'
    if state.get('phase')!='internal_training' or state.get('resource_passed') is not True or state.get('run')!='v10_inner_age':
        raise ValueError('Recovery is limited to the interrupted internal-training phase')
    if directory.resolve()!=directory or lock.resolve()!=lock:raise ValueError('Unsafe recovery source path')
    allowed={'config.json','progress.json','training.lock','age_stats.json','environment.json','split_binding.json'}
    paths=list(directory.iterdir())
    if any(p.name not in allowed or not p.is_file() or p.is_symlink() for p in paths):
        raise ValueError('Preserve checkpoint, outcome or unknown run content; this recovery cannot handle it')
    if json.loads((directory/'config.json').read_text(encoding='utf-8'))!=recipe('runs/v10_inner_age'):
        raise ValueError('Interrupted recipe changed')
    progress=json.loads((directory/'progress.json').read_text(encoding='utf-8'))
    if progress.get('epoch')!=1:raise ValueError('Recovery requires an incomplete first epoch')
    owners=[int(lock.read_text().strip()),int((directory/'training.lock').read_text().strip())]
    if any(alive(pid) for pid in owners):raise ValueError('Refuse recovery while a recorded owner is live')
    return dict(owners=owners,files=[dict(path=str(p.relative_to(root)),sha256=digest(p)) for p in sorted(paths+[lock])])


def archive_incomplete(root,destination,expected,alive=process_alive):
    root=Path(root).resolve();destination=Path(destination).resolve()
    if destination.parent!=root/'output/v10' or destination.name!='recovery_001':
        raise ValueError('Recovery archive must remain at the fixed workspace destination')
    if destination.exists():raise FileExistsError('Preserve existing recovery archive')
    current=check_incomplete(root,dict(phase='internal_training',resource_passed=True,run='v10_inner_age'),alive)
    if current!=expected:raise ValueError('Interrupted content or owners changed before archive')
    # Both resolved source and destination remain inside the named workspace.
    directory=root/'runs/v10_inner_age';lock=root/'output/v10/operation.lock'
    if directory.resolve()!=directory or lock.resolve()!=lock:raise ValueError('Unsafe recovery source path')
    destination.mkdir(exist_ok=False)
    directory.rename(destination/'interrupted_run')
    lock.rename(destination/'operation.lock.interrupted')
    expected_names={Path(item['path']).name for item in expected['files'] if Path(item['path']).name!='operation.lock'}
    if {p.name for p in (destination/'interrupted_run').iterdir()}!=expected_names:
        raise ValueError('Archived directory content changed during move')
    for item in expected['files']:
        source=Path(item['path'])
        archived=(destination/'operation.lock.interrupted' if source.name=='operation.lock'
                  else destination/'interrupted_run'/source.name)
        if digest(archived)!=item['sha256']:raise ValueError('Archived content verification failed')


def main():
    output=ROOT/'output/v10';status=ROOT/'reports/v10_status.json'
    guard=output/'recovery.guard';recovery_status=ROOT/'reports/v10_recovery_status.json'
    if recovery_status.exists():raise FileExistsError('One finite recovery only; preserve previous attempt')
    with guard.open('x') as stream:stream.write(str(os.getpid()))
    state=None;owned_operation=False;history=[];current='ready'
    allowed={'ready':{'verified','failed'},'verified':{'archived','failed'},'archived':{'training','failed'},
             'training':{'continuation','failed'},'continuation':{'complete','failed'},'complete':set(),'failed':set()}
    def recovery(phase,**details):
        nonlocal current
        if phase not in allowed[current]:raise ValueError('Illegal recovery transition')
        history.append(dict(phase=phase,time=datetime.now().astimezone().isoformat(),details=details));current=phase
        save_json(recovery_status,dict(phase=phase,pid=os.getpid(),history=history,**details))
    def advance(phase,**details):
        nonlocal state
        state=transition(state,phase,details);save_json(status,state)
    try:
        state=json.loads(status.read_text(encoding='utf-8'))
        frozen(ROOT);checked=check_incomplete(ROOT,state)
        verification=ROOT/'reports/v10_recovery_verification.log'
        if 'VERIFY PASSED:' not in verification.read_text(encoding='utf-8'):raise ValueError('Recovery verification required')
        sources=[ROOT/name for name in ('tools/recover_v10.py','tests/test_v10_recovery.py',
                 'docs/RECOVERY-v10.md','docs/REVIEW-v10-recovery.md','reports/v10_recovery_verification.log',
                 'tools/finalize_v10_age.py','docs/FINALIZATION-v10.md','output/v10/frozen_inputs.json')]
        evidence=[dict(path=str(path),sha256=digest(path)) for path in sources]
        original_status=digest(status)
        recovery('verified',original_owners=checked['owners'],original_status_sha256=original_status)
        archive=output/'recovery_001'
        if digest(status)!=original_status:raise ValueError('Coordinator status changed before archive')
        archive_incomplete(ROOT,archive,checked)
        save_json(archive/'recovery_record.json',dict(original_state=state,interrupted_files=checked,
            evidence=evidence,policy='Restart identical first epoch from public initialization; no complete checkpoint exists',
            interruption_cause='No caught error recorded; host rebooted since launch; exact earlier termination cause unknown'))
        recovery('archived',archive=str(archive))
        lock=output/'operation.lock'
        with lock.open('x') as stream:stream.write(str(os.getpid()))
        owned_operation=True
        verify_evidence(evidence);frozen(ROOT)
        recovery('training',log='reports/v10_inner_age_recovery_001.log')
        run(['train_age.py','--run','runs/v10_inner_age'],'v10_inner_age_recovery_001.log')
        validate_completed_age(ROOT/'runs/v10_inner_age',recipe('runs/v10_inner_age'),ROOT)
        verify_evidence(evidence);frozen(ROOT);recovery('continuation');advance('internal_comparison')
        run(['tools/compare_v10_internal.py'],'v10_internal_comparison.log')
        selection=json.loads((output/'internal_selection.json').read_text(encoding='utf-8'))
        verify_evidence(selection['evidence']);verify_evidence(evidence);frozen(ROOT)
        if not selection['gate']['passed']:
            advance('not_promoted',gate=selection['gate'],reason='Actual-Age, AP or missing-Age internal gate failed')
            recovery('complete',coordinator_phase='not_promoted');return
        epoch=selection['selected_epoch'];advance('full_training',run='v10_full_age',frozen_epoch=epoch,gate=selection['gate'])
        run(['train_age.py','--run','runs/v10_full_age','--full','--checkpoint-epoch',str(epoch),'--seed','20261041'],'v10_full_age.log')
        validate_completed_age(ROOT/'runs/v10_full_age',recipe('runs/v10_full_age',True,epoch),ROOT)
        verify_evidence(evidence);frozen(ROOT);advance('full_ready',frozen_epoch=epoch,gate=selection['gate'])
        recovery('complete',coordinator_phase='full_ready');print('V10 RECOVERY FULL CHECKPOINT READY',flush=True)
    except BaseException as error:
        if current not in ('complete','failed'):recovery('failed',error=repr(error))
        if owned_operation and state['phase'] not in ('failed','not_promoted','full_ready'):
            advance('failed',error=repr(error))
        raise
    finally:
        if owned_operation and (output/'operation.lock').read_text().strip()==str(os.getpid()):
            (output/'operation.lock').unlink()
        if guard.read_text().strip()==str(os.getpid()):guard.unlink()


if __name__=='__main__':main()

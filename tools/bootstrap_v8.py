import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from v8_integrity import verify_evidence
if __name__=='__main__':
    gate=json.loads((ROOT/'output/v8/validation_ci_gate.json').read_text())
    if not gate['passed']:raise ValueError('Conservative validation filter did not pass')
    verify_evidence(gate['evidence']);print('Previously computed descriptive bootstrap and input hashes verified',flush=True)

"""Single local/CI verification entrypoint; no external data/network needed."""
import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    temporary=ROOT/'tmp'
    temporary.mkdir(exist_ok=True)
    os.environ['TEMP']=str(temporary)
    os.environ['TMP']=str(temporary)
    sources=list(ROOT.glob('*.py'))+list((ROOT/'tools').glob('*.py'))+list((ROOT/'tests').glob('*.py'))
    for source in sources:
        text=source.read_text(encoding='utf-8')
        ast.parse(text,filename=str(source))
        if any(line.startswith(('<<<<<<< ','=======','>>>>>>> ')) for line in text.splitlines()):
            raise ValueError('Merge marker: '+str(source))
    subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,check=True)
    print('VERIFY PASSED: syntax, merge-marker scan, unit and inference integration tests')


if __name__=='__main__':
    main()

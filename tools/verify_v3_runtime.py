from pathlib import Path
from verify_packaged_runtime import main

if __name__=='__main__': main(Path(__file__).resolve().parents[1]/'output/v3')

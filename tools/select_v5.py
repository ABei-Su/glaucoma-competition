from pathlib import Path
from select_optimized import main

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    main(ROOT/'output/v5',ROOT/'output/v3/selected_config.json',True,'v3_fixed_r50_horizontal_tta',local_limit_ms=100)

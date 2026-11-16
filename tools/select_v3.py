"""Promote v3 only when the actual raw-image path improves on frozen v2."""
from pathlib import Path
from select_optimized import main

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    main(ROOT/'output/v3',ROOT/'output/optimized/selected_config.json',require_improvement=True,
         reference_name='v2_fixed_ensemble')

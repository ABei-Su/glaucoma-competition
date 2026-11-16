"""Fixed v3 candidates and the preserved v2 reference; validation only."""
from pathlib import Path
import compare_candidates as comparison

ROOT=Path(__file__).resolve().parents[1]
comparison.RUNS=['baseline_r18_224','opt_r18_384_w2','opt_r18_384_w1','v3_r50_384_w2','v3_r50_384_w1']

if __name__=='__main__': comparison.main(ROOT/'output/v3')

from pathlib import Path
import compare_candidates as comparison

ROOT=Path(__file__).resolve().parents[1]
comparison.RUNS=['v3_r50_384_w1','v4_r50_448','v4_r50_384','opt_r18_384_w1']

if __name__=='__main__': comparison.main(ROOT/'output/v4')

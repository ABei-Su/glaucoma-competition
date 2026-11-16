"""Choose a validated candidate with local timing margin; never inspect test data."""
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data import image_index,load_rows
from predict import Predictor
from utils import metrics,save_json,write_submission


def main(output=None,reference_config=None,require_improvement=False,reference_name='baseline_r18_224',local_limit_ms=80):
    torch.set_num_threads(4)
    output=Path(output) if output is not None else ROOT/'output/optimized'
    ranked=json.loads((output/'candidate_comparison.json').read_text(encoding='utf-8'))
    raw=ROOT/'data/raw/glaucoma_first'
    rows=load_rows(raw,'valid')
    paths=image_index(raw/'valid')
    probe_dir=ROOT/'tmp/optimization_timing'
    probe_dir.mkdir(parents=True,exist_ok=True)
    timings=[]
    chosen=None
    for rank,item in enumerate(ranked):
        config=output/f'candidate_{rank:02}.json'
        predictor=Predictor(config)
        for row in rows[:5]:
            predictor.predict_one(paths[row['Image_name'].casefold()])
        elapsed=[]
        reference={}
        for repeat in range(3):
            for row in rows[5:105]:
                path=paths[row['Image_name'].casefold()]
                torch.cuda.synchronize()
                started=time.perf_counter()
                probability=predictor.predict_one(path)
                write_submission(probe_dir/(row['Image_name']+'.csv'),[row['Image_name']],[probability])
                torch.cuda.synchronize()
                elapsed.append((time.perf_counter()-started)*1000)
                if repeat and abs(reference[row['Image_name']]-probability)>1e-7:
                    raise RuntimeError('Nondeterministic repeated prediction')
                reference[row['Image_name']]=probability
        mean=sum(elapsed)/len(elapsed)
        timings.append({'rank':rank,'name':item['name'],'mean_ms':mean,'samples':100,'rounds':3,
                        'batch':1,'warmup':5,'source':'official validation images, not official timing set',
                        'gpu':torch.cuda.get_device_name(0),'official_T4_verified':False})
        print('TIMING',rank,mean,flush=True)
        # A local engineering margin, not a claim about T4 performance.
        if mean<=local_limit_ms:
            chosen=dict(item,rank=rank,local_mean_ms=mean)
            break
        del predictor
        torch.cuda.empty_cache()
    save_json(output/'local_timing_candidates.json',timings)
    if chosen is None:
        raise RuntimeError('No candidate met local '+str(local_limit_ms)+'ms gate; investigate without test tuning')
    config=json.loads((output/f"candidate_{chosen['rank']:02}.json").read_text(encoding='utf-8'))
    labels,probabilities,names=[],[],[]
    for n,row in enumerate(rows,1):
        names.append(row['Image_name'])
        labels.append(int(row['True_Label']))
        probabilities.append(predictor.predict_one(paths[row['Image_name'].casefold()]))
        if n%1000==0:
            print('Final inference path validation',n,'/',len(rows),flush=True)
    final_metrics=metrics(labels,probabilities)
    write_submission(output/'validation_predictions.csv',names,probabilities)
    del predictor
    torch.cuda.empty_cache()
    baseline_predictor=Predictor(reference_config or ROOT/'runs/baseline_r18_224/best.pt')
    baseline_probabilities=[]
    for n,row in enumerate(rows,1):
        baseline_probabilities.append(baseline_predictor.predict_one(paths[row['Image_name'].casefold()]))
        if n%1000==0: print('Baseline same-path validation',n,'/',len(rows),flush=True)
    baseline=metrics(labels,baseline_probabilities)
    write_submission(output/'baseline_single_image_validation.csv',names,baseline_probabilities)
    chosen.update(actual_single_image_validation=final_metrics,
                  reference_name=reference_name,
                  auroc_gain_over_baseline=final_metrics['auroc']-baseline['auroc'],
                  baseline_single_image_validation=baseline,
                  official_T4_verified=False,
                  selection_note='Exploratory comparison on the same 191-positive validation set; not independent clinical validation')
    selected_score=(final_metrics['auroc'],final_metrics['average_precision'])
    reference_score=(baseline['auroc'],baseline['average_precision'])
    if selected_score<reference_score or (require_improvement and selected_score==reference_score):
        save_json(output/'selection_rejected.json',chosen)
        raise RuntimeError('Final path underperforms baseline; do not promote')
    config.update(validation_metrics=final_metrics,precision='fp16',batch=1,preprocess_version=1)
    save_json(output/'selected_config.json',config)
    save_json(output/'selection_report.json',chosen)
    print('SELECTED',json.dumps(chosen),flush=True)


if __name__=='__main__':
    main()

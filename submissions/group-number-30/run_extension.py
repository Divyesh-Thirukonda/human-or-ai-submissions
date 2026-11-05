"""One adaptive, predeclared equal-weight character + word/style soft blend."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from data_utils import load_data,get_or_create_split
from experiment import evaluate,paired_bootstrap,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--results',type=Path,default=Path('results'))
    a=p.parse_args();out=a.results
    plan=json.loads((out/'extension_plan.json').read_text())
    config=plan['config']
    if config!={'kind':'char_style_blend','char_C':3.0,'word_style_C':1.0,'weights':[.5,.5]}:
        raise ValueError('This experiment is the one predeclared equal-weight blend')
    data=load_data(a.data/'train.csv',expected_rows=10000)
    train,val=get_or_create_split(data,out/'split.json')
    audit_path=out/'source_audit.json'
    if not audit_path.exists(): audit_path=out/'audit.json'
    audit=json.loads(audit_path.read_text())
    record=evaluate(config,train,val,out/'extension','equal_blend',save_model=True,
        sensitivity_exclude_ids=audit['source_overlap_sensitivity']['excluded_validation_ids'])
    base=pd.read_csv(out/'runs'/'char_lr_C3.predictions.tsv',sep='\t',dtype={'id':str})
    style=pd.read_csv(out/'bonus'/'word_style_lr.predictions.tsv',sep='\t',dtype={'id':str})
    blend=pd.read_csv(out/'extension'/'equal_blend.predictions.tsv',sep='\t',dtype={'id':str})
    if not (base.id.tolist()==style.id.tolist()==blend.id.tolist()): raise ValueError('Unpaired validation IDs')
    expected=(base.probability_ai.to_numpy()+style.probability_ai.to_numpy())/2
    if not np.allclose(expected,blend.probability_ai,rtol=0,atol=1e-8):
        raise ValueError('Refitted blend does not match fixed component average')
    effect=paired_bootstrap(base.label,base.probability_ai,blend.probability_ai)
    promote=effect['delta_auroc']>0 and effect['ci_95_percentile'][0]>0
    recommended=config if promote else {'kind':'char_lr','C':3.0}
    write_json(out/'extension_comparison.json',{'run':record,'baseline_name':'char_lr_C3',
        'effect_vs_core':effect,'promoted_for_day1_handoff':promote,'recommended_config':recommended,
        'selection_rule':plan['promotion_rule'],
        'component_prediction_max_abs_difference':float(np.max(np.abs(expected-blend.probability_ai))),
        'caution':'Adaptive exploratory extension selected after earlier comparisons; bootstrap is not confirmatory. Day 2 should review generalization and complexity.'})
    write_json(out/'recommended_config.json',recommended)
    print(json.dumps({'recommended':recommended,'effect':effect,'promote':promote}))

if __name__=='__main__': main()

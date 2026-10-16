"""Refit the declared recipe on training data only, without tuning or answer files."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
import clean_causal96_stage2 as financial

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=ROOT/'models_retrained')
    args=parser.parse_args()
    assert not args.output_dir.exists(),'Use a new directory to preserve frozen models.'
    raw=pd.read_csv(ROOT/'data/train.csv',encoding='cp949')
    recipe=json.loads((ROOT/'recipe.json').read_text())
    finance=dict(prior=financial.fit_prior(raw,recipe['prior']),
        selected=[financial.fit(raw,s) for s in recipe['selected']],half_blend=recipe['half_blend'])
    health=financial.c.fit_recipe(raw,'Target_2',recipe['health'])
    args.output_dir.mkdir(parents=True)
    joblib.dump(finance,args.output_dir/'Target_1.joblib')
    joblib.dump(health,args.output_dir/'Target_2.joblib')
    print('Fixed recipe refitted; final target-label training boundary: Time_step 921.')

if __name__=='__main__':main()

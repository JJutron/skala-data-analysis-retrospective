"""Reproduce the frozen highest reported eligible candidate without reading answers."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import clean_causal96_stage2 as financial

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-dir',type=Path,default=ROOT/'models')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(ROOT/'data/train.csv',encoding='cp949')
    finance=joblib.load(args.model_dir/'Target_1.joblib')
    health=joblib.load(args.model_dir/'Target_2.joblib')
    for kind in ['public','private']:
        test=pd.read_csv(ROOT/f'data/test_{kind}.csv',encoding='cp949').drop(columns=financial.c.TARGETS,errors='ignore')
        raw=pd.concat([train,test],ignore_index=True)
        out=pd.read_csv(ROOT/f'data/sample_submission_{kind}.csv')
        p1=financial.predict_finance(finance,raw)
        p2=financial.c.predict_recipe(health,raw)
        for values,name in [(p1,'pred_1'),(p2,'pred_2')]:
            out[name]=pd.Series(values,index=raw.Time_step).loc[out.Time_step].to_numpy()
        expected=pd.read_csv(ROOT/f'expected/{kind}.csv')
        pd.testing.assert_frame_equal(out,expected,rtol=0,atol=1e-9)
        assert np.isfinite(out[['pred_1','pred_2']]).all().all()
        assert out[['pred_1','pred_2']].ge(0).all().all() and out[['pred_1','pred_2']].le(1).all().all()
        out.to_csv(args.output_dir/f'{kind}_P071_김형주.csv',index=False)
        print(kind,len(out),'rows: matched frozen predictions')

if __name__=='__main__':main()

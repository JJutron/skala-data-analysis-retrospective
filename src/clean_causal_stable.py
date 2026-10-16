"""A smaller train-only follow-up after unstable nested candidate selection.

Both 150-row and 100-row development windows must improve over the fixed v2
baseline. These reused windows rank recipes; they are not independent test scores.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import clean_causal as c

OUT = c.ROOT/'clean_causal_stable_run'
FOLDS = [(474,150),(624,150),(774,150),(524,100),(624,100),(724,100),(824,100)]


def specs(target):
    baseline = c.specifications(target)[0]
    if target == 'Target_1':
        # The original v3 recipe was based on train CV; retrain without its model cache.
        core = dict(name='v3_core_fixed',kind='baseline_fin',group='core_weekday',C=.02)
        return [baseline,core], {'baseline':[baseline['name']], 'core':[core['name']],
                                'anchored_half':[baseline['name'],core['name']]}
    calendar3 = dict(name='calendar_cat3',kind='cat',group='food_calendar',depth=3)
    calendar5 = dict(name='calendar_cat5',kind='cat',group='food_calendar',depth=5)
    return [baseline,calendar3,calendar5], {'baseline':[baseline['name']],
        'calendar3':[calendar3['name']], 'calendar5':[calendar5['name']],
        'anchored3':[baseline['name'],calendar3['name']],
        'anchored5':[baseline['name'],calendar5['name']]}


def fit(raw,target,spec,end):
    frame,groups = c.feature_frames(raw)
    groups['core_weekday'] = list(c.OPTIMIZED_FEATURES_TARGET_1)
    if spec['name'] != 'v3_core_fixed':
        return c.fit_one(frame,groups,raw[target],spec,end)
    cols = groups['core_weekday']
    m = c.make_model('optimized',cols,[x for x in cols if x!='Weekday'],['Weekday'],{'C':.02})
    m.fit(frame.iloc[c.FIRST:end-c.GAP][cols],raw[target].iloc[c.FIRST:end-c.GAP])
    return dict(model=m,spec=spec,features=cols,fit_first_time=c.FIRST+1,
                fit_last_time=end-c.GAP,label_maturity_last_time=end,warnings=[])


def choose(records):
    base = records['baseline']
    eligible = [name for name,r in records.items() if name!='baseline'
                and r['mean150'] > base['mean150']+.002
                and r['mean100'] > base['mean100']+.002]
    # Selection is confined to development labels. Never open Public answers.
    return max(eligible,key=lambda n:records[n]['mean']) if eligible else 'baseline'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true')
    args = parser.parse_args()
    c.OUT = OUT
    if args.verify:
        report = json.loads((OUT/'report.json').read_text())
        for path,expected in report['source_hashes'].items():
            assert c.sha(c.ROOT/path)==expected
        for target,expected in report['model_hashes'].items():
            assert c.sha(OUT/f'{target}.joblib')==expected
        bundles = {t:joblib.load(OUT/f'{t}.joblib') for t in c.TARGETS}
        assert c.export(bundles,verify=True)==report['inference_checks']
        print('PASS:',c.verify_preservation(),'existing artifacts unchanged; all 300 prefixes and saved-model predictions match.')
        return
    assert not OUT.exists()
    OUT.mkdir()
    (OUT/'existing_hashes.json').write_bytes((c.ROOT/'clean_causal_run/existing_hashes.json').read_bytes())
    raw = c.read_data(c.ROOT/'train.csv')
    report = dict(state='training',selection_uses_public_labels=False,
        uses_future_observations=False,previous_models_or_predictions_used=False,
        validation_role='reused_train_development_not_independent',
        earlier_nested_run='clean_causal_run/report.json',folds=FOLDS,targets={})
    bundles = {}
    prediction_records = []
    for target in c.TARGETS:
        candidates,recipes = specs(target)
        cached = {s['name']:[] for s in candidates}
        for end,size in FOLDS:
            for spec in candidates:
                # Only observations before validation are used in this fit.
                b = fit(raw.iloc[:end],target,spec,end)
                p = c.predict_recipe([b],raw.iloc[:end+size])[end:]
                cached[spec['name']].append(p)
            print('FOLD',target,end,size,flush=True)
        records = {}
        for name,members in recipes.items():
            predictions = [np.mean([cached[m][i] for m in members],axis=0) for i in range(len(FOLDS))]
            scores = [float(roc_auc_score(raw[target].iloc[e:e+s],p)) for (e,s),p in zip(FOLDS,predictions)]
            records[name] = dict(members=members,scores=scores,mean150=float(np.mean(scores[:3])),
                                 mean100=float(np.mean(scores[3:])),mean=float(np.mean(scores)))
            for (end,size),p in zip(FOLDS,predictions):
                prediction_records.extend(dict(target=target,recipe=name,fold_start=end+1,fold_size=size,
                    Time_step=int(time),label=int(y),prediction=float(z))
                    for time,y,z in zip(raw.Time_step.iloc[end:end+size],raw[target].iloc[end:end+size],p))
        winner = choose(records)
        selected = [s for s in candidates if s['name'] in recipes[winner]]
        bundles[target] = [fit(raw,target,s,len(raw)) for s in selected]
        joblib.dump(bundles[target],OUT/f'{target}.joblib')
        report['targets'][target] = dict(records=records,winner=winner,selected=selected)
        (OUT/'progress.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print('SELECTED',target,winner,records[winner],flush=True)
    report['development_mean_auc'] = {window:{key:float(np.mean([
        r['records']['baseline' if key=='baseline' else r['winner']][window]
        for r in report['targets'].values()])) for key in ['baseline','selected']}
        for window in ['mean150','mean100','mean']}
    report['final_fit_last_time'] = 921
    report['public_auc'] = None
    report['private_auc'] = None
    report['model_hashes'] = {t:c.sha(OUT/f'{t}.joblib') for t in c.TARGETS}
    report['source_hashes'] = {p:c.sha(c.ROOT/p) for p in ['clean_causal.py','clean_causal_stable.py',
        'features.py','validate.py','train.csv','test_public.csv','test_private.csv',
        'sample_submission_public.csv','sample_submission_private.csv']}
    pd.DataFrame(prediction_records).to_csv(OUT/'development_predictions.csv',index=False)
    report['state'] = 'recipes_frozen_before_test_inference'
    (OUT/'progress.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    report['inference_checks'] = c.export(bundles)
    report['existing_artifacts_preserved'] = c.verify_preservation()
    report['state'] = 'completed'
    report['note'] = ('The first nested search did not improve its baseline; it is retained. This simpler '
        'follow-up selects on reused train-only development windows, so scores have selection bias. '
        'No Public or Private answer is read. Future demand is estimated from past weekdays only. '
        'Row-level causal validity assumes current-day inputs were available at the forecast cutoff.')
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('RESULT',report['development_mean_auc'],flush=True)


if __name__=='__main__':
    main()

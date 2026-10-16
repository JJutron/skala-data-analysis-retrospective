"""Train-only financial proxy decomposition; keep the clean health recipe fixed.

Future observations occur only inside each historical training prefix to create
auxiliary training responses. Inference uses past/current observations only.
"""
import argparse
import hashlib
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import clean_causal as c
from clean_causal_stable import FOLDS

ROOT = c.ROOT
OUT = ROOT/'clean_causal96_stage1_run'
US = ['FI_Sp500','FI_Nasdaq','FI_Dow']


def matrix(raw, group):
    frame, groups = c.feature_frames(raw)
    if group == 'core':
        return frame[groups['core']]
    if group == 'macro':
        return frame[groups['macro']+groups['us_calendar']].loc[:,lambda x:~x.columns.duplicated()]
    raise ValueError(group)


def observed_component(raw, financial_columns, offset):
    return pd.DataFrame({col:raw[col].diff(offset) if col in US else np.zeros(len(raw))
                         for col in financial_columns},index=raw.index).fillna(0)


def configurations():
    specs = [dict(name=f'core_o{o}_c{co}_a{a}',group='core',offset=o,C=co,alpha=a)
             for o in [1,2] for co in [.01,.1] for a in [100,1000]]
    specs += [dict(name=f'macro_o{o}',group='macro',offset=o,C=.1,alpha=1000) for o in [1,2]]
    specs += [dict(name=f'observed_o{o}',group='core',offset=o,C=.1,alpha=0) for o in [1,2]]
    return specs


def fit_proxy(raw,spec):
    financial = [col for col in raw if col.startswith('FI_')]
    offset = spec['offset']
    future = pd.DataFrame({col:raw[col].shift(-(3-offset))-raw[col].shift(offset)
                           if col in US else raw[col].shift(-3)-raw[col]
                           for col in financial},index=raw.index)
    end = len(raw)-3
    rows = slice(c.FIRST,end)
    assert future.iloc[rows].notna().all().all()
    assert raw.Time_step.iloc[end-1]+3 == raw.Time_step.iloc[-1]
    teacher = make_pipeline(StandardScaler(),LogisticRegression(C=spec['C'],max_iter=2000,random_state=42))
    with warnings.catch_warnings(record=True) as caught:
        teacher.fit(future.iloc[rows],raw.Target_1.iloc[rows])
        coefficients = teacher[-1].coef_[0]/teacher[0].scale_
        known = observed_component(raw,financial,offset).to_numpy()@coefficients
        remainder_target = teacher.decision_function(future.iloc[rows])-known[c.FIRST:end]
        student = None
        if spec['alpha']:
            student = make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
                                    StandardScaler(),Ridge(alpha=spec['alpha']))
            x = matrix(raw,spec['group'])
            student.fit(x.iloc[rows],remainder_target)
            remainder = student.predict(x.iloc[rows])
        else:
            remainder = np.full(end-c.FIRST,remainder_target.mean())
    reference = known[c.FIRST:end]+remainder
    return dict(spec=spec,financial_columns=financial,coefficients=coefficients,
                student=student,remainder_mean=float(remainder_target.mean()),
                center=float(reference.mean()),scale=max(float(reference.std()),1e-8),
                fit_first_time=c.FIRST+1,fit_last_time=int(raw.Time_step.iloc[end-1]),
                auxiliary_observation_last_time=int(raw.Time_step.iloc[-1]),
                warnings=[str(w.message) for w in caught])


def predict_proxy(bundle,raw):
    spec = bundle['spec']
    known = observed_component(raw,bundle['financial_columns'],spec['offset']).to_numpy()@bundle['coefficients']
    remainder = np.full(len(raw),bundle['remainder_mean']) if bundle['student'] is None else bundle['student'].predict(matrix(raw,spec['group']))
    margin = (known+remainder-bundle['center'])/bundle['scale']
    return .5+np.arctan(margin)/np.pi


def finance_predict(bundle,raw):
    previous = c.predict_recipe(bundle['baseline'],raw)
    if bundle['winner'] == 'baseline':
        return previous
    parts = []
    for component in bundle['components']:
        p = predict_proxy(component,raw)
        parts.append((p+previous)/2 if bundle['blend_with_baseline'] else p)
    return np.mean(parts,axis=0)


def choose(records):
    baseline = records['baseline']
    eligible = [name for name,rec in records.items() if name!='baseline'
                and rec['mean150']>baseline['mean150']+.002
                and rec['mean100']>baseline['mean100']+.002]
    return max(eligible,key=lambda name:records[name]['mean']) if eligible else 'baseline'


def source_paths():
    return ['clean_causal96_stage1.py','clean_causal.py','clean_causal_stable.py','features.py','validate.py',
            'train.csv','test_public.csv','test_private.csv','sample_submission_public.csv','sample_submission_private.csv',
            'clean_causal_stable_run/report.json','clean_causal_stable_run/Target_2.joblib']


def preservation():
    snapshot = json.loads((OUT/'existing_hashes.json').read_text())
    for path,expected in snapshot.items():
        assert c.sha(ROOT/path)==expected,path
    return len(snapshot)


def export(finance,health,verify=False):
    train = c.read_data(ROOT/'train.csv')
    checks = {}
    for kind in ['public','private']:
        data = c.read_data(ROOT/f'test_{kind}.csv').drop(columns=c.TARGETS,errors='ignore')
        sample = c.read_data(ROOT/f'sample_submission_{kind}.csv')
        assert data.Time_step.min()==train.Time_step.max()+1
        assert set(data.Time_step)==set(sample.Time_step)
        raw = pd.concat([train,data],ignore_index=True)
        p1 = finance_predict(finance,raw)
        p2 = c.predict_recipe(health,raw)
        for index in range(len(train),len(raw)):
            short = raw.iloc[:index+1]
            np.testing.assert_allclose(finance_predict(finance,short)[-1],p1[index],rtol=0,atol=1e-12)
            frame,_ = c.feature_frames(short)
            z = np.mean([c.predict_one(b,frame.iloc[[-1]])[0] for b in health])
            np.testing.assert_allclose(z,p2[index],rtol=0,atol=1e-12)
        changed = raw.copy();changed[c.TARGETS] = 1
        np.testing.assert_allclose(p1,finance_predict(finance,changed),rtol=0,atol=1e-12)
        np.testing.assert_allclose(p2,c.predict_recipe(health,changed),rtol=0,atol=1e-12)
        out = sample.copy()
        for pred,name in [(p1,'pred_1'),(p2,'pred_2')]:
            out[name] = pd.Series(pred,index=raw.Time_step).loc[sample.Time_step].to_numpy()
        assert out.shape==(100 if kind=='public' else 200,3)
        assert list(out)==['Time_step','pred_1','pred_2']
        assert np.isfinite(out[['pred_1','pred_2']]).all().all()
        assert out[['pred_1','pred_2']].ge(0).all().all() and out[['pred_1','pred_2']].le(1).all().all()
        prior_path = ROOT/f'clean_causal_stable_run/{kind}_causal_clean.csv'
        previous = pd.read_csv(prior_path)
        np.testing.assert_allclose(out.pred_2,previous.pred_2,rtol=0,atol=1e-12)
        path = OUT/f'{kind}_causal96_stage1.csv'
        if verify:
            pd.testing.assert_frame_equal(out,pd.read_csv(path),rtol=0,atol=1e-12)
        else:
            assert not path.exists();out.to_csv(path,index=False)
            pd.testing.assert_frame_equal(out,pd.read_csv(path),rtol=0,atol=1e-12)
        checks[kind] = dict(rows=len(out),all_rows_prefix_invariant=True,targets_ignored=True,
                            health_unchanged=True,sha256=c.sha(path))
    return checks


def install_read_guards():
    allowed_csv = {ROOT/name for name in ['train.csv','test_public.csv','test_private.csv',
        'sample_submission_public.csv','sample_submission_private.csv']}
    for kind in ['public','private']:
        allowed_csv.add(ROOT/f'clean_causal_stable_run/{kind}_causal_clean.csv')
        allowed_csv.add(OUT/f'{kind}_causal96_stage1.csv')
    allowed_models = {ROOT/'clean_causal_stable_run/Target_2.joblib',OUT/'Target_1.joblib'}
    original_read,original_load = pd.read_csv,joblib.load
    reads = dict(csv=[],models=[])
    def read(path,*args,**kwargs):
        resolved = Path(path).resolve()
        assert resolved in allowed_csv,f'Forbidden CSV: {path}'
        reads['csv'].append(str(resolved.relative_to(ROOT)))
        return original_read(path,*args,**kwargs)
    def load(path,*args,**kwargs):
        resolved = Path(path).resolve()
        assert resolved in allowed_models,f'Forbidden previous model: {path}'
        reads['models'].append(str(resolved.relative_to(ROOT)))
        return original_load(path,*args,**kwargs)
    pd.read_csv,joblib.load = read,load
    return reads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true')
    args = parser.parse_args()
    reads = install_read_guards()
    if args.verify:
        report = json.loads((OUT/'report.json').read_text())
        for path,expected in report['source_hashes'].items():
            assert c.sha(ROOT/path)==expected,path
        assert c.sha(OUT/'Target_1.joblib')==report['model_sha256']
        finance = joblib.load(OUT/'Target_1.joblib')
        health = joblib.load(ROOT/'clean_causal_stable_run/Target_2.joblib')
        assert export(finance,health,verify=True)==report['inference_checks']
        print('PASS: every Public/Private prediction reconstructed using causal prefixes; health preserved;',
              preservation(),'existing artifacts unchanged.',flush=True)
        print('ALLOWLIST READS',reads,flush=True)
        return
    assert not OUT.exists()
    OUT.mkdir()
    files = [ROOT/p for p in json.loads((ROOT/'clean_causal_run/existing_hashes.json').read_text())]
    for folder in ['clean_causal_run','clean_causal_stable_run']:
        files += [p for p in (ROOT/folder).rglob('*') if p.is_file()]
    files += [ROOT/p for p in ['clean_causal.py','clean_causal_stable.py','test_clean_causal.py']]
    snapshot = {str(p.relative_to(ROOT)):c.sha(p) for p in files}
    (OUT/'existing_hashes.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2))
    raw = c.read_data(ROOT/'train.csv')
    assert len(raw)==924
    configurations_frozen = configurations()
    report = dict(state='training',goal_mean_auc=.96,goal_achieved=False,
        configurations_frozen=configurations_frozen,selection_uses_public_labels=False,
        uses_future_prediction_inputs=False,future_training_auxiliary_responses=True,
        validation_role='reused_train_development_not_independent',folds=FOLDS)
    # Freeze the candidate list before opening any test observations.
    (OUT/'progress.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    baseline_predictions = []
    predictions = {spec['name']:[] for spec in configurations_frozen}
    diagnostics = {spec['name']:[] for spec in configurations_frozen}
    for end,size in FOLDS:
        prefix = raw.iloc[:end]
        base = c.fit_recipe(prefix,'Target_1',[c.specifications('Target_1')[0]])
        baseline_predictions.append(c.predict_recipe(base,raw.iloc[:end+size])[end:])
        for spec in configurations_frozen:
            b = fit_proxy(prefix,spec)
            assert b['auxiliary_observation_last_time'] < int(raw.Time_step.iloc[end])
            predictions[spec['name']].append(predict_proxy(b,raw.iloc[:end+size])[end:])
            diagnostics[spec['name']].append(b['warnings'])
        print('FOLD',end,size,'complete',flush=True)
    predictions['baseline'] = baseline_predictions
    for spec in configurations_frozen:
        name = spec['name']
        predictions[name+'_half'] = [(a+b)/2 for a,b in zip(predictions[name],baseline_predictions)]
    # Equal weighting across the two empirical US offsets is a predeclared recipe.
    observed = ['core_o1_c0.1_a1000','core_o2_c0.1_a1000']
    predictions['two_offsets_equal'] = [np.mean([predictions[name][i] for name in observed],axis=0) for i in range(len(FOLDS))]
    predictions['two_offsets_half'] = [(a+b)/2 for a,b in zip(predictions['two_offsets_equal'],baseline_predictions)]
    records = {};rows=[]
    for name,parts in predictions.items():
        scores = [float(roc_auc_score(raw.Target_1.iloc[e:e+s],p)) for (e,s),p in zip(FOLDS,parts)]
        records[name] = dict(scores=scores,mean150=float(np.mean(scores[:3])),
                             mean100=float(np.mean(scores[3:])),mean=float(np.mean(scores)))
        for (end,size),part in zip(FOLDS,parts):
            rows.extend(dict(recipe=name,fold_start=end+1,fold_size=size,Time_step=int(time),
                label=int(label),prediction=float(p)) for time,label,p in
                zip(raw.Time_step.iloc[end:end+size],raw.Target_1.iloc[end:end+size],part))
    winner = choose(records)
    selected_names = [] if winner=='baseline' else observed if winner.startswith('two_offsets') else [winner.removesuffix('_half')]
    selected = [spec for spec in configurations_frozen if spec['name'] in selected_names]
    finance = dict(winner=winner,components=[fit_proxy(raw,spec) for spec in selected],
        blend_with_baseline=winner.endswith('_half'),
        baseline=c.fit_recipe(raw,'Target_1',[c.specifications('Target_1')[0]]))
    path = OUT/'Target_1.joblib';joblib.dump(finance,path)
    health = joblib.load(ROOT/'clean_causal_stable_run/Target_2.joblib')
    clean_report = json.loads((ROOT/'clean_causal_stable_run/report.json').read_text())
    health_record = clean_report['targets']['Target_2']
    health_scores = health_record['records'][health_record['winner']]
    report.update(records=records,winner=winner,selected=selected,diagnostics=diagnostics,
        financial_development_auc=records[winner],baseline_financial_auc=records['baseline'],
        fixed_health_development_auc=health_scores,
        mean_development_auc={key:(records[winner][key]+health_scores[key])/2 for key in ['mean150','mean100','mean']},
        needed_financial_auc={key:2*.96-health_scores[key] for key in ['mean150','mean100','mean']},
        public_auc=None,private_auc=None,final_fit_last_time=921,
        model_sha256=c.sha(path),source_hashes={p:c.sha(ROOT/p) for p in source_paths()})
    pd.DataFrame(rows).to_csv(OUT/'development_predictions.csv',index=False)
    report['state'] = 'recipe_frozen_before_test_inference'
    (OUT/'progress.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('SELECTED',winner,records[winner],'MEAN',report['mean_development_auc'],flush=True)
    report['inference_checks'] = export(finance,health)
    report['existing_artifacts_preserved'] = preservation()
    report['allowed_reads'] = reads
    report['state'] = 'completed'
    report['goal_achieved'] = False
    report['note'] = ('Candidate selection uses only train development labels, not Public answers. Auxiliary future '
        'moves stay inside each fit prefix; the test-time remainder is forecast. Overlapping reused development '
        'windows do not establish independent accuracy. Neither Public nor Private AUC was measured. '
        'Current-row features require availability by the prediction cutoff. The 0.96 objective remains unproven.')
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('COMPLETE',report['mean_development_auc'],'0.96 goal not established.',flush=True)


if __name__=='__main__':
    main()

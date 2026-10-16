"""Compare nonlinear financial models against the frozen clean stage-one recipe."""
import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import clean_causal96_stage1 as first

c = first.c
ROOT = c.ROOT
OUT = ROOT/'clean_causal96_stage2_run'


def specs():
    candidates = [dict(name=f'{g}_svm_{C}',kind='svm',group=g,C=C)
                  for g in ['core','trajectory'] for C in [.03,.3,3]]
    candidates += [dict(name=f'core_nca_{k}',kind='nca',group='core',neighbors=k) for k in [45,90]]
    candidates += [dict(name=f'core_qda_{reg}',kind='qda',group='core',reg=reg) for reg in [.5,.9]]
    return candidates


def matrix(raw,group):
    x,groups = c.feature_frames(raw)
    cols = groups['core'] if group=='core' else [s for s in x if '_shock' in s or '_lagchange' in s or s.startswith('day')]
    return x[cols]


def estimator(spec):
    pre = [SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler()]
    if spec['kind']=='svm':
        return make_pipeline(*pre,SVC(C=spec['C'],gamma='scale'))
    if spec['kind']=='qda':
        return make_pipeline(*pre,QuadraticDiscriminantAnalysis(reg_param=spec['reg']))
    return make_pipeline(*pre,NeighborhoodComponentsAnalysis(n_components=4,init='pca',max_iter=80,random_state=42),
                         KNeighborsClassifier(n_neighbors=spec['neighbors'],weights='distance'))


def margin(model,x):
    return model.decision_function(x) if hasattr(model,'decision_function') else logit(model.predict_proba(x)[:,1].clip(1e-8,1-1e-8))


def fit(raw,spec):
    x = matrix(raw,spec['group'])
    model = estimator(spec)
    end = len(raw)-3
    with warnings.catch_warnings(record=True) as caught:
        model.fit(x.iloc[28:end],raw.Target_1.iloc[28:end])
    reference = margin(model,x.iloc[28:end])
    return dict(model=model,spec=spec,center=float(reference.mean()),scale=max(float(reference.std()),1e-8),
                fit_last_time=end,warnings=[str(w.message) for w in caught])


def predict_model(bundle,raw):
    z = margin(bundle['model'],matrix(raw,bundle['spec']['group']))
    return .5+np.arctan((z-bundle['center'])/bundle['scale'])/np.pi


def fit_prior(raw,recipe):
    return dict(winner=recipe['winner'],components=[first.fit_proxy(raw,s) for s in recipe['selected']],
        blend_with_baseline=recipe['winner'].endswith('_half'),
        baseline=c.fit_recipe(raw,'Target_1',[c.specifications('Target_1')[0]]))


def predict_finance(bundle,raw):
    prior = first.finance_predict(bundle['prior'],raw)
    if not bundle['selected']:
        return prior
    parts = [predict_model(b,raw) for b in bundle['selected']]
    p = np.mean(parts,axis=0)
    return (prior+p)/2 if bundle['half_blend'] else p


def guard_reads():
    original_read,original_load = pd.read_csv,joblib.load
    allowed = {ROOT/f'{prefix}{kind}.csv' for kind in ['public','private'] for prefix in ['test_','sample_submission_']}
    allowed.add(ROOT/'train.csv')
    for kind in ['public','private']:
        allowed.add(ROOT/f'clean_causal_stable_run/{kind}_causal_clean.csv')
        allowed.add(OUT/f'{kind}_causal96_stage2.csv')
    models = {OUT/'Target_1.joblib',ROOT/'clean_causal_stable_run/Target_2.joblib'}
    reads = {'csv':[],'models':[]}
    def read(path,*args,**kwargs):
        p = first.Path(path).resolve();assert p in allowed,f'Forbidden CSV: {p}'
        reads['csv'].append(str(p.relative_to(ROOT)))
        return original_read(path,*args,**kwargs)
    def load(path,*args,**kwargs):
        p = first.Path(path).resolve();assert p in models,f'Forbidden model: {p}'
        reads['models'].append(str(p.relative_to(ROOT)))
        return original_load(path,*args,**kwargs)
    pd.read_csv,joblib.load = read,load
    return reads


def preservation():
    snapshot = json.loads((OUT/'existing_hashes.json').read_text())
    for path,expected in snapshot.items():
        assert c.sha(ROOT/path)==expected,path
    return len(snapshot)


def export(finance,health,verify=False):
    train = c.read_data(ROOT/'train.csv');checks={}
    for kind in ['public','private']:
        data = c.read_data(ROOT/f'test_{kind}.csv').drop(columns=c.TARGETS,errors='ignore')
        sample = c.read_data(ROOT/f'sample_submission_{kind}.csv')
        raw = pd.concat([train,data],ignore_index=True)
        p1,p2 = predict_finance(finance,raw),c.predict_recipe(health,raw)
        # Entire prefix checks retain all historical observations but remove every later row.
        for index in range(924,len(raw)):
            prefix = raw.iloc[:index+1]
            np.testing.assert_allclose(predict_finance(finance,prefix)[-1],p1[index],rtol=0,atol=1e-12)
        changed = raw.copy();changed[c.TARGETS]=1
        np.testing.assert_allclose(predict_finance(finance,changed),p1,rtol=0,atol=1e-12)
        np.testing.assert_allclose(c.predict_recipe(health,changed),p2,rtol=0,atol=1e-12)
        out = sample.copy()
        for p,name in [(p1,'pred_1'),(p2,'pred_2')]:
            out[name]=pd.Series(p,index=raw.Time_step).loc[sample.Time_step].to_numpy()
        previous = pd.read_csv(ROOT/f'clean_causal_stable_run/{kind}_causal_clean.csv')
        np.testing.assert_allclose(out.pred_2,previous.pred_2,rtol=0,atol=1e-12)
        assert out.shape==(100 if kind=='public' else 200,3)
        assert list(out)==['Time_step','pred_1','pred_2'] and out.Time_step.equals(sample.Time_step)
        assert np.isfinite(out[['pred_1','pred_2']]).all().all()
        assert out[['pred_1','pred_2']].ge(0).all().all() and out[['pred_1','pred_2']].le(1).all().all()
        path = OUT/f'{kind}_causal96_stage2.csv'
        if verify:
            pd.testing.assert_frame_equal(out,pd.read_csv(path),rtol=0,atol=1e-12)
        else:
            assert not path.exists();out.to_csv(path,index=False)
            pd.testing.assert_frame_equal(out,pd.read_csv(path),rtol=0,atol=1e-12)
        checks[kind] = dict(rows=len(out),all_financial_prefixes_invariant=True,health_unchanged=True,
                            targets_ignored=True,sha256=c.sha(path))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify',action='store_true')
    args = parser.parse_args();reads=guard_reads()
    if args.verify:
        report = json.loads((OUT/'report.json').read_text())
        for path,expected in report['source_hashes'].items():assert c.sha(ROOT/path)==expected,path
        assert c.sha(OUT/'Target_1.joblib')==report['model_sha256']
        finance = joblib.load(OUT/'Target_1.joblib')
        health = joblib.load(ROOT/'clean_causal_stable_run/Target_2.joblib')
        assert export(finance,health,verify=True)==report['inference_checks']
        print('PASS: saved-model reconstruction, all financial prefixes, unchanged health,',preservation(),'existing artifacts preserved.')
        return
    assert not OUT.exists();OUT.mkdir()
    previous = json.loads((first.OUT/'existing_hashes.json').read_text())
    for p in first.OUT.rglob('*'):
        if p.is_file():previous[str(p.relative_to(ROOT))]=c.sha(p)
    previous['clean_causal96_stage1.py']=c.sha(ROOT/'clean_causal96_stage1.py')
    (OUT/'existing_hashes.json').write_text(json.dumps(previous,ensure_ascii=False,indent=2))
    raw = c.read_data(ROOT/'train.csv')
    prior = json.loads((first.OUT/'report.json').read_text())
    assert prior['state']=='completed' and not prior['selection_uses_public_labels']
    candidates=specs();records={};diagnostics={};baseline_parts=[]
    parts={s['name']:[] for s in candidates}
    for end,size in first.FOLDS:
        prefix=raw.iloc[:end]
        baseline=fit_prior(prefix,prior)
        baseline_parts.append(first.finance_predict(baseline,raw.iloc[:end+size])[end:])
        for spec in candidates:
            b=fit(prefix,spec);parts[spec['name']].append(predict_model(b,raw.iloc[:end+size])[end:])
            diagnostics.setdefault(spec['name'],[]).append(b['warnings'])
        print('FOLD',end,size,'complete',flush=True)
    parts['baseline']=baseline_parts
    for spec in candidates:
        name=spec['name'];parts[name+'_half']=[(a+b)/2 for a,b in zip(parts[name],baseline_parts)]
    family_names=['core_svm_0.3','core_nca_90','core_qda_0.9']
    parts['diverse_equal']=[np.mean([parts[n][i] for n in family_names],axis=0) for i in range(len(first.FOLDS))]
    parts['diverse_half']=[(a+b)/2 for a,b in zip(parts['diverse_equal'],baseline_parts)]
    rows=[]
    for name,predictions in parts.items():
        scores=[float(roc_auc_score(raw.Target_1.iloc[e:e+s],p)) for (e,s),p in zip(first.FOLDS,predictions)]
        records[name]=dict(scores=scores,mean150=float(np.mean(scores[:3])),mean100=float(np.mean(scores[3:])),mean=float(np.mean(scores)))
        for (end,size),pred in zip(first.FOLDS,predictions):
            rows.extend(dict(recipe=name,fold_start=end+1,fold_size=size,Time_step=int(t),label=int(y),prediction=float(p))
                        for t,y,p in zip(raw.Time_step.iloc[end:end+size],raw.Target_1.iloc[end:end+size],pred))
    winner=first.choose(records)
    names=[] if winner=='baseline' else family_names if winner.startswith('diverse') else [winner.removesuffix('_half')]
    selected=[s for s in candidates if s['name'] in names]
    finance=dict(prior=fit_prior(raw,prior),selected=[fit(raw,s) for s in selected],half_blend=winner.endswith('_half'))
    joblib.dump(finance,OUT/'Target_1.joblib')
    health=joblib.load(ROOT/'clean_causal_stable_run/Target_2.joblib')
    health_scores=prior['fixed_health_development_auc']
    paths=first.source_paths()+['clean_causal96_stage2.py','clean_causal96_stage1_run/report.json']
    report=dict(state='recipe_frozen_before_test_inference',goal_mean_auc=.96,goal_achieved=False,
        configurations_frozen=candidates,records=records,winner=winner,selected=selected,diagnostics=diagnostics,
        selection_uses_public_labels=False,uses_future_prediction_inputs=False,validation_role='reused_train_development',
        financial_development_auc=records[winner],prior_financial_auc=records['baseline'],
        fixed_health_development_auc=health_scores,
        mean_development_auc={key:(records[winner][key]+health_scores[key])/2 for key in ['mean150','mean100','mean']},
        public_auc=None,private_auc=None,model_sha256=c.sha(OUT/'Target_1.joblib'),
        source_hashes={p:c.sha(ROOT/p) for p in paths},final_fit_last_time=921)
    pd.DataFrame(rows).to_csv(OUT/'development_predictions.csv',index=False)
    (OUT/'progress.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('SELECTED',winner,records[winner],'MEAN',report['mean_development_auc'],flush=True)
    report['inference_checks']=export(finance,health)
    report['existing_artifacts_preserved']=preservation();report['allowed_reads']=reads
    report['state']='completed'
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('COMPLETE',report['mean_development_auc'],'0.96 goal not established.',flush=True)


if __name__=='__main__':
    main()
